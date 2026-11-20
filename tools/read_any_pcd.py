import struct, numpy as np

def _lzf(src, n):
    out = bytearray(n); ip = op = 0
    while ip < len(src):
        c = src[ip]; ip += 1
        if c < 32:
            l = c + 1; out[op:op + l] = src[ip:ip + l]; ip += l; op += l
        else:
            l = c >> 5; ref = op - ((c & 31) << 8) - 1
            if l == 7: l += src[ip]; ip += 1
            ref -= src[ip]; ip += 1; l += 2
            for _ in range(l): out[op] = out[ref]; op += 1; ref += 1
    return bytes(out)

def read_xyz(path):
    data = open(path, 'rb').read(); h = {}; pos = 0
    while True:
        e = data.index(b'\n', pos); ln = data[pos:e].decode('ascii', 'ignore').strip(); pos = e + 1
        if not ln or ln.startswith('#'): continue
        k, *v = ln.split(); h[k] = v
        if k == 'DATA': break
    f = h['FIELDS']; N = int(h['POINTS'][0]); sz = [int(s) for s in h['SIZE']]
    code = {'F': 'f', 'U': 'u', 'I': 'i'}
    dt = np.dtype([(n, '<' + code[t] + str(s)) for n, t, s in zip(f, h['TYPE'], sz)])
    if h['DATA'][0] == 'binary':
        a = np.frombuffer(data, dt, N, pos); return np.c_[a['x'], a['y'], a['z']].astype(np.float64)
    if h['DATA'][0] == 'binary_compressed':
        comp, raw = struct.unpack('<II', data[pos:pos + 8]); buf = _lzf(data[pos + 8:pos + 8 + comp], raw)
        cols = {}; off = 0
        for n, t, s in zip(f, h['TYPE'], sz):
            cols[n] = np.frombuffer(buf, '<' + code[t] + str(s), N, off); off += N * s
        return np.c_[cols['x'], cols['y'], cols['z']].astype(np.float64)
    raise ValueError('ascii PCD not supported')
