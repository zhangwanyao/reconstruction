"""Print PCD headers (text part only). Usage: python -I pcd_headers.py <out_txt> <pcd>..."""
import sys

out = []
for p in sys.argv[2:]:
    with open(p, 'rb') as fh:
        raw = fh.read(4096)
    hdr = []
    for ln in raw.split(b'\n'):
        t = ln.decode('ascii', 'replace').strip()
        hdr.append(t)
        if t.startswith('DATA'):
            break
    out.append('== %s' % p.replace('\\', '/').split('/')[-1])
    out += ['   ' + h for h in hdr if h and not h.startswith('#')]
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
