"""Collect eval numbers for the spec tables. Usage: python -I summarize_evals.py <cmp_dir> <out_txt>"""
import json, os, sys

C = sys.argv[1]
OUT = sys.argv[2]
names = ['F168', 'F173', 'F455', 'WP', 'SS30', 'SS34', 'WP169', 'WP338']
tags = {
    'A': ['{n}_A', 'A_{n}'], 'A_SS30new': ['A_SS30new'], 'B': ['{n}_B'], 'B20': ['B20_{n}'],
    'v1': ['{n}_proto', 'proto_{n}'], 'p2': ['p2_{n}'], 'p2b': ['p2b_{n}'], 'p2c': ['p2c_{n}'],
}


def load(tag, n):
    for pat in tags[tag]:
        p = os.path.join(C, 'eval', pat.format(n=n) + '.json')
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding='utf-8'))
            except Exception:
                return None
    return None


lines = []
lines.append('== ref vs cloud (how well the reference STEP itself matches the cloud)')
for n in names:
    r = load('p2c', n) or {}
    ref = r.get('ref')
    if ref is None:
        p = os.path.join(C, 'eval', n + '_ref.json')
        if os.path.exists(p):
            ref = json.load(open(p, encoding='utf-8')).get('ref')
    if ref:
        cm = ref['cloud_to_model']; cov = ref['model_covered_by_cloud']
        lines.append('%-6s ref units=%s faces=%s cloud->ref w1=%.3f w3=%.3f w10=%.3f p50=%s p95=%s | ref covered all=%.3f up=%s side=%s' % (
            n, ref['units'], ref['faces'], cm['within_1mm'], cm['within_3mm'], cm['within_10mm'], cm['p50'], cm['p95'],
            cov['all'], cov['up_faces'], cov['side_faces']))
lines.append('')
lines.append('== out vs ref per method (recall/precision %, found, valid, F1, cloud->out within3, out faces, out units)')
for n in names:
    for tag in ['A', 'A_SS30new', 'B', 'B20', 'v1', 'p2', 'p2b', 'p2c']:
        if tag == 'A_SS30new' and n != 'SS30':
            continue
        r = load(tag, n)
        if not r or 'out_vs_ref' not in r:
            continue
        o = r['out_vs_ref']
        if 'error' in o:
            lines.append('%-6s %-9s error %s' % (n, tag, o['error'])); continue
        rc, pr = o['surface_recall'], o['surface_precision']
        f1 = 2 * rc * pr / max(rc + pr, 1e-9)
        out = r.get('out', {})
        cm = out.get('cloud_to_model', {})
        lines.append('%-6s %-9s R=%5.1f P=%5.1f F1=%5.1f found=%2d/%-2d valid=%2d/%-2d cloud->out w3=%s faces=%s units=%s cloud_pts=%s' % (
            n, tag, rc * 100, pr * 100, f1 * 100, o['ref_units_found'], o['ref_units'], o['out_units_valid'], o['out_units'],
            cm.get('within_3mm'), out.get('faces'), out.get('units'), r.get('cloud_points')))
    lines.append('')
lines.append('== p2c parts summary')
for n in names:
    p = os.path.join(C, 'proto2c', n, 'parts.json')
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding='utf-8'))
    kinds = {}
    for s in d['solids']:
        kinds[s['kind']] = kinds.get(s['kind'], 0) + 1
    hl = d.get('h_levels', [])
    lines.append('%-6s has_base=%s base_t=%s up_tilt=%s px=%s occ_ratio=%s rho_top=%s fill_rect=%s kinds=%s total=%ss' % (
        n, d.get('has_base'), d.get('base_t_used'), d.get('up_tilt_from_sensor_deg'), d['heightmap']['px'], d['heightmap']['occ_ratio'],
        d['heightmap']['rho_top'], d.get('base_fill_rect'), kinds, d.get('total_s')))
    if 'base_plane' in d:
        lines.append('       base_plane=%s' % json.dumps(d['base_plane']))
    for c in hl:
        lines.append('       level %s' % json.dumps(c))
    for L in d.get('lean', []):
        lines.append('       lean cand L=%s lean=%s inl=%s cand=%s zspan=%s rms=%s f2c=%s rel_dens=%s ok=%s dense=%s' % (
            L['L'], L['lean'], L['inl'], L['cand'], L['zspan'], L['rms'], L['face_to_cap'], L['rel_dens'], L['ok'], L['dense']))
    for s in d['solids']:
        if s['kind'].startswith('lean'):
            lines.append('       LEAN SOLID %s' % json.dumps(s))
lines.append('')
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('ok', len(lines))
