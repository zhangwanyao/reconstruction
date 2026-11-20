"""One-line-per-dataset summary of refviz/*_parts.json for the spec. Usage: python -I summarize_refs.py <cmp_dir> <out_txt>"""
import json, os, sys

C, OUT = sys.argv[1], sys.argv[2]
lines = []
for n in ['F168', 'F173', 'F455', 'WP', 'SS30', 'SS34', 'WP169', 'WP338']:
    p = os.path.join(C, 'refviz', n + '_parts.json')
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding='utf-8'))
    parts = d.get('parts', [])
    kinds, roles = {}, {}
    for q in parts:
        kinds[q.get('kind')] = kinds.get(q.get('kind'), 0) + 1
        roles[q.get('role')] = roles.get(q.get('role'), 0) + 1
    ts = sorted(round(float(q['t']), 1) for q in parts if q.get('t') is not None)
    lines.append('%-6s base_top=%s n=%d kinds=%s roles=%s t=%s' % (n, d.get('base_top'), len(parts), kinds, roles, ts))
    for q in parts:
        lines.append('   #%s kind=%s role=%s t=%s w=%s L=%s faces=%s holes=%s z=%s covered=%s' % (
            q.get('i'), q.get('kind'), q.get('role'), q.get('t'), q.get('w'), q.get('L'), q.get('faces'),
            q.get('inner_loops'), q.get('z'), q.get('covered')))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('ok', len(lines))
