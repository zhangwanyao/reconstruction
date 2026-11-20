"""比较两个 parts.json 的键结构(任务书 11.3)。
用法: python -I cmp_keys.py <Python 原型的 parts.json> <C++ 的 parts.json>
  - 第一个文件里有、第二个里没有的键: 缺键(错误)。
  - 第二个文件多出来的键: 列出来(允许, 11.3 说 C++ 可以加键)。
  - 顶层键的相对顺序和第一个文件不一样: 报错(加的键要放在已有键后面)。
  solids 按 kind 分组比较(只有一边有的 kind 只提示, 不算缺键); heightmap.occ_ratio 的子键是试过的像素, 不比。
退出码 0 = 没有缺键、顺序也对。
"""
import json, sys


def paths(x, p, out, empty):
    if isinstance(x, dict):
        for k, v in x.items():
            q = p + "/" + k
            out.add(q)
            if k != "occ_ratio":
                paths(v, q, out, empty)
    elif isinstance(x, list) and not x:
        empty.add(p)
    elif isinstance(x, list) and all(isinstance(e, dict) for e in x):
        for e in x:
            paths(e, p + ("[kind=%s]" % e["kind"] if "kind" in e else "[]"), out, empty)
    return out


a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
ea, eb = set(), set()
pa, pb = paths(a, "", set(), ea), paths(b, "", set(), eb)
kinds = lambda d: {s.get("kind") for s in d.get("solids", []) if isinstance(s, dict)}
only_a, only_b = sorted(kinds(a) - kinds(b), key=str), sorted(kinds(b) - kinds(a), key=str)
skip = lambda p: (any("[kind=%s]" % k in p for k in only_a + only_b)       # 只有一边有的 kind 不逐键比
                  or any(p.startswith(e + "[") for e in ea | eb))           # 一边是空列表, 没法比元素的键
miss = sorted(p for p in pa - pb if not skip(p))
extra = sorted(p for p in pb - pa if not skip(p))
ka = [k for k in a if k in b]
kb = [k for k in b if k in a]
order_ok = ka == kb
for m in miss:
    print("MISSING", m)
for e in extra:
    print("extra  ", e)
if only_a or only_b:
    print("note    solids kind only in python: %s; only in c++: %s (not a key error)" % (only_a, only_b))
for e in sorted(ea ^ eb):
    print("note    list %s is empty in %s (its element keys not compared)" % (e, "c++" if e in eb else "python"))
if not order_ok:
    print("ORDER   python:", ka)
    print("        c++   :", kb)
print("missing=%d extra=%d order=%s" % (len(miss), len(extra), "ok" if order_ok else "WRONG"))
sys.exit(0 if not miss and order_ok else 1)
