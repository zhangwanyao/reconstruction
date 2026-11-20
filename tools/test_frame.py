"""只测 detect_frame: 判底板 / 上方向精修 / 底板底面配对。用法: python -I test_frame.py <pcd> [<pcd> ...]"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_any_pcd import read_xyz
from protolib import detect_frame

for p in sys.argv[1:]:
    t0 = time.time()
    P = read_xyz(p); P = P[np.isfinite(P).all(1)]
    F = detect_frame(P)
    L = F["log"]
    print("==", os.path.basename(p), "has_base", F["has_base"], "base_t", F["base_t"], "up_tilt", L["up_tilt_from_sensor_deg"],
          "cells", L["grid_cells"], "flat", L["flat_cells"], "time %.1fs" % (time.time() - t0))
    print("   base_plane", json.dumps(L.get("base_plane")))
    for c in L["h_levels"]:
        print("   ", json.dumps(c))
