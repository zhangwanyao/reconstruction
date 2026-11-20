"""Generate small expected values for the C++ unit tests (disk SE, Zhang-Suen thinning, components, minAreaRect).
Usage: python -I unit_expect.py <out_txt>"""
import sys
import numpy as np
import cv2

out = []


def show(name, a):
    out.append(name)
    for row in np.asarray(a).astype(int):
        out.append("  " + "".join(str(int(v)) for v in row))


for r in (1, 2, 3, 4):
    show("disk(%d)" % r, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

# Zhang-Suen on a 30x5 bar with 3 px margin
img = np.zeros((11, 36), np.uint8); img[3:8, 3:33] = 1
sk = cv2.ximgproc.thinning(img * 255) > 0
rows, cols = np.nonzero(sk)
out.append("bar30x5 skeleton rows=%s cols=%d..%d n=%d" % (sorted(set(rows.tolist())), cols.min(), cols.max(), len(rows)))
show("bar30x5 skeleton", sk[2:9])

# plus shape: two 5-wide bars crossing
img = np.zeros((41, 41), np.uint8); img[18:23, 3:38] = 1; img[3:38, 18:23] = 1
sk = (cv2.ximgproc.thinning(img * 255) > 0).astype(np.uint8)
ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
br = []
for r, c in zip(*np.nonzero(sk)):
    v = [sk[r + dr, c + dc] for dr, dc in ring]
    cn = sum(1 for i in range(8) if v[i] == 0 and v[(i + 1) % 8] == 1)
    if cn >= 3:
        br.append((int(r), int(c), cn))
out.append("plus skeleton pixels=%d branch(cn>=3)=%s" % (int(sk.sum()), br))
show("plus skeleton (rows 14..26, cols 14..26)", sk[14:27, 14:27])

# thinned thick diagonal bar: neighbour-count>=3 vs crossing number (v1 cut such lines into many pieces)
for ang_deg in (20, 35, 60):
    img = np.zeros((200, 200), np.uint8)
    a = np.radians(ang_deg)
    p0 = (20, 20); p1 = (int(20 + 160 * np.cos(a)), int(20 + 160 * np.sin(a)))
    cv2.line(img, p0, p1, 1, 7)
    sk = (cv2.ximgproc.thinning(img * 255) > 0).astype(np.uint8)
    cnt3 = 0; cn3 = 0
    for r, c in zip(*np.nonzero(sk)):
        v = [sk[r + dr, c + dc] for dr, dc in ring]
        if sum(v) >= 3:
            cnt3 += 1
        if sum(1 for i in range(8) if v[i] == 0 and v[(i + 1) % 8] == 1) >= 3:
            cn3 += 1
    out.append("thinned 7px bar at %d deg: skeleton pixels=%d neighbours>=3: %d crossing>=3: %d" % (ang_deg, int(sk.sum()), cnt3, cn3))

# same, but the bar rasterised as 'pixel centre within 3.5 px of the segment' (what the C++ test will do)
for ang_deg in (20, 35, 60):
    a = np.radians(ang_deg); p0 = np.array([20.0, 20.0]); p1 = p0 + 160 * np.array([np.cos(a), np.sin(a)])
    rr, cc = np.mgrid[0:200, 0:200]; q = np.stack([cc, rr], -1).astype(float)
    d = p1 - p0; t = np.clip(((q - p0) @ d) / (d @ d), 0, 1)
    dist = np.linalg.norm(q - (p0 + t[..., None] * d), axis=-1)
    img = (dist <= 3.5).astype(np.uint8)
    sk = (cv2.ximgproc.thinning(img * 255) > 0).astype(np.uint8)
    cnt3 = 0; cn3 = 0
    for r, c in zip(*np.nonzero(sk)):
        v = [sk[r + dr, c + dc] for dr, dc in ring]
        if sum(v) >= 3:
            cnt3 += 1
        if sum(1 for i in range(8) if v[i] == 0 and v[(i + 1) % 8] == 1) >= 3:
            cn3 += 1
    out.append("dist<=3.5 bar at %d deg: bar pixels=%d skeleton pixels=%d neighbours>=3: %d crossing>=3: %d" % (ang_deg, int(img.sum()), int(sk.sum()), cnt3, cn3))

# outer contour of a 6x4 rectangle, fill holes of a ring
img = np.zeros((20, 20), np.uint8); img[5:9, 4:10] = 1
cs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
out.append("contour of 6x4 rect: %d points, area(shoelace)=%.1f" % (len(cs[0]), cv2.contourArea(cs[0])))

# connected components: 4 vs 8
img = np.array([[1, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0]], np.uint8)
n4, l4 = cv2.connectedComponents(img, connectivity=4)
n8, l8 = cv2.connectedComponents(img, connectivity=8)
show("cc input", img)
show("cc labels 4-conn (n=%d incl. background)" % n4, l4)
show("cc labels 8-conn (n=%d incl. background)" % n8, l8)

# EDT vs OpenCV 5x5 at a few offsets from a single zero pixel
img = np.ones((41, 41), np.uint8); img[20, 20] = 0
dt5 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
dtp = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
for dr, dc in [(0, 1), (1, 1), (0, 10), (7, 7), (3, 12), (0, 20)]:
    out.append("dt(%d,%d): exact=%.4f cv5x5=%.4f cvprecise=%.4f" % (dr, dc, np.hypot(dr, dc), dt5[20 + dr, 20 + dc], dtp[20 + dr, 20 + dc]))

# minAreaRect of a rotated 200x50 rectangle at 30 deg
th = np.radians(30); ex = np.array([np.cos(th), np.sin(th)]); ey = np.array([-np.sin(th), np.cos(th)])
g = np.array([[u, v] for u in np.linspace(-100, 100, 41) for v in np.linspace(-25, 25, 11)])
pts = (g[:, :1] * ex + g[:, 1:] * ey + [500, 300]).astype(np.float32)
(cx, cy), (w, h), ang = cv2.minAreaRect(pts)
long_ang = ang if w >= h else ang + 90
out.append("minAreaRect: center=(%.3f,%.3f) size=(%.3f,%.3f) angle=%.3f long_side_angle=%.3f" % (cx, cy, w, h, ang, long_ang))

open(sys.argv[1], "w", encoding="utf-8").write("\n".join(out) + "\n")
print("ok", len(out))
