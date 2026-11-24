"""Build the graphical abstract as hand-authored SVG (1328 x 531 logical px).

Panels (left -> right): shield microstructures -> plane-wave physics model -> percolation result.
The percolation curve is computed from the McLachlan GEM equation with the paper's Table 2 parameters.
"""
import math, random, sys
from pathlib import Path

OUT = sys.argv[1]
W, H = 1328, 531
INK = "#1a1a1a"; GREY = "#6b6b6b"; LIGHT = "#d9d9d9"; PALE = "#f1f1f1"
BLUE = "#1f77b4"; RED = "#d62728"          # same convention as paper Fig. 2
SANS = "Arial, Helvetica, sans-serif"; SERIF = "'Times New Roman', Times, serif"

# ---------- physics: McLachlan GEM (paper Sec. 2.3 / Table 2) ----------
sf, sm, fc, tt = 1e5, 1e-10, 0.01, 2.0
A = (1 - fc) / fc
def gem(f):
    a, b = sm ** (1/tt), sf ** (1/tt)
    def F(ls):
        s = (10 ** ls) ** (1/tt)
        return (1-f)*(a-s)/(a+A*s) + f*(b-s)/(b+A*s)
    lo, hi = math.log10(sm), math.log10(sf)
    for _ in range(200):
        mid = (lo+hi)/2
        if F(mid) > 0: lo = mid
        else: hi = mid
    return 10 ** ((lo+hi)/2)
def lin(f): return f*sf + (1-f)*sm

if "--check" in sys.argv:
    for f in (0.005, 0.010, 0.012, 0.020, 0.050, 0.100):
        print(f"f={f:.3f}  linear={lin(f):.2e}  GEM={gem(f):.2e}")
    sys.exit()

el = []
def add(s): el.append(s)
def text(x, y, s, size=22, weight="normal", anchor="start", fill=INK, family=SANS, style="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" text-anchor="{anchor}" fill="{fill}">{s}</text>')

# ---------- layout ----------
P1x, P1w = 30, 372
P2x, P2w = 452, 420
P3x, P3w = 922, 376
TY = 50
for x, w, t in ((P1x, P1w, "Shield microstructure"), (P2x, P2w, "Homogeneous-slab model"),
                (P3x, P3w, "Measured against predicted")):
    text(x + w/2, TY, t, 25, "bold", "middle")

def arrow(x0, x1, y):
    add(f'<line x1="{x0}" y1="{y}" x2="{x1-12}" y2="{y}" stroke="{GREY}" stroke-width="3.5"/>')
    add(f'<path d="M{x1-18},{y-11} L{x1},{y} L{x1-18},{y+11} Z" fill="{GREY}"/>')
arrow(P1x + P1w + 10, P2x - 6, 229)
arrow(P2x + P2w + 6, P3x - 8, 229)

# ---------- panel 1: four cross-sections (wave travels left->right) ----------
rnd = random.Random(7)
cw, ch, gx, gy = 176, 140, 20, 58
cells = [(P1x, 78), (P1x + cw + gx, 78), (P1x, 78 + ch + gy), (P1x + cw + gx, 78 + ch + gy)]
labels = ["Metal film", "Layered film", "Composite", "Foam / aerogel"]
clip_id = 0
for (cx, cy), lab in zip(cells, labels):
    clip_id += 1
    add(f'<clipPath id="c{clip_id}"><rect x="{cx}" y="{cy}" width="{cw}" height="{ch}"/></clipPath>')
    g = [f'<g clip-path="url(#c{clip_id})">']
    if lab == "Metal film":
        g.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{LIGHT}"/>')
        seeds = [(cx + rnd.uniform(0, cw), cy + rnd.uniform(0, ch)) for _ in range(16)]
        for i, p in enumerate(seeds):
            poly = [(cx-1, cy-1), (cx+cw+1, cy-1), (cx+cw+1, cy+ch+1), (cx-1, cy+ch+1)]
            for j, q in enumerate(seeds):
                if i == j: continue
                mx, my = (p[0]+q[0])/2, (p[1]+q[1])/2; nx, ny = q[0]-p[0], q[1]-p[1]
                keep = lambda v: (v[0]-mx)*nx + (v[1]-my)*ny <= 0
                out = []
                for k in range(len(poly)):
                    a, b = poly[k], poly[(k+1) % len(poly)]
                    if keep(a): out.append(a)
                    if keep(a) != keep(b):
                        da = (a[0]-mx)*nx + (a[1]-my)*ny; db = (b[0]-mx)*nx + (b[1]-my)*ny
                        t = da/(da-db); out.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
                poly = out
            shade = rnd.choice(["#cfcfcf", "#d9d9d9", "#e2e2e2", "#c8c8c8"])
            g.append('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in poly) + f'" fill="{shade}" stroke="{GREY}" stroke-width="1.3"/>')
    elif lab == "Layered film":
        g.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{PALE}"/>')
        x = cx + 30
        while x < cx + cw - 30:
            wdt = rnd.uniform(5, 8)
            y0 = cy + rnd.uniform(-6, 10); y1 = cy + ch - rnd.uniform(-6, 10)
            g.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{wdt:.1f}" height="{y1-y0:.1f}" fill="{GREY}"/>')
            x += wdt + rnd.uniform(3.5, 6)
    elif lab == "Composite":
        g.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{PALE}"/>')
        for _ in range(34):
            x, y = cx + rnd.uniform(-10, cw+10), cy + rnd.uniform(-10, ch+10)
            a = rnd.uniform(0, math.pi); L = rnd.uniform(26, 44)
            g.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+L*math.cos(a):.1f}" y2="{y+L*math.sin(a):.1f}" stroke="{INK}" stroke-width="2.6" stroke-linecap="round"/>')
    else:  # Foam / aerogel: conducting walls around voids
        g.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="{GREY}"/>')
        placed = []
        for _ in range(900):
            r = rnd.uniform(9, 24); x = cx + rnd.uniform(-8, cw + 8); y = cy + rnd.uniform(-8, ch + 8)
            if all(math.hypot(x - a, y - b) > r + c + 3 for a, b, c in placed):
                placed.append((x, y, r))
        for x, y, r in placed:
            g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#ffffff"/>')
    g.append('</g>')
    add("".join(g))
    add(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
    text(cx + cw/2, cy + ch + 30, lab, 22, anchor="middle")

# ---------- panel 2: plane wave on a slab ----------
sx, sw, sy, sh = P2x + 190, 64, 84, 290
cy = sy + sh/2
add(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" fill="{LIGHT}" stroke="{INK}" stroke-width="1.5"/>')
def wave(x0, x1, amp0, amp1, color, width=3, cycles=2.5):
    n = 120; pts = []
    for i in range(n+1):
        u = i/n; x = x0 + (x1-x0)*u
        amp = amp0 * (amp1/amp0) ** u
        pts.append(f"{x:.1f},{cy - amp*math.sin(2*math.pi*cycles*u):.1f}")
    add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{width}"/>')
wave(P2x + 8, sx - 4, 46, 46, INK, cycles=2.5)
wave(sx, sx + sw, 46, 10, RED, 2.4, cycles=0.6)         # attenuation inside the shield
wave(sx + sw + 4, P2x + P2w - 8, 10, 10, INK, cycles=2.0)
# reflected arrow
rx0, ry0 = sx - 10, sy + 26
add(f'<path d="M{rx0},{ry0} Q{sx-80},{sy-8} {P2x+40},{sy+8}" fill="none" stroke="{GREY}" stroke-width="3"/>')
add(f'<path d="M{P2x+44},{sy-2} L{P2x+28},{sy+10} L{P2x+46},{sy+18} Z" fill="{GREY}"/>')
text(P2x + 14, cy + 82, "Incident", 21, fill=INK)
text(sx - 98, sy + 52, "Reflected", 21, fill=GREY, anchor="middle")
text(sx + sw/2, sy + sh + 28, "Absorbed", 21, fill=RED, anchor="middle")
text(P2x + P2w - 8, cy + 82, "Transmitted", 21, anchor="end")
# governing relation + modules
text(P2x + P2w/2, 446, '<tspan font-style="italic">SE</tspan> = <tspan font-style="italic">R</tspan> + <tspan font-style="italic">A</tspan> + <tspan font-style="italic">B</tspan>', 25, family=SERIF, anchor="middle")
text(P2x + P2w/2, 478, "exact prediction from conductivity and thickness", 19, fill=GREY, anchor="middle")

# ---------- panel 3: residual by material group ----------
import csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_analysis import load  # noqa: E402
rows = load()
npapers = len({r["paper"] for r in rows})
short = ["Metal films", "Metal meshes", "Dense 2D films", "Porous 2D films", "Solid composites", "Foams, aerogels"]
ax0, ax1, ay0, ay1 = P3x + 150, P3x + P3w - 8, 96, 400
gmin, gmax = -60, 60
def GX(v): return ax0 + (v - gmin) / (gmax - gmin) * (ax1 - ax0)
add(f'<rect x="{GX(-10)}" y="{ay0-14}" width="{GX(10)-GX(-10)}" height="{ay1-ay0+28}" fill="{PALE}"/>')
add(f'<line x1="{GX(0)}" y1="{ay0-14}" x2="{GX(0)}" y2="{ay1+14}" stroke="{INK}" stroke-width="1.5"/>')
add(f'<line x1="{ax0}" y1="{ay1+14}" x2="{ax1}" y2="{ay1+14}" stroke="{INK}" stroke-width="1.5"/>')
for v in (-40, 0, 40):
    add(f'<line x1="{GX(v)}" y1="{ay1+14}" x2="{GX(v)}" y2="{ay1+20}" stroke="{INK}" stroke-width="1.5"/>')
    text(GX(v), ay1 + 42, ("+" if v > 0 else "") + str(v).replace("-", "−"), 19, anchor="middle")
text((ax0 + ax1) / 2, ay1 + 72, "Measured − predicted (dB)", 21, anchor="middle")
jit = random.Random(11)
for i, lab in enumerate(short):
    y = ay0 + (i + 0.5) * (ay1 - ay0) / len(short)
    text(ax0 - 12, y + 7, lab, 20, anchor="end")
    for r in rows:
        if r["group"] == i:
            colour = BLUE if r["sigma"] >= 1e4 else RED
            v = max(gmin, min(gmax, r["res"]))
            cx, cy = GX(v), y + jit.uniform(-9, 9)
            if r["quality"] == "C":
                add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.9" fill="white" stroke="{colour}" stroke-width="1.6"/>')
            else:
                add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" fill="{colour}" fill-opacity="0.85"/>')
lx = P3x + 30
add(f'<circle cx="{lx}" cy="72" r="4.6" fill="{BLUE}"/>')
text(lx + 10, 78, "above 10⁴ S/m", 17, fill=GREY)
add(f'<circle cx="{lx + 170}" cy="72" r="4.6" fill="{RED}"/>')
text(lx + 180, 78, "below 10⁴ S/m", 17, fill=GREY)

# ---------- footer ----------
text(W/2, H - 14, f"{len(rows)} source-verified measurements from {npapers} papers, 10 MHz – 40 GHz", 19, fill=GREY, anchor="middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(e for e in el if e) + '</svg>')
open(OUT, "w").write(svg)
html = ('<!doctype html><html><head><meta charset="utf-8"><style>@page{size:1328px 531px;margin:0}'
        'html,body{margin:0;width:1328px;height:531px;overflow:hidden;background:#fff}</style></head><body>' + svg + '</body></html>')
open(OUT.replace(".svg", ".html"), "w").write(html)
print("ok")
