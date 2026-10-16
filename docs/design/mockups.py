"""Generates the node-UI concept mockups: python3 docs/design/mockups.py docs/design
Sigma values: turbo/quality, read from docs/readme/schedule-dark.svg (y=70 -> 1, y=316 -> 0).
Mockups only - no widget here is wired to anything; some widgets are omitted for space."""
import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1]

YS = [70, 73.7, 77.7, 82.1, 86.7, 91.6, 96.9, 102.7, 109.0, 115.8, 123.3, 131.5, 316, 156.1, 196.1, 236.1, 276, 316]
SIG = [(316 - y) / 246 for y in YS]

THEMES = {
    "dark": dict(canvas="#151819", dot="#262b2d", body="#1e2223", border="#3b4144", widget="#2b3032",
                 text="#eceee9", muted="#b3b9b5", faint="#8a918e", line="#24292b",
                 comp="#3987e5", ident="#199e70", tex="#d95926", plunge="#b3b9b5", restart="#4a2a20",
                 h_comp="#1f3f66", h_ident="#14503c", h_tex="#6e2f14", h_neutral="#2b3032",
                 htext="#eceee9", shadow="#000000", warn="#e0a33a"),
    "light": dict(canvas="#eef0ec", dot="#d6dad4", body="#f8f9f6", border="#c9cec7", widget="#e8ebe6",
                  text="#15181a", muted="#4b5155", faint="#6f7578", line="#e1e5df",
                  comp="#2a78d6", ident="#13845d", tex="#c24d1f", plunge="#6f7578", restart="#f3e3de",
                  h_comp="#dbe7f7", h_ident="#d5efe4", h_tex="#f7e0d5", h_neutral="#e1e5df",
                  htext="#15181a", shadow="#5a6260", warn="#9a6200"),
}
SOCK = {"MODEL": "#b39ddb", "CLIP": "#ffd500", "CONDITIONING": "#ffa931", "LATENT": "#ff9cf9", "VAE": "#ff6e6e",
        "IMAGE": "#64b5f6", "SIGMAS": "#cddc39", "MASK": "#81c784", "STRING": "#9e9e9e", "INT": "#9e9e9e"}
FONT = "font-family=\"Inter,system-ui,-apple-system,'Segoe UI',sans-serif\""


def t(x, y, s, size=11.5, fill="#fff", anchor="start", weight=400, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT} {extra}>{escape(s)}</text>')


class Node:
    W_ROW = 24

    def __init__(self, T, x, y, w, title, head, badge=None):
        self.T, self.x, self.y, self.w, self.title, self.head, self.badge = T, x, y, w, title, head, badge
        self.parts, self.cy = [], y + 30 + 8

    def socket(self, name, typ, side="in", optional=False):
        T = self.T
        cx = self.x if side == "in" else self.x + self.w
        col = SOCK.get(typ, "#9e9e9e")
        if optional:
            self.parts.append(f'<circle cx="{cx}" cy="{self.cy + 6}" r="4.5" fill="{T["body"]}" stroke="{col}" stroke-width="1.6"/>')
        else:
            self.parts.append(f'<circle cx="{cx}" cy="{self.cy + 6}" r="5" fill="{col}"/>')
        if side == "in":
            self.parts.append(t(self.x + 12, self.cy + 10, name, 11, T["muted"]))
        else:
            self.parts.append(t(self.x + self.w - 12, self.cy + 10, name, 11, T["muted"], "end"))
        return self

    def sockets(self, ins, outs=()):
        n = max(len(ins), len(outs))
        for i in range(n):
            if i < len(ins):
                nm, ty, *o = ins[i]
                self.socket(nm, ty, "in", bool(o))
            if i < len(outs):
                nm, ty = outs[i]
                self.socket(nm, ty, "out")
            self.cy += 18
        self.cy += 4
        return self

    def widget(self, label, value, dim=False, changed=None, combo=False):
        T = self.T
        op = ' opacity="0.42"' if dim else ""
        x0, w = self.x + 10, self.w - 20
        g = [f'<g{op}>', f'<rect x="{x0}" y="{self.cy}" width="{w}" height="20" rx="10" fill="{T["widget"]}"/>']
        lx = x0 + 12
        if changed:
            g.append(f'<circle cx="{x0 + 9}" cy="{self.cy + 10}" r="3" fill="{T[changed]}"/>')
            lx = x0 + 17
        g.append(t(lx, self.cy + 14, label, 11, T["muted"]))
        g.append(t(x0 + w - (22 if combo else 12), self.cy + 14, value, 11, T["text"], "end"))
        if combo:
            g.append(t(x0 + w - 10, self.cy + 14, "▾", 10, T["faint"], "end"))
        g.append("</g>")
        self.parts.append("".join(g))
        self.cy += self.W_ROW
        return self

    def section(self, label, note="", open_=False, changed=None):
        T = self.T
        x0 = self.x + 10
        self.parts.append(f'<line x1="{x0}" y1="{self.cy + 2}" x2="{self.x + self.w - 10}" y2="{self.cy + 2}" stroke="{T["line"]}"/>')
        self.parts.append(t(x0 + 2, self.cy + 17, ("▾ " if open_ else "▸ ") + label, 11.5, T["text"], weight=600))
        if note:
            if changed:
                self.parts.append(f'<circle cx="{self.x + self.w - 14 - 6.3 * len(note) - 8}" cy="{self.cy + 13}" r="3" fill="{T[changed]}"/>')
            self.parts.append(t(self.x + self.w - 12, self.cy + 17, note, 10.5, T["faint"], "end"))
        self.cy += 24
        return self

    def note(self, s, color="faint", size=10.5):
        self.parts.append(t(self.x + 12, self.cy + 11, s, size, self.T[color]))
        self.cy += 16
        return self

    def chip(self, s, color="warn"):
        T = self.T
        w = 7.0 * len(s) + 16
        self.parts.append(f'<rect x="{self.x + 10}" y="{self.cy}" width="{w}" height="20" rx="4" fill="none" stroke="{T[color]}"/>')
        self.parts.append(t(self.x + 18, self.cy + 14, s, 11, T[color]))
        self.cy += 26
        return self

    def raw(self, svg, h):
        self.parts.append(svg(self.x, self.cy, self.w))
        self.cy += h
        return self

    def render(self):
        T = self.T
        h = self.cy - self.y + 6
        out = [f'<g>',
               f'<rect x="{self.x + 2}" y="{self.y + 4}" width="{self.w}" height="{h}" rx="8" fill="{T["shadow"]}" opacity="0.25"/>',
               f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{h}" rx="8" fill="{T["body"]}" stroke="{T["border"]}"/>',
               f'<path d="M{self.x} {self.y + 30} V{self.y + 8} a8 8 0 0 1 8 -8 H{self.x + self.w - 8} a8 8 0 0 1 8 8 V{self.y + 30} Z" fill="{T[self.head]}"/>',
               f'<circle cx="{self.x + 14}" cy="{self.y + 15}" r="3.5" fill="{T["htext"]}" opacity="0.55"/>',
               t(self.x + 25, self.y + 19.5, self.title, 12.5, T["htext"], weight=600)]
        if self.badge:
            out.append(t(self.x + self.w - 12, self.y + 19.5, self.badge, 10.5, T["htext"], "end", extra='opacity="0.7"'))
        out += self.parts + ["</g>"]
        return "\n".join(out)


def sigma_strip(T, highlight=None, compact=False, caption=True):
    """highlight: None = all phases coloured; a phase key = only that phase coloured (LoRA Phase)."""
    phases = []
    for i in range(len(SIG) - 1):
        a, b = SIG[i], SIG[i + 1]
        if b > a:
            continue  # restart jump: no model call
        if a == SIG[11] and b == 0:
            phases.append("plunge")
        elif a > 0.85 + 1e-9:
            phases.append("comp")
        elif a > 0.75 + 1e-9:
            phases.append("ident")
        else:
            phases.append("tex")

    def draw(x, y, w):
        pad = 12
        x0, W = x + pad, w - 2 * pad
        g = []
        spark_h = 0 if compact else 34
        n = len(SIG)
        if not compact:
            # restart band behind the tail (safelight)
            jx = x0 + W * 12 / (n - 1)
            g.append(f'<rect x="{jx:.1f}" y="{y}" width="{x0 + W - jx:.1f}" height="{spark_h}" fill="{T["restart"]}" opacity="0.55"/>')
            pts = []
            for i, s in enumerate(SIG):
                px = x0 + W * i / (n - 1)
                py = y + 2 + (1 - s) * (spark_h - 4)
                pts.append((px, py))
            d = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts[:13])
            g.append(f'<path d="{d}" fill="none" stroke="{T["muted"]}" stroke-width="1.4"/>')
            d2 = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts[13:])
            g.append(f'<path d="{d2}" fill="none" stroke="{T["muted"]}" stroke-width="1.4"/>')
            g.append(f'<path d="M{pts[12][0]:.1f} {pts[12][1]:.1f} L{pts[13][0]:.1f} {pts[13][1]:.1f}" stroke="{T["tex"]}" stroke-width="1.2" stroke-dasharray="2 2"/>')
        cy = y + spark_h + 4
        cw = W / len(phases)
        for i, ph in enumerate(phases):
            active = {"comp": ("comp",), "ident": ("ident", "plunge", "tex"), "tex": ("tex",)}
            if highlight is None:
                col = T[ph]
            else:
                col = T[highlight] if ph in active[highlight] else T["widget"]
            g.append(f'<rect x="{x0 + i * cw + 1:.1f}" y="{cy}" width="{cw - 2:.1f}" height="8" rx="1.5" fill="{col}"/>')
        if caption:
            if highlight is None:
                g.append(t(x0, cy + 23, "16 calls", 10.5, T["text"], weight=600))
                g.append(t(x0 + 50, cy + 23, "compose 8  identity 3  draft 1  texture 4", 10.5, T["faint"]))
            else:
                name = {"ident": "active σ 0.85 → 0", "tex": "active σ 0.65 → 0", "comp": "active σ 1.0 → 0.85"}[highlight]
                g.append(t(x0, cy + 23, name, 10.5, T["faint"]))
        return "\n".join(g)

    h = (34 if not compact else 0) + 4 + 8 + (28 if caption else 8)
    return draw, h


def latent_frame(T):
    def draw(x, y, w):
        fw, fh = 34, 50
        fx = x + 14
        return (f'<rect x="{fx}" y="{y}" width="{fw}" height="{fh}" rx="2" fill="none" stroke="{T["muted"]}" stroke-width="1.2"/>'
                f'<line x1="{fx + fw / 3:.1f}" y1="{y}" x2="{fx + fw / 3:.1f}" y2="{y + fh}" stroke="{T["line"]}"/>'
                f'<line x1="{fx}" y1="{y + fh / 3:.1f}" x2="{fx + fw}" y2="{y + fh / 3:.1f}" stroke="{T["line"]}"/>'
                + t(fx + fw + 14, y + 20, "1088 × 1600", 13, T["text"], weight=600)
                + t(fx + fw + 14, y + 38, "1.74 MP, portrait 3:2", 10.5, T["faint"]))
    return draw, 58


def report(T):
    rows = [("face 1", "identity 0.71", "kept", "ident"), ("face 2", "identity 0.58 → 0.69", "retried", "warn")]

    def draw(x, y, w):
        g = [f'<rect x="{x + 10}" y="{y}" width="{w - 20}" height="{18 * len(rows) + 10}" rx="6" fill="{T["widget"]}"/>']
        for i, (a, b, c, col) in enumerate(rows):
            yy = y + 18 + 18 * i
            g.append(t(x + 20, yy, a, 11, T["muted"]))
            g.append(t(x + 70, yy, b, 11, T["text"]))
            g.append(t(x + w - 20, yy, c, 11, T[col], "end"))
        return "\n".join(g)
    return draw, 18 * len(rows) + 16


def thumb(T, h=96):
    def draw(x, y, w):
        return (f'<rect x="{x + 10}" y="{y}" width="{w - 20}" height="{h}" rx="4" fill="{T["widget"]}"/>'
                + t(x + w / 2, y + h / 2 + 4, "preview", 11, T["faint"], "middle"))
    return draw, h + 8


def board(theme):
    T = THEMES[theme]
    W, H = 1280, 900
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<defs><pattern id="g" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="{T["dot"]}"/></pattern></defs>',
         f'<rect width="{W}" height="{H}" fill="{T["canvas"]}"/><rect width="{W}" height="{H}" fill="url(#g)"/>']
    cols = [("Setup", 24), ("Sampling", 330), ("Sampling, every knob", 650), ("Finishing and output", 970)]
    for name, x in cols:
        s.append(t(x, 34, name, 14, T["muted"], weight=600))

    # --- setup
    n = Node(T, 24, 52, 270, "Empty Latent", "h_neutral")
    n.sockets([], [("LATENT", "LATENT")]).widget("size", "L  ~1.7 MP", combo=True).widget("aspect", "3:2", combo=True).widget("batch_size", "1")
    n.raw(*latent_frame(T)); s.append(n.render())
    n = Node(T, 24, 316, 270, "Encode", "h_neutral")
    n.sockets([("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
    n.raw(lambda x, y, w: (f'<rect x="{x + 10}" y="{y}" width="{w - 20}" height="58" rx="6" fill="{T["widget"]}"/>'
                           + t(x + 20, y + 18, "cinematic lighting, anamorphic…", 11, T["faint"])
                           + t(x + 20, y + 36, "a woman reading a magazine on a", 11, T["text"])
                           + t(x + 20, y + 52, "train at dusk, window light", 11, T["text"])), 66)
    n.widget("style", "cinematic", combo=True).chip("“magazine” may render as print or text")
    s.append(n.render())
    n = Node(T, 24, 560, 270, "LoRA Phase", "h_ident", "identity")
    n.sockets([("model", "MODEL")], [("MODEL", "MODEL")]).widget("lora_name", "character_v3", combo=True).widget("strength", "1.00").widget("phase", "identity", combo=True)
    n.raw(*sigma_strip(T, "ident", compact=True)); s.append(n.render())
    n = Node(T, 24, 744, 270, "LoRA Phase", "h_tex", "texture")
    n.sockets([("model", "MODEL")], [("MODEL", "MODEL")]).widget("lora_name", "film_grain", combo=True).widget("phase", "texture", combo=True)
    n.raw(*sigma_strip(T, "tex", compact=True)); s.append(n.render())

    # --- simple sampler
    n = Node(T, 330, 52, 290, "KreaPhoton Sampler", "h_comp")
    n.sockets([("model", "MODEL"), ("positive", "CONDITIONING"), ("latent_image", "LATENT")], [("LATENT", "LATENT")])
    n.widget("seed", "1001").widget("preset", "turbo/quality", combo=True).widget("variety", "off", combo=True)
    n.raw(*sigma_strip(T))
    n.widget("denoise", "1.00").widget("preview_method", "auto", combo=True)
    n.section("More inputs", "1 of 4 changed", changed="comp")
    n.raw(*thumb(T, 150))
    s.append(n.render())
    n = Node(T, 330, 620, 290, "Composition Drafts", "h_comp")
    n.sockets([("model", "MODEL"), ("positive", "CONDITIONING")], [("sheet", "IMAGE"), ("best_seed", "INT")])
    n.widget("n_drafts", "6").widget("seed", "1000")

    def sheet(x, y, w):
        g = []
        cw, ch = (w - 20 - 10) / 3, 52
        for i in range(6):
            cx, cy = x + 10 + (i % 3) * (cw + 5), y + (i // 3) * (ch + 5)
            best = i == 4
            g.append(f'<rect x="{cx:.1f}" y="{cy}" width="{cw:.1f}" height="{ch}" rx="3" fill="{T["widget"]}" '
                     + (f'stroke="{T["comp"]}" stroke-width="2"' if best else "") + "/>")
            g.append(t(cx + 6, cy + ch - 7, str(1000 * (i + 1)), 10, T["muted"]))
            g.append(t(cx + cw - 6, cy + ch - 7, ["0.61", "0.48", "0.70", "0.55", "0.83", "0.66"][i], 10,
                       T["text"] if best else T["faint"], "end"))
        return "\n".join(g)
    n.raw(sheet, 120).note("Click a draft to use its seed")
    s.append(n.render())

    # --- advanced
    n = Node(T, 650, 52, 290, "KreaPhoton Sampler (Advanced)", "h_comp")
    n.sockets([("model", "MODEL"), ("positive", "CONDITIONING"), ("latent_image", "LATENT"), ("sigmas", "SIGMAS")],
              [("LATENT", "LATENT")])
    n.widget("seed", "1001")
    n.chip("Start from preset…", "muted")
    n.section("Solver", "euler")
    n.section("Noise", "2 changed", changed="comp")
    n.section("Detail envelope", "1 changed", open_=True, changed="comp")
    n.widget("detail_space", "step", combo=True)
    n.widget("detail_amount", "0.35", changed="comp")
    n.widget("detail_start", "0.15").widget("detail_end", "0.95").widget("detail_peak", "0.60")
    n.widget("detail_sigma_hi", "0.90", dim=True).widget("detail_sigma_peak", "0.50", dim=True).widget("detail_sigma_lo", "0.12", dim=True)
    n.note("σ values apply when detail_space is sigma")
    n.section("Guidance", "window")
    n.section("Perturbed attention", "off")
    n.section("Variety and blend", "off")
    n.section("Phase models", "")
    n.section("Experimental, Sampler v2", "off")
    n.section("Preview", "auto")
    s.append(n.render())
    n = Node(T, 650, 690, 290, "KreaPhoton Scheduler", "h_comp")
    n.sockets([], [("SIGMAS", "SIGMAS")]).widget("steps", "16").widget("restart_frac", "0.25")
    n.raw(*sigma_strip(T, caption=False))
    n.chip("Feeds SamplerCustom: restart will be ignored")
    s.append(n.render())

    # --- finishing
    n = Node(T, 970, 52, 286, "Upscale v2", "h_tex")
    n.sockets([("model", "MODEL"), ("positive", "CONDITIONING"), ("image", "IMAGE"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
    n.widget("preset", "detail", combo=True).widget("scale", "2.00")
    n.note("2176 × 3200 after the last run", "muted")
    s.append(n.render())
    n = Node(T, 970, 300, 286, "Face Detailer", "h_tex")
    n.sockets([("model", "MODEL"), ("image", "IMAGE"), ("vae", "VAE"), ("reference_image", "IMAGE", "opt")],
              [("image", "IMAGE"), ("mask", "MASK"), ("report", "STRING")])
    n.widget("preset", "standard", combo=True).widget("max_faces", "2").widget("identity_boost", "1.50")
    n.raw(*report(T))
    s.append(n.render())
    n = Node(T, 970, 610, 286, "Save Image", "h_neutral")
    n.sockets([("images", "IMAGE")])
    n.widget("folder_path", "~/renders/krea/2026-10").widget("format", "webp", combo=True).widget("quality", "92")
    n.chip("Choose folder…", "muted")
    s.append(n.render())

    s.append("</svg>")
    return "\n".join(s)


def strip_anatomy(theme):
    T = THEMES[theme]
    W, H = 900, 250
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="{T["body"]}"/>']
    draw, h = sigma_strip(T, caption=False)
    s.append(f'<g transform="translate(40 40) scale(2.3)">{draw(0, 0, 360)}</g>')
    s.append(t(68, 30, "\u03c3 curve of the preset, computed on the server", 12, T["muted"]))
    s.append(t(560, 30, "restart: the draft is re-noised to \u03c3 0.65", 12, T["muted"]))
    cw = 336 / 16
    for c, name, k in [(3.5, "composition, 8 calls", "comp"), (9, "identity, 3", "ident"), (11, "draft, 1", "plunge"), (13.5, "texture, 4", "tex")]:
        x = 40 + 2.3 * (12 + cw * (c + 0.5))
        y0 = 40 + 2.3 * 46 + 2
        s.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + 22 + (14 if k == "plunge" else 0):.1f}" stroke="{T[k]}" stroke-width="1.5"/>')
        s.append(t(x, y0 + 38 + (14 if k == "plunge" else 0), name, 12, T["text"], "middle"))
    s.append(t(40, H - 14, "One cell = one model call. Colour is never the only carrier: the caption repeats the counts in words.", 12, T["faint"]))
    s.append("</svg>")
    return "\n".join(s)


for th in ("dark", "light"):
    open(f"{OUT}/node-board-{th}.svg", "w").write(board(th))
    open(f"{OUT}/sigma-strip-{th}.svg", "w").write(strip_anatomy(th))
print("ok")
