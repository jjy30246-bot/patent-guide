#!/usr/bin/env python3
"""신길튼튼365정형외과 · 메리 크리스마스 & 해피 뉴 이어 (도시 감성 모던 모션그래픽)

1920x1080 · 30fps · 60초 · 경쾌한 배경음악(직접 합성).

밤의 도시 → 빛으로 그려지는 트리 → 카운트다운 → 불꽃놀이 → 새해 일출 → 엔딩.

사용:  python3 render_city.py --fonts <폰트폴더> --out <출력.mp4> [--preview 초,초,...]
폰트:  NotoSansKR-{Regular,Bold,Black}.ttf, Montserrat-ExtraBold.ttf, GreatVibes.ttf
"""
import argparse
import math
import os
import subprocess
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render import (smoothstep, sstep, ease_out_cubic, ease_out_back, value_noise, fbm,
                    render_line, pluck, pad, bell, midi_hz, SR)

W, H = 1920, 1080
FPS = 30
DUR = 60.0
NFRAMES = int(FPS * DUR)
WW = W + 900  # 시차 이동을 위한 넓은 도시 레이어

BPM = 112
BEAT = 60 / BPM


def bt(b):
    return b * BEAT


# 장면 시각 (음악 박자에 맞춤)
T_COUNT = bt(52)   # 카운트다운 시작 ≈ 27.9s
T_DROP = bt(62)    # 2027! ≈ 33.2s
T_DAWN = 44.0
T_END = 52.5

WHITE = (255, 255, 255)
GOLD = (255, 206, 96)
CORAL = (255, 112, 104)
MINT = (96, 226, 196)
PINK = (255, 150, 200)
SKY = (120, 190, 255)
NAVY = (30, 36, 84)


def lerp(a, b, k):
    return a + (b - a) * k


def lerpc(c1, c2, k):
    return np.array(c1, np.float32) * (1 - k) + np.array(c2, np.float32) * k


# ---------------------------------------------------------------- 폰트

class Fonts:
    def __init__(self, d):
        self.files = dict(
            reg=os.path.join(d, "NotoSansKR-Regular.ttf"),
            bold=os.path.join(d, "NotoSansKR-Bold.ttf"),
            black=os.path.join(d, "NotoSansKR-Black.ttf"),
            mont=os.path.join(d, "Montserrat-ExtraBold.ttf"),
            script=os.path.join(d, "GreatVibes.ttf"),
        )

    def get(self, which, size):
        f = ImageFont.truetype(self.files[which], size)
        try:  # 가변 폰트이면 굵기 축을 지정
            axes = f.get_variation_axes()
            want = {"mont": 800, "black": 900, "bold": 700, "reg": 400}.get(which)
            if want and axes:
                f.set_variation_by_axes([min(max(want, axes[0]["minimum"]), axes[0]["maximum"])])
        except Exception:
            pass
        return f


def shadowize(line, col=(16, 18, 52), blur=7, opacity=0.55, off=(0, 5)):
    """글자 이미지에 부드러운 그림자를 굽는다."""
    def one(im):
        a = np.asarray(im)[..., 3].astype(np.float32)
        sh = Image.fromarray(a.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(blur))
        sa = np.asarray(sh, np.float32) * opacity
        out = Image.new("RGBA", im.size, col + (0,))
        arr = np.zeros((im.size[1], im.size[0], 4), np.float32)
        arr[..., :3] = col
        arr[..., 3] = np.roll(np.roll(sa, off[1], 0), off[0], 1)
        out = Image.fromarray(arr.astype(np.uint8), "RGBA")
        out.alpha_composite(im)
        return out
    im, boxes = line
    return one(im), [(a, one(c)) for a, c in boxes]


class KText:
    """모던 키네틱 타이포: 'pop'(글자가 통통 튀며 등장) / 'rise'(블러→선명 상승)."""

    def __init__(self, line, cx, cy, t_in, t_out, mode="rise", d_in=0.6, d_out=0.45, stagger=0.045, rise=24,
                 shimmer=()):
        self.img, boxes = line
        self.w, self.h = self.img.size
        self.cx, self.cy = cx, cy
        self.t_in, self.t_out = t_in, t_out
        self.mode = mode
        self.d_in, self.d_out = d_in, d_out
        self.stagger = stagger
        self.rise = rise
        self.shimmer = shimmer
        self.chars = []
        for a, ci in boxes:
            lv = [ci] + [ci.filter(ImageFilter.GaussianBlur(r)) for r in (2.5, 5)]
            self.chars.append((a, ci, lv))
        self.xs = np.arange(self.w, dtype=np.float32)[None, :]

    def end_in(self):
        return self.t_in + self.stagger * max(0, len(self.chars) - 1) + self.d_in

    def draw(self, canvas, t):
        if t < self.t_in or t > self.t_out + self.d_out:
            return
        ox, oy = int(self.cx - self.w / 2), int(self.cy - self.h / 2)
        if t < self.end_in():
            for i, (a, ci, lv) in enumerate(self.chars):
                p = (t - self.t_in - i * self.stagger) / self.d_in
                if p <= 0:
                    continue
                if self.mode == "pop":
                    s = max(0.05, ease_out_back(p, 2.6))
                    al = min(1.0, p * 3)
                    im = ci if s == 1 else ci.resize((max(1, int(ci.size[0] * s)), max(1, int(ci.size[1] * s))), Image.BILINEAR)
                    im = fade(im, al)
                    canvas.alpha_composite(im, (int(ox + a + ci.size[0] / 2 - im.size[0] / 2),
                                                int(oy + ci.size[1] / 2 - im.size[1] / 2 - 30 * (1 - ease_out_cubic(p)))))
                else:
                    e = ease_out_cubic(p)
                    im = lv[0 if e > 0.8 else (1 if e > 0.45 else 2)]
                    canvas.alpha_composite(fade(im, e), (ox + a, oy + int(self.rise * (1 - e))))
            return
        pout = sstep(0, 1, (t - self.t_out) / self.d_out)
        im = self.img
        for (ts, dur) in self.shimmer:
            if ts <= t <= ts + dur:
                q = (t - ts) / dur
                bx = -0.25 * self.w + 1.5 * self.w * q
                g = (np.exp(-((self.xs - bx) / (self.w * 0.05)) ** 2) * 0.75)[..., None]
                arr = np.asarray(im).astype(np.float32)
                arr[..., :3] = arr[..., :3] * (1 - g) + 255 * g
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        if pout > 0:
            s = 1 - 0.08 * pout
            im = im.resize((max(1, int(self.w * s)), max(1, int(self.h * s))), Image.BILINEAR)
            im = fade(im, 1 - pout)
        canvas.alpha_composite(im, (int(self.cx - im.size[0] / 2), int(self.cy - im.size[1] / 2 - 16 * pout)))


def fade(im, a):
    if a >= 0.999:
        return im
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3].astype(np.float32) * max(0.0, a)).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------- 도시

def build_city():
    """3겹의 빌딩 실루엣(마스크) + 창문(켜지는 시각/색)."""
    layers = []
    specs = [  # 바닥 y, 높이 범위, 폭 범위, 창문 여부, seed
        (965, (190, 470), (60, 150), False, 11),
        (1000, (160, 430), (80, 180), True, 22),
        (1085, (150, 380), (110, 230), True, 33),
    ]
    for li, (base, hr, wr, wins, seed) in enumerate(specs):
        r = np.random.default_rng(seed)
        S = 2
        mask = Image.new("L", (WW * S, H * S), 0)
        d = ImageDraw.Draw(mask)
        wT = Image.new("F", (WW, H), 999.0)
        wC = Image.new("RGB", (WW, H), (0, 0, 0))
        dT, dC = ImageDraw.Draw(wT), ImageDraw.Draw(wC)
        x = -40
        while x < WW + 40:
            bw = r.uniform(*wr)
            bh = r.uniform(*hr)
            top = base - bh
            kind = r.choice(["flat", "step", "slant", "dome", "spire"], p=[0.35, 0.25, 0.15, 0.1, 0.15])
            pts = [(x, base + 200), (x, top), (x + bw, top), (x + bw, base + 200)]
            d.polygon([(px * S, py * S) for px, py in pts], fill=255)
            if kind == "step":
                d.rectangle([(x + bw * 0.2) * S, (top - bh * 0.14) * S, (x + bw * 0.8) * S, top * S], fill=255)
            elif kind == "slant":
                d.polygon([(x * S, top * S), ((x + bw) * S, top * S), ((x + bw) * S, (top - bw * 0.35) * S)], fill=255)
            elif kind == "dome":
                d.pieslice([(x + bw * 0.1) * S, (top - bw * 0.4) * S, (x + bw * 0.9) * S, (top + bw * 0.4) * S], 180, 360, fill=255)
            elif kind == "spire":
                d.rectangle([(x + bw * 0.47) * S, (top - 70) * S, (x + bw * 0.53) * S, top * S], fill=255)
            if wins:
                gx, gy = r.choice([14, 16, 18]), r.choice([18, 20, 22])
                ww_, wh_ = gx * 0.5, gy * 0.48
                lit_p = r.uniform(0.35, 0.7)
                warm = r.random() < 0.75
                for yy in np.arange(top + 16, base - 10, gy):
                    for xx in np.arange(x + 10, x + bw - 10, gx):
                        if r.random() > lit_p:
                            continue
                        tt = 0.7 + abs((xx - WW * 0.32) / WW) * 3.0 + (base - yy) / 900 + r.random() * 0.5
                        if warm:
                            c = (255, int(r.uniform(196, 226)), int(r.uniform(120, 160)))
                        else:
                            c = (int(r.uniform(170, 210)), int(r.uniform(214, 236)), 255)
                        if r.random() < 0.06:
                            c = (255, 150, 170)
                        dT.rectangle([xx, yy, xx + ww_, yy + wh_], fill=float(tt))
                        dC.rectangle([xx, yy, xx + ww_, yy + wh_], fill=c)
            x += bw + r.uniform(-8, 14)
        m = np.asarray(mask.resize((WW, H), Image.BOX), np.float32) / 255
        T = np.asarray(wT, np.float32)
        C = np.asarray(wC, np.float32) / 255
        lit = (T < 900).astype(np.float32)
        glow = np.asarray(Image.fromarray((lit * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
        layers.append(dict(mask=m, T=T, C=C, glow=glow, depth=[0.3, 0.6, 1.0][li], has_win=wins))
    return layers


CITY_NIGHT = [(56, 54, 118), (36, 38, 92), (22, 24, 62)]
CITY_DAWN = [(222, 160, 186), (164, 118, 166), (98, 76, 132)]


def pan(t):
    return 760 * sstep(0, DUR, t)


def rise_off(t, k):
    """인트로: 레이어가 아래에서 솟아오른다."""
    p = (t - 0.2 - k * 0.35) / 1.6
    return int(260 * (1 - ease_out_back(p, 1.3))) if p < 1 else 0


# ---------------------------------------------------------------- 스프라이트

def soft_dot(r, col, hard=0.5):
    n = int(r * 4) + 2
    g = np.mgrid[0:n, 0:n].astype(np.float32) - (n - 1) / 2
    d = np.hypot(g[0], g[1]) / r
    a = np.clip(np.exp(-d ** 2 * 2) * (1 - hard) + hard * smoothstep(1.0, 0.7, d), 0, 1)
    arr = np.zeros((n, n, 4), np.uint8)
    arr[..., :3] = col
    arr[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def gift_sprite(w, h, col, rib):
    S = 3
    im = Image.new("RGBA", (int(w * S), int((h + h * 0.45) * S)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    top = h * 0.45 * S
    bw, bh = w * S, h * S
    dark = tuple(int(c * 0.82) for c in col)
    d.rounded_rectangle([0, top + bh * 0.18, bw, top + bh], radius=S * 6, fill=col + (255,))
    d.rounded_rectangle([-S * 4 + S * 4, top, bw, top + bh * 0.26], radius=S * 6, fill=dark + (255,))
    d.rectangle([bw * 0.44, top, bw * 0.56, top + bh], fill=rib + (255,))
    # 리본 매듭
    cx = bw / 2
    for sgn in (-1, 1):
        d.ellipse([cx + sgn * bw * 0.04 - bw * 0.17 * (sgn < 0) - bw * 0.0, top - bh * 0.30,
                   cx + sgn * bw * 0.04 + bw * 0.17 * (sgn > 0), top + bh * 0.02], outline=rib + (255,), width=int(S * 5))
    d.ellipse([cx - bw * 0.05, top - bh * 0.1, cx + bw * 0.05, top + bh * 0.02], fill=rib + (255,))
    return im.resize((int(w), int(h * 1.45)), Image.LANCZOS)


def star_sprite(R, col):
    S = 4
    n = int(R * 2.4 * S)
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = n / 2
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = R * S if k % 2 == 0 else R * S * 0.45
        pts.append((c + math.cos(a) * rr, c + math.sin(a) * rr))
    d.polygon(pts, fill=col + (255,))
    return im.resize((n // S, n // S), Image.LANCZOS)


def cloud_sprite(w, r):
    S = 2
    h = int(w * 0.42)
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    base = h * S * 0.92
    d.rounded_rectangle([0, base - h * S * 0.32, w * S, base], radius=h * S * 0.16, fill=(255, 255, 255, 235))
    for k in range(4):
        cx = w * S * (0.2 + 0.2 * k + r.uniform(-0.04, 0.04))
        rr = h * S * r.uniform(0.25, 0.45)
        d.ellipse([cx - rr, base - h * S * 0.2 - rr * 2, cx + rr, base - h * S * 0.2], fill=(255, 255, 255, 235))
    return im.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- 장면

class Scene:
    def __init__(self, fonts_dir):
        print("장면 준비 중...", flush=True)
        self.fonts = Fonts(fonts_dir)
        self.city = build_city()
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.yy = yy
        r = np.random.default_rng(7)
        # 별
        self.stars = [(r.random() * W, r.random() * 620, r.random() * math.tau, 0.6 + r.random() * 1.8,
                       r.integers(0, 2)) for _ in range(170)]
        self.star_spr = [soft_dot(1.4, (255, 255, 255), 0.6), soft_dot(2.2, (255, 246, 220), 0.5)]
        # 달 / 해 빛무리
        gy, gx = np.mgrid[0:H * 2, 0:W * 2].astype(np.float32)
        self.halo = np.exp(-(((gx - W) / 520) ** 2 + ((gy - H) / 520) ** 2)).astype(np.float32)
        self.halo_big = np.exp(-(((gx - W) / 1100) ** 2 + ((gy - H) / 700) ** 2)).astype(np.float32)
        self.gy_full, self.gx_full = np.mgrid[0:H, 0:W].astype(np.float32)
        # 눈
        self.snow = []
        for i in range(260):
            dep = r.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            self.snow.append(dict(x=r.random() * W, y=r.random() * H, v=[34, 60, 105][dep] * (0.8 + 0.4 * r.random()),
                                  amp=10 + 25 * r.random(), fr=0.15 + 0.3 * r.random(), ph=r.random() * math.tau,
                                  dep=dep))
        self.snow_spr = [soft_dot(1.6, (255, 255, 255), 0.5), soft_dot(2.8, (255, 255, 255), 0.4),
                         soft_dot(5.0, (255, 255, 255), 0.1)]
        # 트리 (빛 점으로 쌓이는 기하학 트리)
        self.tree = []
        cx, base, top = 470, 905, 330
        R = 22
        pal = [GOLD, GOLD, GOLD, (255, 238, 190), CORAL, MINT, PINK]
        for row in range(R):
            yv = base - row * (base - top) / R
            wv = 420 * (1 - row / R) + 10
            n = max(1, int(wv / 24))
            for i in range(n):
                xv = cx - wv / 2 + (i + 0.5) * wv / n
                c = pal[r.integers(0, len(pal))]
                if (i + row) % 6 == 0:
                    c = CORAL
                if (i - row) % 9 == 0:
                    c = MINT
                self.tree.append(dict(x=xv, y=yv, c=c, t=4.6 + row * 0.11 + abs(i - n / 2) * 0.012,
                                      ph=r.random() * math.tau, s=2.2 + r.random() * 1.6))
        self.tree_star = dict(x=cx, y=top - 34, t=4.6 + R * 0.11 + 0.25)
        self.star_img = star_sprite(36, GOLD)
        self.tree_cx = cx
        # 줄 전구
        self.bulbs = []
        pal2 = [GOLD, CORAL, MINT, PINK, SKY]
        for si, (y0, sag, ph) in enumerate([(70, 70, 0), (40, 90, 0.5)]):
            n = 34
            for k in range(n + 1):
                u = k / n
                x = -20 + (W + 40) * u
                y = y0 + sag * 4 * u * (1 - u) + 18 * math.sin(u * math.tau * 2 + ph)
                self.bulbs.append(dict(x=x, y=y, c=pal2[(k + si * 2) % len(pal2)], t=12.4 + u * 1.4 + si * 0.3,
                                       k=k + si))
        # 선물상자
        gifts = [(1090, 110, CORAL, GOLD), (1215, 84, MINT, WHITE), (1340, 128, GOLD, CORAL),
                 (1630, 96, PINK, WHITE), (1750, 120, SKY, GOLD), (1470, 70, CORAL, MINT)]
        self.gifts = [dict(x=x, w=w, img=gift_sprite(w, w * 0.85, c, rb), t=13.0 + i * 0.16) for i, (x, w, c, rb) in enumerate(gifts)]
        # 꽃가루(컨페티)
        cols = [CORAL, MINT, GOLD, PINK, SKY, WHITE]
        self.confetti = []
        for i in range(200):
            burst = i >= 90
            c = cols[r.integers(0, len(cols))]
            sw, sh = int(r.uniform(9, 15)), int(r.uniform(5, 8))
            spr = Image.new("RGBA", (sw, sh), c + (255,))
            if burst:
                a = r.uniform(-math.pi, 0)
                v = r.uniform(400, 1100)
                self.confetti.append(dict(t0=T_DROP + r.random() * 0.25, x=960 + r.normal() * 60, y=560,
                                          vx=math.cos(a) * v * 0.9, vy=math.sin(a) * v, spr=spr, rot=r.random() * 360,
                                          vr=r.uniform(-400, 400), fl=r.uniform(3, 9), burst=True, ph=r.random() * 6))
            else:
                self.confetti.append(dict(t0=20.3 + r.random() * 7.0, x=r.random() * W, y=-30 - r.random() * 200,
                                          vx=r.uniform(-30, 30), vy=r.uniform(90, 170), spr=spr, rot=r.random() * 360,
                                          vr=r.uniform(-300, 300), fl=r.uniform(3, 9), burst=False, ph=r.random() * 6))
        # 불꽃놀이
        self.fw = []
        rf = np.random.default_rng(2027)
        plan = [(T_DROP, 520, 300, GOLD, CORAL), (T_DROP, 1400, 260, MINT, WHITE), (T_DROP + 0.05, 960, 190, PINK, GOLD)]
        tt = T_DROP + 0.9
        while tt < 43.5:
            plan.append((tt, rf.uniform(250, 1670), rf.uniform(150, 420),
                         [GOLD, CORAL, MINT, PINK, SKY][rf.integers(0, 5)], [WHITE, GOLD][rf.integers(0, 2)]))
            tt += rf.uniform(0.55, 1.05)
        for (t0, x, y, c1, c2) in plan:
            n = 150
            parts = []
            for k in range(n):
                a = rf.random() * math.tau
                v = (0.55 + 0.45 * math.sqrt(rf.random())) * rf.uniform(400, 520)
                parts.append((math.cos(a) * v, math.sin(a) * v, c1 if k % 3 else c2, rf.random() * 6))
            self.fw.append(dict(t0=t0, x=x, y=y, parts=parts, life=2.3))
        self.fw_times = [f["t0"] for f in self.fw]
        # 새벽: 구름, 새
        self.clouds = [dict(img=cloud_sprite(int(w), r), x=x, y=y, v=v) for (w, x, y, v) in
                       [(380, 180, 170, 12), (260, 1300, 120, 9), (320, 1600, 300, 14), (220, 700, 260, 10)]]
        self.birds = [dict(x=-60 - i * 70 - r.random() * 40, y=300 + (i % 3) * 36 + r.random() * 20, s=14 + r.random() * 6,
                           ph=r.random() * math.tau) for i in range(7)]
        # 카운트다운 숫자
        self.count = []
        for k, ch in enumerate("54321"):
            im, _ = shadowize(render_line(self.fonts, [(ch, WHITE)], "mont", 300, 0), blur=12, opacity=0.5, off=(0, 8))
            self.count.append((T_COUNT + k * 2 * BEAT, im))
        self.texts = self.build_texts()
        print("준비 완료", flush=True)

    def T(self, segs, font, size, sp=0, shadow=True):
        ln = render_line(self.fonts, segs, font, size, sp)
        return shadowize(ln) if shadow else ln

    def build_texts(self):
        T = []
        rx = 1260
        # 크리스마스
        T.append(KText(self.T([("MERRY", WHITE)], "mont", 118, 10), rx, 360, 5.0, 12.0, mode="pop", stagger=0.07, d_in=0.55))
        T.append(KText(self.T([("CHRISTMAS", GOLD)], "mont", 118, 8), rx, 490, 5.5, 12.1, mode="pop", stagger=0.06, d_in=0.55,
                       shimmer=((8.5, 1.0),)))
        T.append(KText(self.T([("행복이 가득한 크리스마스 되세요", WHITE)], "bold", 44, 1), rx, 610, 7.0, 12.2))
        T.append(KText(self.T([("따뜻하고 포근한 ", WHITE), ("성탄절", CORAL)], "black", 66, 1), rx, 420, 12.8, 19.8, mode="pop", stagger=0.05))
        T.append(KText(self.T([("사랑하는 사람들과 웃음 가득한 하루 보내세요", WHITE)], "bold", 40, 1), rx, 520, 13.6, 19.9))
        T.append(KText(self.T([("올 한 해 ", WHITE), ("신길튼튼365정형외과", MINT), ("와", WHITE)], "black", 54, 1), rx, 420, 20.4, 27.2, mode="pop", stagger=0.04))
        T.append(KText(self.T([("함께해 주셔서 진심으로 ", WHITE), ("감사합니다", GOLD)], "black", 54, 1), rx, 512, 21.0, 27.3, mode="pop", stagger=0.04))
        T.append(KText(self.T([("여러분의 건강한 웃음이 저희의 가장 큰 기쁨입니다", WHITE)], "bold", 36, 1), rx, 600, 22.2, 27.4))
        # 새해
        T.append(KText(self.T([("HAPPY NEW YEAR", WHITE)], "mont", 112, 8), 960, 330, T_DROP + 0.25, 43.6, mode="pop", stagger=0.05, d_in=0.5))
        T.append(KText(self.T([("2027", GOLD)], "mont", 230, 10), 960, 520, T_DROP - 0.02, 43.7, mode="pop", stagger=0.08, d_in=0.6,
                       shimmer=((36.0, 1.2), (40.5, 1.2))))
        T.append(KText(self.T([("새해 ", WHITE), ("복", CORAL), (" 많이 받으세요", WHITE)], "black", 72, 2), 960, 715, T_DROP + 1.6, 43.8, mode="pop", stagger=0.06))
        # 일출
        lite = lambda ln: shadowize(ln, col=(255, 255, 255), blur=8, opacity=0.8, off=(0, 0))
        T.append(KText(lite(render_line(self.fonts, [("2027년, 늘 ", NAVY), ("건강", (12, 150, 128)), ("하고 ", NAVY), ("행복", (232, 128, 20)), ("한 한 해 되세요", NAVY)], "black", 62, 1)),
                       960, 300, 45.2, 52.0, mode="pop", stagger=0.04))
        T.append(KText(lite(render_line(self.fonts, [("더 친절하게, 더 튼튼하게 곁에 있겠습니다", NAVY)], "bold", 40, 1)), 960, 395, 46.2, 52.1))
        # 엔딩 (유리 패널 위)
        T.append(KText(render_line(self.fonts, [("Merry Christmas & Happy New Year", CORAL)], "script", 78, 0), 960, 395, T_END + 0.5, 99, d_in=0.8, stagger=0.025))
        T.append(KText(render_line(self.fonts, [("신길튼튼365정형외과", NAVY)], "black", 92, 1), 960, 515, T_END + 1.1, 99, mode="pop", stagger=0.05,
                       shimmer=((55.6, 1.2),)))
        T.append(KText(render_line(self.fonts, [("서울특별시 영등포구 가마산로 483  3A01호~3A05호", (70, 76, 120))], "reg", 32, 1), 960, 612, T_END + 2.0, 99))
        T.append(KText(render_line(self.fonts, [("임직원 일동", (110, 110, 150))], "bold", 28, 6), 960, 668, T_END + 2.5, 99))
        return T

    # ---------------- 배경 (하늘 + 도시)
    def background(self, t):
        dawn = sstep(T_DAWN - 0.5, T_DAWN + 5.5, t)
        top = lerpc(lerpc((10, 14, 42), (14, 16, 48), sstep(T_COUNT, T_DROP, t)), (104, 168, 236), dawn)
        bot = lerpc((78, 48, 128), (255, 196, 160), dawn)
        mid = lerpc((36, 30, 86), (196, 186, 230), dawn)
        y = self.yy
        k1 = np.clip(y / 0.6, 0, 1)
        k2 = np.clip((y - 0.6) / 0.4, 0, 1)
        sky = np.where(y < 0.6, top * (1 - k1) + mid * k1, mid * (1 - k2) + bot * k2) / 255
        img = np.broadcast_to(sky, (H, W, 3)).copy()
        # 달 (밤) / 해 (새벽)
        mo = (1 - dawn) * sstep(0.3, 2.0, t)
        if mo > 0.01:
            mx, my = 1580, 200
            g = self.halo[H - my:2 * H - my, W - mx:2 * W - mx]
            img += g[..., None] * np.array([0.25, 0.22, 0.35], np.float32) * mo
            dd = np.hypot(self.gx_full[110:290, 1490:1670] - mx, self.gy_full[110:290, 1490:1670] - my)
            a = smoothstep(72, 68, dd) * mo
            sub = img[110:290, 1490:1670]
            sub[:] = sub * (1 - a[..., None]) + np.array([1.0, 0.97, 0.88], np.float32) * a[..., None]
        if dawn > 0.01:
            sy = int(lerp(1060, 600, ease_out_cubic((t - T_DAWN) / 7.0)))
            g = self.halo_big[H - sy:2 * H - sy, W - 960:2 * W - 960]
            if True:
                img += g[..., None] * np.array([0.55, 0.32, 0.12], np.float32) * dawn
            y0, y1 = max(0, sy - 140), min(H, sy + 140)
            dd = np.hypot(self.gx_full[y0:y1, 820:1100] - 960, self.gy_full[y0:y1, 820:1100] - sy)
            a = smoothstep(122, 116, dd) * dawn
            sub = img[y0:y1, 820:1100]
            sub[:] = sub * (1 - a[..., None]) + np.array([1.0, 0.86, 0.6], np.float32) * a[..., None]
        np.clip(img, 0, 1, out=img)
        # 도시 레이어
        p = pan(t)
        for li, L in enumerate(self.city):
            off = int(p * L["depth"])
            dy = rise_off(t, li)
            m = L["mask"][:, off:off + W]
            col = lerpc(CITY_NIGHT[li], CITY_DAWN[li], dawn) / 255
            if dy > 0:
                m = np.vstack([np.zeros((dy, W), np.float32), m[:H - dy]])
            img *= 1 - m[..., None]
            img += m[..., None] * col
            if L["has_win"]:
                on = (1 - sstep(T_DAWN + 1.0, T_DAWN + 6.0, t))
                if on > 0.01:
                    T = L["T"][:, off:off + W]
                    C = L["C"][:, off:off + W]
                    G = L["glow"][:, off:off + W]
                    if dy > 0:
                        T = np.vstack([np.full((dy, W), 999, np.float32), T[:H - dy]])
                        C = np.vstack([np.zeros((dy, W, 3), np.float32), C[:H - dy]])
                        G = np.vstack([np.zeros((dy, W), np.float32), G[:H - dy]])
                    lit = smoothstep(0, 0.25, t - T) * on
                    img = img * (1 - lit[..., None]) + C * lit[..., None]
                    img += G[..., None] * np.array([0.30, 0.22, 0.12], np.float32) * on * sstep(0.8, 3.5, t) * 0.6
        # 카운트다운 중 살짝 어둡게
        dim = sstep(T_COUNT - 0.6, T_COUNT, t) * (1 - sstep(T_DROP, T_DROP + 0.4, t)) * 0.35
        if dim > 0:
            img *= 1 - dim
        return np.clip(img, 0, 1)

    # ---------------- 빛(가산 합성) 버퍼
    def lights(self, t):
        core = Image.new("RGB", (W, H), (0, 0, 0))
        half = Image.new("RGB", (W // 2, H // 2), (0, 0, 0))
        dc, dh = ImageDraw.Draw(core), ImageDraw.Draw(half)

        def dot(x, y, r, c, a):
            cc = tuple(int(v * a) for v in c)
            dc.ellipse([x - r, y - r, x + r, y + r], fill=cc)
            rh = r * 1.6
            dh.ellipse([x / 2 - rh, y / 2 - rh, x / 2 + rh, y / 2 + rh], fill=cc)

        p_near = pan(t)
        # 트리
        tf = 1 - sstep(T_COUNT - 1.0, T_COUNT + 0.2, t)
        if tf > 0 and t > 4.4:
            for d in self.tree:
                k = (t - d["t"]) / 0.35
                if k <= 0:
                    continue
                s = ease_out_back(k, 3) if k < 1 else 1
                tw = 0.65 + 0.35 * math.sin(t * 3.1 + d["ph"])
                dot(d["x"] - p_near, d["y"], d["s"] * s, d["c"], tw * tf * min(1, k * 2))
        # 줄 전구 (흐르는 체이스 패턴)
        bf = 1 - sstep(T_COUNT - 0.5, T_COUNT + 0.3, t)
        bf2 = sstep(T_DROP, T_DROP + 1.0, t) * (1 - sstep(T_DAWN, T_DAWN + 3, t))
        for b in self.bulbs:
            if t < b["t"]:
                continue
            chase = 0.55 + 0.45 * math.sin(t * 5 - b["k"] * 0.8)
            a = min(1, (t - b["t"]) * 4) * chase * max(bf, bf2)
            if a > 0.02:
                dot(b["x"], b["y"], 4.2, b["c"], a)
        # 별똥별
        for (t0, x0, y0) in ((9.6, 200, 90), (17.2, 900, 60)):
            q = (t - t0) / 1.1
            if 0 < q < 1:
                for j in range(14):
                    qq = q - j * 0.012
                    if qq < 0:
                        continue
                    dot(x0 + 700 * qq, y0 + 220 * qq, 3.0 - j * 0.18, WHITE, (1 - j / 14) * (1 - q) * 1.2)
        # 불꽃놀이 로켓 + 폭발
        for f in self.fw:
            dt = t - f["t0"]
            if -0.9 < dt < 0:
                q = (dt + 0.9) / 0.9
                yy = lerp(H + 20, f["y"], ease_out_cubic(q))
                for j in range(6):
                    dot(f["x"] + math.sin(j) * 1.5, yy + j * 9, 2.6 - j * 0.3, (255, 230, 180), 1 - j / 6)
            if 0 <= dt < f["life"]:
                fade_ = (1 - dt / f["life"]) ** 1.4
                for (vx, vy, c, ph) in f["parts"]:
                    for j, lag in enumerate((0, 0.05, 0.1)):
                        tt = dt - lag
                        if tt < 0:
                            continue
                        e = (1 - math.exp(-2.0 * tt)) / 2.0
                        x = f["x"] + vx * e
                        y = f["y"] + vy * e + 46 * tt * tt
                        tw = 1 if dt < 1.2 else (0.5 + 0.5 * math.sin(ph + t * 25))
                        dot(x, y, 3.8 - j * 0.9, c, min(1.0, fade_ * tw * (1 - j * 0.3) * 1.3))
                if dt < 0.15:
                    dot(f["x"], f["y"], 14 * (1 - dt / 0.15), WHITE, 1)
        g1 = half.filter(ImageFilter.GaussianBlur(4)).resize((W, H), Image.BILINEAR)
        g2 = half.filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR)
        out = (np.asarray(core, np.float32) * 0.9 + np.asarray(g1, np.float32) * 1.4 + np.asarray(g2, np.float32) * 1.2) / 255
        return out

    def camera(self, canvas, t):
        z = 1.02 + 0.03 * sstep(0, DUR, t) + 0.025 * math.exp(-max(0, t - T_DROP) * 3) * (t > T_DROP)
        cw, ch = W / z, H / z
        sh = 0
        if 0 < t - T_DROP < 0.5:
            sh = 9 * (1 - (t - T_DROP) / 0.5)
        ox = (W - cw) / 2 + sh * math.sin(t * 90)
        oy = (H - ch) / 2 + sh * math.cos(t * 77)
        ox = min(max(ox, 0), W - cw); oy = min(max(oy, 0), H - ch)
        return canvas.resize((W, H), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))

    def frame(self, t):
        img = self.background(t)
        # 별
        night = 1 - sstep(T_DAWN - 0.5, T_DAWN + 4, t)
        img += self.lights(t)
        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        if night > 0.01:
            for (x, y, ph, sp, k) in self.stars:
                a = night * sstep(0.5, 2.5, t) * (0.35 + 0.65 * math.sin(t * sp + ph) ** 2)
                canvas.alpha_composite(fade(self.star_spr[k], a), (int(x), int(y)))
        # 트리 꼭대기 별
        st = self.tree_star
        tf = 1 - sstep(T_COUNT - 1.0, T_COUNT + 0.2, t)
        if t > st["t"] and tf > 0:
            k = (t - st["t"]) / 0.5
            s = ease_out_back(k, 3) if k < 1 else 1 + 0.06 * math.sin(t * 4)
            im = self.star_img.resize((max(2, int(self.star_img.size[0] * s)),) * 2, Image.BILINEAR).rotate(t * 25, Image.BICUBIC, expand=True)
            canvas.alpha_composite(fade(im, tf), (int(st["x"] - pan(t) - im.size[0] / 2), int(st["y"] - im.size[1] / 2)))
        # 새벽 구름 / 새
        dawn = sstep(T_DAWN, T_DAWN + 4, t)
        if dawn > 0.01:
            for c in self.clouds:
                x = c["x"] - (t - T_DAWN) * c["v"]
                canvas.alpha_composite(fade(c["img"], dawn * 0.9), (int(x), int(c["y"])))
            d = ImageDraw.Draw(canvas)
            for b in self.birds:
                tt = t - T_DAWN - 1.5
                if tt < 0:
                    continue
                x = b["x"] + tt * 170
                y = b["y"] - tt * 12 + 6 * math.sin(tt * 1.3 + b["ph"])
                f = math.sin(tt * 9 + b["ph"]) * b["s"] * 0.55
                s = b["s"]
                d.line([(x - s, y - f), (x, y), (x + s, y - f)], fill=(70, 60, 110, 255), width=3)
        # 선물상자
        gf = 1 - sstep(T_COUNT - 0.8, T_COUNT, t)
        for g in self.gifts:
            k = (t - g["t"]) / 0.55
            if k <= 0 or gf <= 0:
                continue
            s = ease_out_back(k, 2.4) if k < 1 else 1.0
            im = g["img"]
            nh = max(1, int(im.size[1] * s))
            im2 = im.resize((max(1, int(im.size[0] * (0.7 + 0.3 * s))), nh), Image.BILINEAR)
            yb = 1062 + 140 * (1 - gf)
            canvas.alpha_composite(fade(im2, gf), (int(g["x"] - im2.size[0] / 2), int(yb - nh)))
        canvas = self.camera(canvas, t)
        # 눈
        sf = sstep(3.0, 5.0, t) * (1 - sstep(T_DAWN - 1, T_DAWN + 2, t))
        if sf > 0.01:
            for f in self.snow:
                y = (f["y"] + f["v"] * t) % (H + 40) - 20
                x = (f["x"] + f["amp"] * math.sin(f["fr"] * math.tau * t + f["ph"]) - pan(t) * 0.2 * (f["dep"] + 1)) % W
                canvas.alpha_composite(fade(self.snow_spr[f["dep"]], sf * (0.55 + 0.2 * f["dep"])), (int(x), int(y)))
        # 컨페티
        for c in self.confetti:
            dt = t - c["t0"]
            if dt < 0:
                continue
            if c["burst"]:
                x = c["x"] + c["vx"] * (1 - math.exp(-2.2 * dt)) / 2.2 + 25 * math.sin(dt * 3 + c["ph"])
                y = c["y"] + c["vy"] * (1 - math.exp(-2.2 * dt)) / 2.2 + 120 * dt * dt * 0.5 + 60 * dt
                life = 1 - sstep(8.5, 10.5, dt)
            else:
                x = c["x"] + c["vx"] * dt + 30 * math.sin(dt * 2 + c["ph"])
                y = c["y"] + c["vy"] * dt
                life = 1 - sstep(T_COUNT - 0.8, T_COUNT, t)
            if y > H + 30 or life <= 0:
                continue
            sx = abs(math.cos(dt * c["fl"] + c["ph"]))
            spr = c["spr"].resize((c["spr"].size[0], max(1, int(c["spr"].size[1] * (0.2 + 0.8 * sx)))))
            im = spr.rotate(c["rot"] + c["vr"] * dt, Image.BILINEAR, expand=True)
            canvas.alpha_composite(fade(im, life), (int(x), int(y)))
        # 카운트다운
        for k, (t0, im) in enumerate(self.count):
            dt = t - t0
            dur = 2 * BEAT
            if 0 <= dt < dur:
                s = 1.55 - 0.55 * ease_out_back(min(1, dt / 0.28), 2.0)
                if dt > dur - 0.15:
                    s *= 1 - 0.4 * (dt - (dur - 0.15)) / 0.15
                a = min(1, dt / 0.1) * (1 - sstep(dur - 0.15, dur, dt))
                im2 = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR)
                canvas.alpha_composite(fade(im2, a), (int(960 - im2.size[0] / 2), int(520 - im2.size[1] / 2)))
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(ov)
                rr = 230
                dd.ellipse([960 - rr, 540 - rr, 960 + rr, 540 + rr], outline=(255, 255, 255, int(50 * a)), width=4)
                dd.arc([960 - rr, 540 - rr, 960 + rr, 540 + rr], -90, -90 + 360 * min(1, dt / dur),
                       fill=[CORAL, MINT, GOLD, PINK, SKY][k] + (int(255 * a),), width=10)
                canvas.alpha_composite(ov)
        # 엔딩 유리 패널
        if t > T_END:
            q = ease_out_back(min(1, (t - T_END) / 0.8), 1.4)
            pw, ph_ = int(1180 * (0.85 + 0.15 * q)), int(400 * (0.85 + 0.15 * q))
            x0, y0 = 960 - pw // 2, 530 - ph_ // 2
            box = (max(0, x0), max(0, y0), min(W, x0 + pw), min(H, y0 + ph_))
            region = canvas.crop(box).filter(ImageFilter.GaussianBlur(18))
            mask = Image.new("L", region.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, region.size[0] - 1, region.size[1] - 1], radius=40, fill=int(255 * min(1, (t - T_END) / 0.5)))
            tint = Image.new("RGBA", region.size, (255, 255, 255, 0))
            ImageDraw.Draw(tint).rounded_rectangle([0, 0, region.size[0] - 1, region.size[1] - 1], radius=40, fill=(255, 255, 255, 170),
                                                   outline=(255, 255, 255, 230), width=3)
            region = region.convert("RGBA")
            region.alpha_composite(tint)
            canvas.paste(region, box[:2], mask)
        # 문구
        for tx in self.texts:
            tx.draw(canvas, t)
        out = np.asarray(canvas.convert("RGB"), np.float32) / 255
        # 2027 순간 섬광
        if 0 <= t - T_DROP < 0.45:
            out = out + (1 - out) * 0.55 * (1 - (t - T_DROP) / 0.45)
        f_in = sstep(0, 0.8, t)
        f_out = 1 - sstep(DUR - 1.2, DUR - 0.05, t)
        out = out * f_in * f_out + (1 - f_out) * f_in * np.array([1.0, 0.98, 0.96], np.float32)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------- 음악 (경쾌한 C장조, 112 BPM)

def kick(amp=0.9):
    n = int(0.32 * SR); t = np.arange(n) / SR
    f = 45 + 90 * np.exp(-t / 0.035)
    return (np.sin(math.tau * np.cumsum(f) / SR) * np.exp(-t / 0.12) * amp).astype(np.float32)


def noise_hp(n, seed, order=2):
    x = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    for _ in range(order):
        x = np.diff(x, prepend=0)
    return x / (np.max(np.abs(x)) + 1e-9)


def clap(amp=0.35, seed=1):
    n = int(0.22 * SR); t = np.arange(n) / SR
    x = noise_hp(n, seed, 1)
    env = np.exp(-t / 0.07) * (1 + 0.6 * ((t > 0.01) & (t < 0.02)))
    return (x * env * amp).astype(np.float32)


def hat(amp=0.12, seed=2, dec=0.03):
    n = int(0.08 * SR); t = np.arange(n) / SR
    return (noise_hp(n, seed, 3) * np.exp(-t / dec) * amp).astype(np.float32)


def sleigh(amp=0.10, seed=3):
    n = int(0.16 * SR); t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    r = np.random.default_rng(seed)
    for k in range(6):
        o = int(r.uniform(0, 0.03) * SR)
        f = r.uniform(5200, 8200)
        seg = np.sin(math.tau * f * t[: n - o]) * np.exp(-t[: n - o] / 0.03)
        y[o:] += seg
    y += noise_hp(n, seed + 9, 3) * np.exp(-t / 0.05) * 0.6
    return (y / 6 * amp * 3).astype(np.float32)


def bass(freq, dur, amp=0.22):
    n = int(dur * SR); t = np.arange(n) / SR
    y = np.sin(math.tau * freq * t) + 0.35 * np.sin(2 * math.tau * freq * t) + 0.1 * np.sin(3 * math.tau * freq * t)
    env = np.minimum(1, t / 0.005) * np.exp(-t / 0.35)
    return (y * env * amp).astype(np.float32)


def glock(freq, dur=1.4, amp=0.16):
    n = int(dur * SR); t = np.arange(n) / SR
    y = np.sin(math.tau * freq * t) * np.exp(-t / 0.7) + 0.4 * np.sin(math.tau * freq * 2.76 * t) * np.exp(-t / 0.18) \
        + 0.2 * np.sin(math.tau * freq * 5.4 * t) * np.exp(-t / 0.06)
    return (y * np.minimum(1, t / 0.002) * amp).astype(np.float32)


def make_music(path, scene_fw_times):
    n = int((DUR + 3) * SR)
    L = np.zeros(n, np.float32); R = np.zeros(n, np.float32)

    def add(sig, t, pan_=0.0, g=1.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[: n - i] * g
        gl, gr = math.cos((pan_ + 1) * math.pi / 4), math.sin((pan_ + 1) * math.pi / 4)
        L[i:i + len(sig)] += sig * gl
        R[i:i + len(sig)] += sig * gr

    prog = [(48, [60, 64, 67]), (43, [59, 62, 67]), (45, [60, 64, 69]), (41, [60, 65, 69])]  # C G Am F
    total_beats = int(DUR / BEAT) + 1
    for bar in range(0, total_beats // 4 + 1):
        b0 = bar * 4
        root, tones = prog[bar % 4]
        if b0 >= 104:
            root, tones = 48, [60, 64, 67]
        tb = bt(b0)
        if tb > DUR:
            break
        countdown = 52 <= b0 < 62
        dawnsec = 86 <= b0 < 100
        # 패드
        add(pad([midi_hz(m) for m in tones], 4 * BEAT + 1.5, 0.07), tb, 0, 1.0)
        if b0 >= 104:
            add(pad([midi_hz(m) for m in [48, 55, 60, 64, 67, 72]], 7.5, 0.08), tb)
            continue
        # 베이스 (8분음표 통통)
        if 8 <= b0 and not countdown:
            for k in range(8):
                f = midi_hz(root + (12 if k % 4 == 3 else 0))
                add(bass(f, BEAT * 0.6, 0.20 if not dawnsec else 0.12), tb + k * BEAT / 2)
        # 코드 스탭 (엇박)
        if 4 <= b0 and not countdown:
            for k in range(4):
                for m in tones:
                    add(pluck(midi_hz(m + 12), 0.6, 0.045), tb + k * BEAT + BEAT / 2, 0.2)
        # 드럼
        if 8 <= b0 < 52 or 62 <= b0 < 86:
            full = b0 >= 62
            for k in range(4):
                add(kick(0.8 if full else 0.6), tb + k * BEAT)
                if k % 2 == 1:
                    add(clap(0.32, b0 + k), tb + k * BEAT, 0.05)
            for k in range(8):
                add(sleigh(0.09 if not full else 0.11, b0 * 8 + k), tb + k * BEAT / 2, 0.35)
                if b0 >= 40:
                    add(hat(0.06, b0 * 16 + k), tb + k * BEAT / 2 + BEAT / 4, -0.3)
        if dawnsec:
            for k in range(4):
                add(sleigh(0.05, b0 * 4 + k), tb + k * BEAT, 0.3)
            add(clap(0.12, b0), tb + 3 * BEAT)
    # 멜로디 (글로켄슈필 + 플럭)
    P1 = [(0, 76, 1), (1, 79, 1), (2, 81, .5), (2.5, 79, .5), (3, 76, 1),
          (4, 74, 1), (5, 79, 1), (6, 83, 1), (7, 81, 1),
          (8, 81, 1), (9, 84, 1), (10, 83, .5), (10.5, 81, .5), (11, 76, 1),
          (12, 77, 1), (13, 76, .5), (13.5, 74, .5), (14, 72, 2)]
    P2 = P1[:14] + [(12, 77, 1), (13, 79, 1), (14, 84, 2)]
    sections = [(4, P1[:9], 0.10), (8, P1, 0.16), (24, P2, 0.16), (40, P1, 0.15), (62, P2, 0.18), (78, P1, 0.18),
                (88, P1[:9], 0.11), (96, P2[9:], 0.12), (100, [(0, 79, 1), (1, 84, 1), (2, 88, 2), (4, 84, 4)], 0.14)]
    for (b0, ph, a) in sections:
        for (b, m, d) in ph:
            tt = bt(b0 + b)
            add(glock(midi_hz(m), 1.4, a), tt, 0.15)
            add(pluck(midi_hz(m), max(0.8, d * BEAT + 0.4), a * 0.9), tt, -0.1)
            if b0 >= 62 and b0 < 86:
                add(glock(midi_hz(m + 12), 1.0, a * 0.35), tt + 0.005, 0.4)
    # 카운트다운: 라이저 + 째깍
    rn = int((T_DROP - T_COUNT) * SR)
    tt = np.arange(rn) / SR
    k = tt / tt[-1]
    sweep = np.sin(math.tau * np.cumsum(220 + 900 * k ** 2) / SR) * 0.05 * k
    nz = noise_hp(rn, 77, 2) * 0.05 * k ** 2
    add((sweep + nz).astype(np.float32), T_COUNT)
    for i in range(5):
        add(glock(midi_hz(84 + [0, 2, 4, 5, 7][i]), 0.8, 0.14), T_COUNT + i * 2 * BEAT)
        add(kick(0.5), T_COUNT + i * 2 * BEAT)
    # 2027! 크래시 + 불꽃 쿵
    cn = int(2.5 * SR); ct = np.arange(cn) / SR
    add((noise_hp(cn, 88, 2) * np.exp(-ct / 0.9) * 0.22).astype(np.float32), T_DROP)
    for ft in scene_fw_times[3:]:
        th = int(0.7 * SR); tth = np.arange(th) / SR
        add((np.sin(math.tau * 55 * tth) * np.exp(-tth / 0.25) * 0.18 + noise_hp(th, int(ft * 100), 1) * np.exp(-tth / 0.08) * 0.04).astype(np.float32), ft, 0)
    # 장면 전환 종소리
    for tt_, m in [(0.3, 84), (T_DAWN, 88), (T_END, 91)]:
        add(bell(midi_hz(m), 4.0, 0.08), tt_, 0.2)

    # 리버브
    def reverb(x, seed):
        rr = np.random.default_rng(seed)
        ir_n = int(1.8 * SR)
        t_ = np.arange(ir_n) / SR
        ir = rr.normal(size=ir_n).astype(np.float32) * np.exp(-t_ / 0.45)
        ir[: int(0.01 * SR)] = 0
        ir = np.convolve(ir, np.ones(8) / 8, mode="same")
        nfft = 1 << int(math.ceil(math.log2(len(x) + ir_n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
        return y / np.sqrt(np.sum(ir ** 2)) * 0.06

    L2 = L + reverb(L, 1) * 1.4
    R2 = R + reverb(R, 2) * 1.4
    out = np.stack([L2, R2], 1)[: int(DUR * SR)]
    t = np.arange(len(out)) / SR
    out *= (np.minimum(1, t / 0.3) * (1 - smoothstep(DUR - 2.5, DUR, t)))[:, None]
    out /= np.max(np.abs(out)) + 1e-9
    out = np.tanh(out * 1.3) / np.tanh(1.3) * 0.89
    pcm = (out * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- 렌더

SCENE = None


def render_chunk(args):
    idx, f0, f1, outdir = args
    path = os.path.join(outdir, f"city_seg_{idx:02d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-x264-params", "keyint=30", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(SCENE.frame(f / FPS).tobytes())
        if (f - f0) % 60 == 0:
            print(f"  [{idx}] {f}/{f1}", flush=True)
    p.stdin.close()
    p.wait()
    return path


def main():
    global SCENE
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--preview", default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    work = a.work or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(work, exist_ok=True)
    SCENE = Scene(a.fonts)
    if a.preview:
        for s in a.preview.split(","):
            SCENE.frame(float(s)).save(os.path.join(work, f"city_{float(s):05.1f}.png"))
        return
    wav = os.path.join(work, "city_music.wav")
    make_music(wav, SCENE.fw_times)
    print("음악 생성 완료", flush=True)
    nj = max(1, a.jobs)
    edges = [round(i * NFRAMES / nj) for i in range(nj + 1)]
    with Pool(nj) as pool:
        segs = pool.map(render_chunk, [(i, edges[i], edges[i + 1], work) for i in range(nj)])
    lst = os.path.join(work, "city_segs.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("완료:", a.out)


if __name__ == "__main__":
    main()
