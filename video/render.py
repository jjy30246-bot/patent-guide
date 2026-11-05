#!/usr/bin/env python3
"""신길튼튼365정형외과 확장 이전 · 감사 인사 영상 (수묵 모션그래픽)

1920x1080 · 30fps · 30초. 홍매화와 청록 대나무가 먹 번짐과 함께 그려지고,
꽃잎이 흩날리며 문구가 부드럽게 전환된다.

사용:  python3 render.py --fonts <폰트폴더> --out <출력.mp4> [--preview 초,초,...]
필요:  numpy, Pillow, ffmpeg
"""
import argparse
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
FPS = 30
DUR = 30.0
NFRAMES = int(FPS * DUR)
SS = 2  # 회화 레이어 슈퍼샘플링 배율
LM = 60  # 회화 레이어 여백 (시차 이동용)
W0_, H0_ = 1920, 1080

rng = np.random.default_rng(365)

# ---------------------------------------------------------------- utilities

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def sstep(e0, e1, x):  # scalar
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_out_back(x, s=1.7):
    x = min(max(x, 0.0), 1.0)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def value_noise(w, h, cell, seed):
    """부드러운 값 노이즈 (0..1)."""
    r = np.random.default_rng(seed)
    gw, gh = max(2, int(w / cell) + 2), max(2, int(h / cell) + 2)
    g = (r.random((gh, gw)) * 255).astype(np.uint8)
    img = Image.fromarray(g, "L").resize((int(gw * cell), int(gh * cell)), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32)[:h, :w] / 255.0


def fbm(w, h, base, octaves, seed):
    acc = np.zeros((h, w), np.float32)
    amp, tot, cell = 1.0, 0.0, base
    for o in range(octaves):
        acc += amp * value_noise(w, h, cell, seed + o * 17)
        tot += amp
        amp *= 0.5
        cell = max(2, cell / 2)
    return acc / tot


# ---------------------------------------------------------------- paper

def make_paper():
    base = np.array([244, 237, 222], np.float32) / 255
    low = fbm(W, H, 300, 4, 1) - 0.5
    fine = value_noise(W, H, 2, 7) - 0.5
    paper = base[None, None, :] * (1 + 0.05 * low[..., None] + 0.018 * fine[..., None])
    # 한지 섬유
    fib = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(fib)
    r = np.random.default_rng(3)
    for _ in range(2600):
        x, y = r.random() * W, r.random() * H
        a = r.random() * math.pi
        L = 8 + r.random() * 40
        pts = [(x, y)]
        for k in range(4):
            a += r.normal() * 0.35
            x += math.cos(a) * L / 4
            y += math.sin(a) * L / 4
            pts.append((x, y))
        d.line(pts, fill=int(25 + r.random() * 40), width=1)
    fib = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
    paper *= (1 - 0.10 * fib[..., None])
    # 비네팅
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = ((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.66)) ** 2
    paper *= (1 - 0.16 * np.clip(v, 0, 1.4)[..., None] ** 1.3)
    return np.clip(paper, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- paint layers

class Layer:
    """색(RGB) + 농도(alpha) + 드러나는 시각(T, 초) 를 가진 회화 레이어."""

    def __init__(self, name):
        self.name = name
        self.dabs = []   # (order, kind, data)

    def dab(self, t, x, y, r, col, a):
        self.dabs.append((t, 0, (x, y, r, col, a)))

    def poly(self, t, pts, col, a):
        self.dabs.append((t, 1, (pts, col, a)))

    def line(self, t, pts, col, a, w):
        self.dabs.append((t, 2, (pts, col, a, w)))

    def rasterize(self, texture_seed, dry=0.35, bleed_r=10, bleed_a=0.35):
        return self._rasterize(texture_seed, dry, bleed_r, bleed_a)

    def _rasterize(self, texture_seed, dry, bleed_r, bleed_a):
        M = LM
        W, H = W0_ + 2 * M, H0_ + 2 * M
        Ws, Hs = W * SS, H * SS
        rgb = Image.new("RGB", (Ws, Hs), (0, 0, 0))
        al = Image.new("L", (Ws, Hs), 0)
        tm = Image.new("F", (Ws, Hs), 999.0)
        dr, da, dt = ImageDraw.Draw(rgb), ImageDraw.Draw(al), ImageDraw.Draw(tm)
        # 늦은 것부터 그리면 겹치는 곳은 가장 이른 시각이 남는다
        for t, kind, d in sorted(self.dabs, key=lambda q: -q[0]):
            if kind == 0:
                x, y, r, col, a = d
                bb = [(x + M - r) * SS, (y + M - r) * SS, (x + M + r) * SS, (y + M + r) * SS]
                dr.ellipse(bb, fill=col); da.ellipse(bb, fill=int(a * 255)); dt.ellipse(bb, fill=float(t))
            elif kind == 1:
                pts, col, a = d
                p = [((px + M) * SS, (py + M) * SS) for px, py in pts]
                dr.polygon(p, fill=col); da.polygon(p, fill=int(a * 255)); dt.polygon(p, fill=float(t))
            else:
                pts, col, a, w = d
                p = [((px + M) * SS, (py + M) * SS) for px, py in pts]
                ww = max(1, int(w * SS))
                dr.line(p, fill=col, width=ww); da.line(p, fill=int(a * 255), width=ww); dt.line(p, fill=float(t), width=ww)
        # 다운샘플 (안티에일리어싱)
        rgb = np.asarray(rgb.resize((W, H), Image.BOX), np.float32) / 255
        a = np.asarray(al.resize((W, H), Image.BOX), np.float32) / 255
        T = np.asarray(tm, np.float32)[::SS, ::SS].copy()
        # 마른 붓 질감 + 먹 농담
        tex = fbm(W, H, 24, 4, texture_seed)
        grain = value_noise(W, H, 1.5, texture_seed + 5)
        a = a * np.clip(1 - dry * (tex - 0.35) * 1.6 - 0.18 * (grain - 0.5), 0.25, 1.15)
        a = np.clip(a, 0, 1)
        # 색 정규화 (premultiplied 해제)
        cov = np.asarray(al.resize((W, H), Image.BOX), np.float32)[..., None] / 255
        # rgb 는 커버리지가 낮은 곳에서 검정과 섞였으므로 보정
        rgb = np.where(cov > 0.02, rgb / np.maximum(cov, 1e-3) * (cov > 0) , rgb)
        rgb = np.clip(rgb, 0, 1)
        # 번짐: 블러된 알파 + 팽창된 시간맵
        blur = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(bleed_r)), np.float32) / 255
        bleed = np.clip(blur * 1.6, 0, 1) * bleed_a
        q = Image.fromarray(T[::4, ::4].copy(), "F")
        for _ in range(3):
            q = q.filter(ImageFilter.MinFilter(5))
        Tq = np.asarray(q, np.float32)
        Td = np.repeat(np.repeat(Tq, 4, 0), 4, 1)[:H, :W]
        # 번짐 색: 레이어 색을 블러하여 사용
        rgbb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(bleed_r * 1.5)), np.float32) / 255
        wsum = np.asarray(Image.fromarray((np.clip(cov[..., 0], 0, 1) * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(bleed_r * 1.5)), np.float32)[..., None] / 255
        rgbb = np.clip(rgbb / np.maximum(wsum, 1e-3), 0, 1)
        # 바운딩 박스로 잘라 연산량 절감
        m = (a > 0.003) | (bleed > 0.003)
        ys, xs = np.where(m)
        if len(xs) == 0:
            self.bbox = None
            return
        x0, x1 = max(0, xs.min() - 4), min(W, xs.max() + 5)
        y0, y1 = max(0, ys.min() - 4), min(H, ys.max() + 5)
        self.bbox = (y0, y1, x0, x1)
        sl = (slice(y0, y1), slice(x0, x1))
        self.rgb = rgb[sl]
        self.a = a[sl]
        self.T = T[sl]
        self.bleed = bleed[sl]
        self.Td = Td[sl]
        self.rgbb = rgbb[sl]
        fin = T[T < 900]
        self.tmax = float(fin.max()) if fin.size else 0
        self.tmin = float(fin.min()) if fin.size else 0

    def apply(self, img, t, dx=0, dy=0):
        if self.bbox is None or t < self.tmin - 0.1:
            return
        y0, y1, x0, x1 = self.bbox
        # 시차(parallax) 이동: 레이어 좌표 -> 화면 좌표
        Y0, X0 = y0 - LM + dy, x0 - LM + dx
        sy0, sx0 = max(0, -Y0), max(0, -X0)
        sy1, sx1 = min(y1 - y0, H - Y0), min(x1 - x0, W - X0)
        if sy1 <= sy0 or sx1 <= sx0:
            return
        cs = (slice(sy0, sy1), slice(sx0, sx1))
        sub = img[Y0 + sy0:Y0 + sy1, X0 + sx0:X0 + sx1]
        dt = t - self.T[cs]
        reveal = smoothstep(0.0, 0.14, dt)
        wet = 1 + 0.35 * np.exp(-np.clip(dt, 0, None) / 0.35) * (dt > 0)
        a = np.clip(self.a[cs] * reveal * wet, 0, 1)
        # 번짐: 빠르게 퍼졌다가 옅게 남는다
        dd = t - self.Td[cs]
        bl = self.bleed[cs] * smoothstep(0.0, 0.5, dd) * (0.45 + 0.55 * np.exp(-np.clip(dd, 0, None) / 1.1))
        # 먹은 종이에 곱해진다 (multiply)
        sub *= 1 - bl[..., None] * (1 - self.rgbb[cs])
        sub *= 1 - a[..., None] * (1 - self.rgb[cs])


# ---------------------------------------------------------------- 홍매화

INK = (40, 33, 31)
INK2 = (66, 54, 48)


def grow_branch(layer, x, y, ang, length, width, t0, speed, depth, blossoms, r, kink=0.42):
    """각진 매화 가지를 붓 점(dab)으로 그린다. t0: 시작 시각, speed: px/s."""
    step = 2.2
    n = int(length / step)
    t = t0
    next_kink = r.integers(18, 45)
    w0 = width
    children = []
    for i in range(n):
        s = i / max(1, n - 1)
        wcur = w0 * (1 - 0.78 * s) + 0.6
        if i == next_kink:
            ang += r.choice([-1, 1]) * kink * (0.5 + r.random())
            next_kink = i + r.integers(20, 55)
        ang += r.normal() * 0.03
        x += math.cos(ang) * step
        y += math.sin(ang) * step
        t += step / speed
        col = INK if r.random() < 0.85 else INK2
        layer.dab(t, x + r.normal() * 0.3, y + r.normal() * 0.3, wcur / 2, col, 0.92)
        # 이끼 점(태점)
        if depth >= 2 and r.random() < 0.012:
            for _ in range(r.integers(1, 3)):
                ox, oy = r.normal() * wcur * 0.6, r.normal() * wcur * 0.6
                layer.dab(t + 0.3, x + ox, y + oy - wcur * 0.4, 1.5 + r.random() * 2.5, (25, 22, 22), 0.95)
        if depth > 0 and i > n * 0.18 and r.random() < 0.022 * (1.2 if depth > 1 else 1.6):
            children.append((x, y, ang, t, wcur, s))
        if depth <= 1 and r.random() < 0.035 and i > 6:
            blossoms.append((x, y, t + 0.15, r))
    for (cx, cy, ca, ct, cw, s) in children:
        sign = r.choice([-1, 1])
        grow_branch(layer, cx, cy, ca + sign * (0.45 + r.random() * 0.6), length * (0.32 + 0.3 * r.random()) * (1 - s * 0.5),
                    cw * 0.75, ct, speed * 0.9, depth - 1, blossoms, r)
    # 가지 끝
    blossoms.append((x, y, t + 0.1, r))
    if r.random() < 0.6:
        blossoms.append((x + r.normal() * 8, y + r.normal() * 8, t + 0.2, r))


def make_blossom_sprite(radius, r, kind):
    S = 4
    R = radius
    size = int(R * 2.6)
    im = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    c = size * S / 2
    if kind == "bud":
        d = ImageDraw.Draw(im)
        rr = R * 0.45 * S
        d.ellipse([c - rr, c - rr * 1.15, c + rr, c + rr * 0.85], fill=(170, 22, 48, 235))
        d.ellipse([c - rr * 0.5, c + rr * 0.4, c + rr * 0.5, c + rr * 1.0], fill=(60, 30, 25, 230))
    else:
        base_ang = r.random() * math.tau
        tones = [(196, 30, 62), (210, 48, 78), (184, 24, 54), (220, 70, 95)]
        for k in range(5):
            layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            a = base_ang + k * math.tau / 5 + r.normal() * 0.12
            pr = R * (0.50 + 0.08 * r.random()) * S
            px, py = c + math.cos(a) * pr * 0.95, c + math.sin(a) * pr * 0.95
            col = tones[r.integers(0, len(tones))]
            # 꽃잎: 바깥쪽이 진한 수채 느낌
            for j, (sc, al) in enumerate([(1.0, 150), (0.82, 90), (0.55, 60)]):
                rr = pr * sc
                cc = tuple(min(255, int(v + j * 18)) for v in col)
                d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=cc + (al,))
            im = Image.alpha_composite(im, layer)
        d = ImageDraw.Draw(im)
        # 꽃술
        cr = R * 0.22 * S
        d.ellipse([c - cr, c - cr, c + cr, c + cr], fill=(120, 18, 36, 230))
        for k in range(9):
            a = r.random() * math.tau
            L = R * (0.45 + 0.25 * r.random()) * S
            ex, ey = c + math.cos(a) * L, c + math.sin(a) * L
            d.line([(c, c), (ex, ey)], fill=(70, 30, 26, 220), width=max(1, int(S * 0.9)))
            dr_ = S * 1.6
            d.ellipse([ex - dr_, ey - dr_, ex + dr_, ey + dr_], fill=(236, 190, 70, 255) if k % 2 else (40, 28, 24, 255))
        if kind == "half":  # 옆모습 꽃: 아래쪽 반을 꽃받침으로 가림
            d.pieslice([c - R * 0.7 * S, c - R * 0.2 * S, c + R * 0.7 * S, c + R * 1.0 * S], 20, 160,
                       fill=(70, 34, 28, 235))
    im = im.resize((size, size), Image.LANCZOS)
    arr = np.asarray(im, np.float32) / 255
    return arr


def in_text_zone(x, y):
    return 470 < x < 1450 and 330 < y < 790


def plum_stroke(L, pts, w0, w1, t0, speed, r):
    """제어점을 따라 굵기가 줄어드는 각진 매화 가지. 샘플 목록을 돌려준다."""
    segs = list(zip(pts[:-1], pts[1:]))
    total = sum(math.dist(a, b) for a, b in segs)
    acc, t, out = 0.0, t0, []
    for (ax, ay), (bx, by) in segs:
        d = math.dist((ax, ay), (bx, by))
        ang = math.atan2(by - ay, bx - ax)
        n = max(2, int(d / 2.0))
        for i in range(n):
            u = i / n
            x = ax + (bx - ax) * u + r.normal() * 0.35
            y = ay + (by - ay) * u + r.normal() * 0.35
            s = (acc + d * u) / total
            w = w0 + (w1 - w0) * s
            w *= 1 + 0.10 * math.sin(acc / 23 + u * 7) + 0.05 * r.normal()
            t = t0 + (acc + d * u) / speed
            col = INK if r.random() < 0.8 else INK2
            L.dab(t, x, y, w / 2, col, 0.93)
            # 마른 붓의 갈라진 가장자리
            if w > 10 and r.random() < 0.25:
                side = r.choice([-1, 1])
                L.dab(t, x + math.cos(ang + math.pi / 2) * side * w * 0.42, y + math.sin(ang + math.pi / 2) * side * w * 0.42,
                      w * 0.12, (30, 26, 24), 0.95)
            out.append((x, y, t, w, ang))
        acc += d
    # 태점
    for _ in range(int(total / 70)):
        x, y, t, w, ang = out[r.integers(0, len(out))]
        if w > 7:
            L.dab(t + 0.25, x + r.normal() * w * 0.3, y - w * 0.45 + r.normal() * 2, 2 + r.random() * 3.5, (22, 20, 20), 0.96)
    return out


def build_plum(blossoms_out):
    L = Layer("plum")
    r = np.random.default_rng(11)
    flowers = []  # (x, y, t)
    sp = 330
    trunk = plum_stroke(L, [(-50, 1110), (70, 960), (45, 850), (160, 720), (255, 655), (300, 540)], 64, 26, 0.8, sp, r)
    tA = trunk[-1][2]
    brA = plum_stroke(L, [(300, 540), (395, 470), (450, 372), (585, 318), (690, 262), (760, 250)], 26, 5, tA, sp * 0.9, r)
    brB = plum_stroke(L, [(300, 540), (262, 420), (335, 290), (300, 170), (350, 70)], 22, 5, tA + 0.05, sp * 0.9, r)
    tC = [q for q in trunk if q[1] < 735][0][2]
    brC = plum_stroke(L, [(160, 720), (290, 775), (440, 760)], 17, 4, tC, sp * 0.8, r)
    top = plum_stroke(L, [(-50, 40), (140, 92), (265, 66), (405, 150), (560, 138), (700, 205), (860, 215)], 30, 5, 1.5, sp * 0.9, r)
    branches = [brA, brB, brC, top, trunk[len(trunk) // 2:]]
    for br in branches:
        for k in range(0, len(br), 6):
            x, y, t, w, ang = br[k]
            if r.random() < (0.10 if w > 12 else 0.16):
                # 곧은 잔가지 (한 번 꺾임)
                a = ang + r.choice([-1, 1]) * (0.55 + r.random() * 0.55)
                l1 = 30 + r.random() * 70
                mx, my = x + math.cos(a) * l1, y + math.sin(a) * l1
                a2 = a + r.normal() * 0.35
                l2 = 20 + r.random() * 50
                ex, ey = mx + math.cos(a2) * l2, my + math.sin(a2) * l2
                if in_text_zone(ex, ey) or in_text_zone(mx, my):
                    continue
                tw = plum_stroke(L, [(x, y), (mx, my), (ex, ey)], max(3.0, w * 0.35), 1.6, t, sp * 0.7, r)
                for q in tw[::14]:
                    if r.random() < 0.55:
                        flowers.append((q[0], q[1], q[2]))
                flowers.append((ex, ey, tw[-1][2]))
            # 가지에 바로 붙은 꽃
            if w < 22 and r.random() < 0.20:
                flowers.append((x + r.normal() * w * 0.4, y + r.normal() * w * 0.4, t))
        x, y, t, w, ang = br[-1]
        flowers.append((x, y, t))
    rr = np.random.default_rng(5)
    for (x, y, t) in flowers:
        if x < -10 or y < -10 or x > W or y > H or in_text_zone(x, y):
            continue
        kind = rr.choice(["full", "half", "bud"], p=[0.66, 0.17, 0.17])
        rad = 13 + rr.random() * 11 if kind != "bud" else 9 + rr.random() * 5
        spr = make_blossom_sprite(rad, rr, kind)
        blossoms_out.append(dict(x=x + rr.normal() * 3, y=y + rr.normal() * 3, t=t + 0.1 + rr.random() * 0.3, spr=spr,
                                 sway=rr.random() * math.tau))
    return L


# ---------------------------------------------------------------- 청록 대나무

TEALS = [(31, 104, 100), (24, 86, 84), (44, 124, 116), (20, 72, 72)]


def leaf_poly(x, y, ang, length, width, nseg=6):
    """끝이 뾰족한 대나무 잎을 몇 조각의 다각형으로."""
    pts_l, pts_r, centers = [], [], []
    for i in range(nseg * 3 + 1):
        s = i / (nseg * 3)
        bend = 0.18 * s * s
        a = ang + bend
        cx = x + math.cos(ang) * length * s + math.cos(ang + math.pi / 2) * bend * length * 0.25
        cy = y + math.sin(ang) * length * s + math.sin(ang + math.pi / 2) * bend * length * 0.25
        w = width * (math.sin(math.pi * min(1, s * 1.15)) ** 0.7) * (1 - 0.55 * s) if s < 0.98 else 0
        w = max(w, 0.4)
        nx, ny = math.cos(a + math.pi / 2), math.sin(a + math.pi / 2)
        pts_l.append((cx + nx * w / 2, cy + ny * w / 2))
        pts_r.append((cx - nx * w / 2, cy - ny * w / 2))
        centers.append((cx, cy))
    segs = []
    for k in range(nseg):
        i0, i1 = k * 3, (k + 1) * 3
        segs.append(pts_l[i0:i1 + 1] + pts_r[i0:i1 + 1][::-1])
    return segs


def build_bamboo():
    L = Layer("bamboo")
    r = np.random.default_rng(77)
    stalks = [  # x 하단, 폭, 색, 시작시각, 기울기
        (1590, 24, TEALS[0], 6.4, -0.06),
        (1735, 32, TEALS[1], 6.0, 0.03),
        (1850, 18, TEALS[2], 6.9, 0.10),
        (1905, 12, TEALS[3], 7.4, -0.05),
    ]
    for si, (bx, bw, col, t0, tilt) in enumerate(stalks):
        y = H + 30
        x = bx
        t = t0
        seg_len = 120 + r.random() * 40
        nodes = []
        while y > -60:
            sl = seg_len * (0.85 + 0.3 * r.random())
            # 한 마디 = 한 붓질 (양끝 진하게)
            steps = int(sl / 2)
            for i in range(steps):
                s = i / steps
                yy = y - s * sl
                xx = x + tilt * (H - yy) * 0.12 + math.sin(yy / 300 + si) * 3
                dens = 0.72 + 0.22 * (abs(s - 0.5) * 2) ** 2
                c2 = tuple(int(v * (0.92 + 0.12 * (abs(s - 0.5) * 2))) for v in col)
                L.dab(t + s * 0.18, xx, yy, bw / 2 * (1 - 0.06 * s), c2, dens)
                if i % 3 == 0:  # 밝은 결
                    L.dab(t + s * 0.18 + 0.01, xx - bw * 0.18, yy, bw * 0.12, tuple(min(255, v + 45) for v in col), 0.5)
            t += 0.22
            y -= sl
            xx = x + tilt * (H - y) * 0.12 + math.sin(y / 300 + si) * 3
            # 마디
            L.line(t, [(xx - bw * 0.62, y + 2), (xx - bw * 0.1, y - 3), (xx + bw * 0.62, y + 1)], (18, 52, 50), 0.95, max(2, bw * 0.16))
            nodes.append((xx, y, t))
            y -= 5
            t += 0.03
        # 잔가지와 잎
        for ni, (nx_, ny_, nt) in enumerate(nodes):
            if ny_ > 720 or ny_ < -40 or r.random() < 0.25:
                continue
            side = -1 if r.random() < 0.6 else 1
            ta = math.pi / 2 + side * (1.05 + r.random() * 0.35)  # 위쪽 대각선
            ta = -ta if False else (-math.pi / 2 + side * (0.9 + r.random() * 0.4))
            tl = 60 + r.random() * 90
            ex, ey = nx_ + math.cos(ta) * tl, ny_ + math.sin(ta) * tl
            L.line(nt + 0.05, [(nx_, ny_), ((nx_ + ex) / 2 + r.normal() * 4, (ny_ + ey) / 2), (ex, ey)], (22, 64, 62), 0.9, 2.6)
            # 잎 무리: 아래로 늘어지는 个 / 介 모양
            nleaf = r.integers(3, 6)
            for k in range(nleaf):
                la = math.pi / 2 + side * (0.5 + r.random() * 1.1) * (1 if k % 2 else -0.6) + r.normal() * 0.15
                la = math.pi / 2 + (k - (nleaf - 1) / 2) * 0.55 + side * 0.35 + r.normal() * 0.1
                ll = 85 + r.random() * 80
                lw = 13 + r.random() * 7
                lc = TEALS[r.integers(0, 4)]
                lt = nt + 0.25 + k * 0.08
                for j, seg in enumerate(leaf_poly(ex, ey, la, ll, lw)):
                    L.poly(lt + j * 0.04, seg, lc, 0.88)
    return L


# ---------------------------------------------------------------- 원경 산 / 먹물 방울

def build_mountains():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for k, (base, amp, alpha, col, seed) in enumerate([
        (840, 120, 0.16, (90, 120, 118), 2),
        (930, 90, 0.24, (70, 96, 96), 9),
    ]):
        x = np.arange(W, dtype=np.float32)
        r = np.random.default_rng(seed)
        h = np.zeros(W, np.float32)
        for f, a in [(1 / 700, 1.0), (1 / 260, 0.45), (1 / 90, 0.15), (1 / 30, 0.05)]:
            h += a * np.sin(x * f * math.tau + r.random() * 10)
        ridge = base - amp * h / 1.6
        d = yy - ridge[None, :]
        a = smoothstep(-2, 6, d) * np.exp(-np.clip(d, 0, None) / 140) * alpha
        a *= 0.65 + 0.5 * fbm(W, H, 80, 3, seed + 3)
        a *= smoothstep(0, 380, xx) * smoothstep(W, W - 520, xx)
        T = 0.4 + 4.5 * (xx / W) + 1.2 * fbm(W, H, 160, 2, seed + 7) + k * 0.8
        out.append((a.astype(np.float32), np.array(col, np.float32) / 255, T.astype(np.float32)))
    return out


def build_inkdrop():
    cx, cy = W / 2, H / 2 - 10
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    n1 = fbm(W, H, 260, 3, 41) - 0.5
    n2 = fbm(W, H, 60, 2, 43) - 0.5
    dn = d * (1 + 0.30 * n1 + 0.08 * n2)
    gran = 0.70 + 0.38 * fbm(W, H, 90, 3, 47) + 0.10 * (value_noise(W, H, 3, 49) - 0.5)
    return dn.astype(np.float32), gran.astype(np.float32)


# ---------------------------------------------------------------- 꽃잎

def make_petal_sprite(size, r):
    S = 4
    w, h = int(size * S), int(size * 0.8 * S)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    col = [(204, 40, 70), (222, 72, 98), (190, 28, 58)][r.integers(0, 3)]
    for j, (sc, al) in enumerate([(1.0, 190), (0.78, 90), (0.5, 60)]):
        ex, ey = w * (1 - sc) / 2, h * (1 - sc) / 2
        cc = tuple(min(255, v + j * 22) for v in col)
        d.ellipse([ex, ey, w - ex, h - ey], fill=cc + (al,))
    # 끝의 홈
    d.ellipse([w * 0.86, h * 0.40, w * 1.08, h * 0.60], fill=(0, 0, 0, 0))
    return im.resize((max(2, int(size)), max(2, int(size * 0.8))), Image.LANCZOS)


def build_petals():
    r = np.random.default_rng(88)
    P = []
    n = 46
    for i in range(n):
        fg = i < 5
        size = (44 + r.random() * 18) if fg else (13 + r.random() * 14)
        P.append(dict(
            t0=2.2 + (i / n) * 26 + r.random() * 1.5 - (6 if fg else 0) * 0,
            x0=-120 + r.random() * 1000, y0=-60 + r.random() * 380,
            vx=55 + r.random() * 70, vy=38 + r.random() * 46,
            amp=22 + r.random() * 45, fr=0.25 + r.random() * 0.35, ph=r.random() * math.tau,
            rot=r.random() * 360, vr=(r.random() - 0.5) * 140, flip=0.5 + r.random() * 1.2,
            spr=make_petal_sprite(size, r), fg=fg,
        ))
        if fg:
            P[-1]["t0"] = 4 + i * 5.5
            P[-1]["vx"] *= 1.6
            P[-1]["vy"] *= 1.3
    return P


# ---------------------------------------------------------------- 텍스트

class Fonts:
    def __init__(self, d):
        self.body = os.path.join(d, "GowunBatang-Bold.ttf")
        self.reg = os.path.join(d, "GowunBatang-Regular.ttf")
        self.title = os.path.join(d, "Myeongjo-EB.ttf")
        self.brush = os.path.join(d, "Brush.ttf")

    def get(self, which, size):
        return ImageFont.truetype(getattr(self, which), size)


DARK = (40, 34, 32)
RED = (178, 30, 56)
TEALT = (24, 92, 88)
GOLD = (158, 118, 48)


def render_line(fonts, segs, font, size, spacing=0):
    """segs: [(text, color)] -> RGBA 이미지 (여백 포함)."""
    f = fonts.get(font, size)
    pad = int(size * 0.6)
    total = 0
    for txt, _ in segs:
        for ch in txt:
            total += f.getlength(ch) + spacing
    asc, desc = f.getmetrics()
    im = Image.new("RGBA", (int(total + pad * 2), asc + desc + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = pad
    boxes = []
    for txt, col in segs:
        for ch in txt:
            d.text((x, pad), ch, font=f, fill=col + (255,))
            adv = f.getlength(ch) + spacing
            if ch.strip():
                cp = int(pad * 0.5)
                ci = Image.new("RGBA", (int(adv + 2 * cp), im.size[1]), (0, 0, 0, 0))
                ImageDraw.Draw(ci).text((cp, pad), ch, font=f, fill=col + (255,))
                boxes.append((int(x) - cp, ci))
            x += adv
    return im, boxes


class TextItem:
    """글자 단위로 시차를 두고 떠오르는(블러→선명, 아래→위) 문구."""
    BLURS = (0.0, 1.5, 3.0, 5.0, 7.5)

    def __init__(self, img, cx, cy, t_in, t_out, d_in=0.9, d_out=0.8, wipe=True, rise=16, stagger=None,
                 shimmer=()):
        img, boxes = img
        self.img = img
        self.cx, self.cy = cx, cy
        self.t_in, self.t_out = t_in, t_out
        self.d_in, self.d_out = d_in, d_out
        self.rise = rise
        self.w, self.h = img.size
        n = max(1, len(boxes))
        self.stagger = stagger if stagger is not None else min(0.06, 1.1 / n)
        self.shimmer = shimmer
        self.chars = []
        for (a, crop) in boxes:
            levels = [crop if r == 0 else crop.filter(ImageFilter.GaussianBlur(r)) for r in self.BLURS]
            self.chars.append((a, levels))
        # 글자 영역이 겹치므로, 완전히 나타난 뒤에는 원본 한 장으로 그린다
        self.full_levels = [img if r == 0 else img.filter(ImageFilter.GaussianBlur(r)) for r in self.BLURS]
        self.alpha_full = np.asarray(img)[..., 3].astype(np.float32) / 255
        self.xs = np.arange(self.w, dtype=np.float32)[None, :]

    def end_in(self):
        return self.t_in + self.stagger * (len(self.chars) - 1) + self.d_in

    @staticmethod
    def _pick(levels, blur):
        k = min(range(len(TextItem.BLURS)), key=lambda q: abs(TextItem.BLURS[q] - blur))
        return levels[k]

    @staticmethod
    def _fade(im, a):
        if a >= 0.999:
            return im
        arr = np.asarray(im).copy()
        arr[..., 3] = (arr[..., 3].astype(np.float32) * a).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    def draw(self, canvas, t):
        if t < self.t_in or t > self.t_out + self.d_out:
            return
        ox, oy = int(self.cx - self.w / 2), int(self.cy - self.h / 2)
        pout = sstep(0, 1, (t - self.t_out) / self.d_out)
        if t < self.end_in():
            for i, (a, levels) in enumerate(self.chars):
                p = (t - self.t_in - i * self.stagger) / self.d_in
                if p <= 0:
                    continue
                e = ease_out_cubic(p)
                im = self._fade(self._pick(levels, 7.5 * (1 - e)), e)
                canvas.alpha_composite(im, (ox + a, oy + int(self.rise * (1 - e))))
            return
        im = self._pick(self.full_levels, 6 * pout)
        # 금빛 광택이 글자 위를 지나간다
        for (ts, dur) in self.shimmer:
            if ts <= t <= ts + dur:
                q = (t - ts) / dur
                band_x = -0.2 * self.w + 1.4 * self.w * ease_out_cubic(q) if False else -0.25 * self.w + 1.5 * self.w * q
                g = np.exp(-((self.xs - band_x) / (self.w * 0.07)) ** 2) * 0.85
                arr = np.asarray(im).astype(np.float32)
                gold = np.array([226, 186, 110], np.float32)
                hi = np.array([255, 240, 200], np.float32)
                g3 = g[..., None]
                core = np.exp(-((self.xs - band_x) / (self.w * 0.018)) ** 2)[..., None] * 0.7
                arr[..., :3] = arr[..., :3] * (1 - g3) + gold * g3
                arr[..., :3] = arr[..., :3] * (1 - core) + hi * core
                im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        im = self._fade(im, 1 - pout)
        canvas.alpha_composite(im, (ox, oy - int(10 * pout)))


def make_seal(fonts, size):
    S = 3
    s = size * S
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([s * 0.04, s * 0.04, s * 0.96, s * 0.96], radius=s * 0.08, fill=(186, 32, 44, 255))
    f = fonts.get("title", int(s * 0.36))
    for i, ch in enumerate("튼튼"):
        # 세로쓰기 (흰 글씨 = 음각)
        bb = d.textbbox((0, 0), ch, font=f)
        cw, chh = bb[2] - bb[0], bb[3] - bb[1]
        d.text((s / 2 - cw / 2 - bb[0], s * (0.30 + 0.40 * i) - chh / 2 - bb[1]), ch, font=f, fill=(250, 240, 230, 255))
    arr = np.asarray(im, np.float32)
    # 인주의 거친 질감
    n = fbm(s, s, 12, 3, 501)
    arr[..., 3] *= np.clip(0.55 + 0.9 * n, 0, 1) ** 0.6
    holes = value_noise(s, s, 3, 503) > 0.86
    arr[..., 3][holes] *= 0.4
    im = Image.fromarray(arr.astype(np.uint8), "RGBA").resize((size, size), Image.LANCZOS)
    return im


def build_texts(fonts):
    T = []
    cx = W / 2
    # 1. 감사
    T.append(TextItem(render_line(fonts, [("늘 곁에서 함께해 주신 여러분께", DARK)], "body", 60, 2), cx, 480, 3.3, 8.5))
    T.append(TextItem(render_line(fonts, [("진심으로 ", DARK), ("감사", RED), ("드립니다", DARK)], "body", 60, 2), cx, 580, 3.9, 8.6))
    # 2. 확장 이전
    T.append(TextItem(render_line(fonts, [("신길튼튼365정형외과", TEALT), ("가", DARK)], "body", 62, 2), cx, 470, 9.3, 14.8))
    T.append(TextItem(render_line(fonts, [("새로운 공간으로 ", DARK), ("확장 이전", RED), ("합니다", DARK)], "body", 60, 2), cx, 575, 9.9, 14.9))
    T.append(TextItem(render_line(fonts, [("2026년 1월 · 영등포구 가마산로 483", TEALT)], "reg", 38, 2), cx, 668, 10.8, 15.0, rise=10))
    # 3. 다짐
    T.append(TextItem(render_line(fonts, [("더 넓어진 공간, 더 편안한 진료", DARK)], "body", 58, 2), cx, 475, 15.5, 21.0))
    T.append(TextItem(render_line(fonts, [("한결같은 마음으로 여러분의 건강을 지키겠습니다", DARK)], "reg", 44, 1), cx, 570, 16.1, 21.1))
    # 4. 엔딩
    T.append(TextItem(render_line(fonts, [("2026. 01  확 장 이 전", GOLD)], "body", 34, 10), cx, 335, 21.9, 40, rise=10))
    T.append(TextItem(render_line(fonts, [("신길튼튼365정형외과", DARK)], "title", 100, 3), cx, 440, 22.3, 40, d_in=1.1, rise=26, stagger=0.09, shimmer=((24.9, 1.5), (27.6, 1.3))))
    T.append(TextItem(render_line(fonts, [("서울특별시 영등포구 가마산로 483  3A01호~3A05호", TEALT)], "body", 40, 1), cx, 578, 23.3, 40))
    T.append(TextItem(render_line(fonts, [("그동안 보내주신 성원에 깊이 감사드립니다", DARK)], "reg", 40, 2), cx, 652, 24.0, 40))
    T.append(TextItem(render_line(fonts, [("감사합니다", RED)], "brush", 84, 2), cx, 748, 25.3, 40, d_in=1.0, stagger=0.16))
    return T


# ---------------------------------------------------------------- scene

class Scene:
    def __init__(self, fonts_dir):
        print("페이퍼/레이어 생성 중...", flush=True)
        self.paper = make_paper()
        self.blossoms = []
        self.plum = build_plum(self.blossoms)
        self.plum.rasterize(101, dry=0.45, bleed_r=11, bleed_a=0.30)
        self.bamboo = build_bamboo()
        self.bamboo.rasterize(202, dry=0.30, bleed_r=13, bleed_a=0.34)
        self.mountains = build_mountains()
        self.drop_d, self.drop_gran = build_inkdrop()
        self.petals = build_petals()
        self.fonts = Fonts(fonts_dir)
        self.texts = build_texts(self.fonts)
        self.seal = make_seal(self.fonts, 96)
        # 엔딩 장식선
        self.rule = Image.new("RGBA", (420, 6), (0, 0, 0, 0))
        dd = ImageDraw.Draw(self.rule)
        dd.line([(0, 3), (420, 3)], fill=(176, 138, 66, 210), width=2)
        # 흐르는 안개
        mist = fbm(W + 900, H, 260, 4, 909)
        yy = np.arange(H, dtype=np.float32)[:, None]
        band = np.exp(-((yy - 860) / 150) ** 2) * 0.9 + np.exp(-((yy - 120) / 120) ** 2) * 0.35
        self.mist = (smoothstep(0.42, 0.78, mist) * band).astype(np.float32)
        self.mist_col = np.array([0.985, 0.975, 0.955], np.float32)
        # 따뜻한 빛 번짐 (light leak)
        gy, gx = np.mgrid[0:H * 2:2, 0:W * 2:2].astype(np.float32)
        gl = np.exp(-(((gx - W) / 900) ** 2 + ((gy - H) / 620) ** 2))
        self.glow = Image.fromarray((gl * 255).astype(np.uint8), "L").resize((W * 2, H * 2), Image.BILINEAR)
        self.glow = np.asarray(self.glow, np.float32) / 255
        self.glow_col = np.array([1.0, 0.86, 0.66], np.float32)
        # 금빛 먼지
        r = np.random.default_rng(1234)
        self.dust = [dict(x=r.random() * W, y=r.random() * H, vy=-(6 + r.random() * 16), vx=(r.random() - 0.5) * 8,
                          amp=8 + r.random() * 20, fr=0.1 + r.random() * 0.25, ph=r.random() * math.tau,
                          tw=0.6 + r.random() * 1.6, size=r.integers(0, 3), t0=0.8 + r.random() * 6,
                          late=r.random() < 0.4) for _ in range(90)]
        self.dust_spr = []
        for sz in (7, 11, 17):
            g = np.mgrid[0:sz, 0:sz].astype(np.float32) - (sz - 1) / 2
            rr = np.hypot(g[0], g[1]) / (sz / 2)
            al = np.clip(np.exp(-rr ** 2 * 4) + 0.9 * np.exp(-rr ** 2 * 30), 0, 1)
            arr = np.zeros((sz, sz, 4), np.uint8)
            arr[..., 0], arr[..., 1], arr[..., 2] = 232, 196, 120
            arr[..., 3] = (al * 255).astype(np.uint8)
            self.dust_spr.append(arr)
        print(f"매화 꽃 {len(self.blossoms)}송이, 꽃잎 {len(self.petals)}장", flush=True)

    def frame_overlay(self, t):
        """모서리에서 시작해 그려지는 이중 금선 테두리."""
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        p = ease_out_cubic((t - 21.3) / 2.2)
        a = int(200 * sstep(21.3, 22.0, t))
        for inset, wd, al in ((44, 2, 1.0), (56, 1, 0.7)):
            x0, y0, x1, y1 = inset, inset, W - inset, H - inset
            hx, hy = (x1 - x0) / 2 * p, (y1 - y0) / 2 * p
            col = (176, 138, 66, int(a * al))
            for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                d.line([(cx, cy), (cx + sx * hx, cy)], fill=col, width=wd)
                d.line([(cx, cy), (cx, cy + sy * hy)], fill=col, width=wd)
        # 모서리 마름모 장식
        r = 7 * p
        for (cx, cy) in ((50, 50), (W - 50, 50), (50, H - 50), (W - 50, H - 50)):
            d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(186, 146, 70, a))
        return ov

    def frame(self, t):
        img = self.paper.copy()
        # 원경 산 (안개처럼 번지며)
        for a, col, T in self.mountains:
            m = a * smoothstep(0, 1.6, t - T)
            img *= 1 - m[..., None] * (1 - col)
        # 흐르는 안개 (오른쪽에서 왼쪽으로)
        off = int(18 * t) % 900
        m = self.mist[:, off:off + W] * (0.55 * sstep(0.5, 4.0, t))
        img += (self.mist_col - img) * m[..., None] * 0.6
        # 먹물 방울: 화면 중앙에서 번졌다가 옅어짐
        if 0.3 < t < 9.5:
            p = ease_out_cubic((t - 0.35) / 2.6)
            rad = 40 + 520 * p
            dens = 0.62 * (1 - sstep(0.5, 3.4, t)) + 0.05 * (1 - sstep(5.5, 9.4, t))
            edge = rad - self.drop_d
            core = smoothstep(-6, 30, edge)
            ring = np.exp(-((edge - 6) / 16) ** 2) * 0.55 * (1 - sstep(1.0, 4.0, t))
            a = np.clip((core * dens + ring * dens) * self.drop_gran, 0, 0.9)
            ink = np.array([0.16, 0.15, 0.16], np.float32)
            img *= 1 - a[..., None] * (1 - ink)
        # 매화 가지, 대나무
        pd = sstep(0, DUR, t)
        pdx, pdy = int(round(34 * pd)), int(round(12 * pd))
        self.plum.apply(img, t, pdx, pdy)
        self.bamboo.apply(img, t, int(round(-30 * pd)), int(round(-6 * pd)))
        # 엔딩: 제목 뒤 옅은 먹 번짐
        if t > 21.4:
            p = ease_out_cubic((t - 21.4) / 3.0)
            edge = 80 + 560 * p - self.drop_d
            a = smoothstep(-10, 60, edge) * 0.07 * self.drop_gran
            img *= 1 - a[..., None] * (1 - np.array([0.5, 0.56, 0.55], np.float32))

        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        # 매화 꽃: 가지가 닿으면 피어난다
        for b in self.blossoms:
            k = (t - b["t"]) / 0.7
            if k <= 0:
                continue
            s = ease_out_back(k, 2.0) if k < 1 else 1.0
            spr = b["spr"]
            hh, ww = spr.shape[:2]
            if s < 0.995:
                nw, nh = max(1, int(ww * s)), max(1, int(hh * s))
                im = Image.fromarray((spr * 255).astype(np.uint8), "RGBA").resize((nw, nh), Image.BILINEAR)
            else:
                im = b.get("_pil")
                if im is None:
                    im = Image.fromarray((spr * 255).astype(np.uint8), "RGBA")
                    b["_pil"] = im
            if k < 1:
                arr = np.asarray(im).copy()
                arr[..., 3] = (arr[..., 3] * min(1, k * 2.5)).astype(np.uint8)
                im = Image.fromarray(arr, "RGBA")
            canvas.alpha_composite(im, (int(b["x"] + pdx - im.size[0] / 2), int(b["y"] + pdy - im.size[1] / 2)))

        # 카메라: 아주 느린 줌 인 + 미세한 패닝
        z = 1.02 + 0.045 * sstep(0, DUR, t)
        cw, ch = W / z, H / z
        ox = (W - cw) / 2 + 10 * math.sin(t * 0.21)
        oy = (H - ch) / 2 + 6 * math.sin(t * 0.17 + 1)
        ox = min(max(ox, 0), W - cw); oy = min(max(oy, 0), H - ch)
        canvas = canvas.resize((W, H), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))

        # 흩날리는 꽃잎
        for p in self.petals:
            dt = t - p["t0"]
            if dt < 0:
                continue
            x = p["x0"] + p["vx"] * dt + p["amp"] * math.sin(p["fr"] * math.tau * dt + p["ph"])
            y = p["y0"] + p["vy"] * dt + 0.5 * p["amp"] * math.sin(p["fr"] * 0.7 * math.tau * dt + p["ph"] * 1.3)
            if x > W + 80 or y > H + 80:
                continue
            spr = p["spr"]
            sx = abs(math.cos(p["flip"] * dt + p["ph"]))
            sx = 0.25 + 0.75 * sx
            sw = max(2, int(spr.size[0] * sx))
            im = spr.resize((sw, spr.size[1]), Image.BILINEAR).rotate(p["rot"] + p["vr"] * dt, Image.BICUBIC, expand=True)
            fade = sstep(0, 0.8, dt)
            if p["fg"]:
                im = im.filter(ImageFilter.GaussianBlur(3.2))
                fade *= 0.75
            if fade < 0.999:
                arr = np.asarray(im).copy()
                arr[..., 3] = (arr[..., 3] * fade).astype(np.uint8)
                im = Image.fromarray(arr, "RGBA")
            canvas.alpha_composite(im, (int(x - im.size[0] / 2), int(y - im.size[1] / 2)))

        # 금빛 먼지: 천천히 떠오르며 반짝인다
        for d in self.dust:
            if t < d["t0"] or (d["late"] and t < 20.5):
                continue
            tt = t - (20.5 if d["late"] else d["t0"])
            x = (d["x"] + d["vx"] * tt + d["amp"] * math.sin(d["fr"] * math.tau * tt + d["ph"])) % W
            y = (d["y"] + d["vy"] * tt) % H
            a = sstep(0, 1.5, tt) * (0.25 + 0.75 * math.sin(d["tw"] * tt + d["ph"]) ** 2)
            arr = self.dust_spr[d["size"]].copy()
            arr[..., 3] = (arr[..., 3] * a * 0.85).astype(np.uint8)
            im = Image.fromarray(arr, "RGBA")
            canvas.alpha_composite(im, (int(x), int(y)))
        # 금박 테두리 (엔딩)
        if t > 21.3:
            canvas.alpha_composite(self.frame_overlay(t))
        # 문구
        for ti in self.texts:
            ti.draw(canvas, t)
        # 엔딩 장식선 + 낙관
        if t > 23.0:
            p = ease_out_cubic((t - 23.0) / 1.2)
            rw = int(420 * p)
            if rw > 2:
                r_ = self.rule.crop(((420 - rw) // 2, 0, (420 + rw) // 2, 6))
                canvas.alpha_composite(r_, (int(W / 2 - rw / 2), 516))
        if t > 24.3:
            k = (t - 24.3) / 0.45
            s = 1.6 - 0.6 * ease_out_cubic(k) if k < 1 else 1.0
            if k >= 1:
                s = 1.0 + 0.04 * math.exp(-(t - 24.75) * 6) * math.sin((t - 24.75) * 30)
            a = min(1, k * 1.8) * 0.92
            sz = max(4, int(96 * s))
            im = self.seal.resize((sz, sz), Image.BILINEAR).rotate(-4, Image.BICUBIC, expand=True)
            arr = np.asarray(im).copy()
            arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
            im = Image.fromarray(arr, "RGBA")
            ti = next(x for x in self.texts if x.t_in == 22.3)
            sx_ = W / 2 + (ti.w - 2 * int(100 * 0.6)) / 2 + 74
            canvas.alpha_composite(im, (int(sx_ - im.size[0] / 2), int(446 - im.size[1] / 2)))

        out = canvas.convert("RGB")
        # 따뜻한 빛 번짐이 화면을 천천히 가로지른다 (screen 합성)
        gx = int(W * (0.15 + 0.6 * sstep(0, DUR, t)) )
        gy = int(H * (0.25 + 0.15 * math.sin(t * 0.2)))
        g = self.glow[H - gy:2 * H - gy, W - gx:2 * W - gx] * (0.10 + 0.05 * math.sin(t * 0.5))
        o = np.asarray(out, np.float32) / 255
        o = 1 - (1 - o) * (1 - g[..., None] * self.glow_col)
        out = Image.fromarray((np.clip(o, 0, 1) * 255).astype(np.uint8), "RGB")
        # 페이드 인/아웃
        f_in = sstep(0, 0.8, t)
        f_out = 1 - sstep(DUR - 1.4, DUR - 0.05, t)
        if f_in < 1:
            out = Image.blend(Image.new("RGB", (W, H), (250, 247, 240)), out, f_in)
        if f_out < 1:
            out = Image.blend(Image.new("RGB", (W, H), (250, 247, 240)), out, f_out)
        return out


# ---------------------------------------------------------------- 음악

SR = 44100


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n, a, d, s, r_, sr=SR):
    e = np.ones(n, np.float32) * s
    na, nd, nr = int(a * sr), int(d * sr), int(r_ * sr)
    na = min(na, n); e[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na); e[na:na + nd] = np.linspace(1, s, nd)
    if nr > 0 and nr < n:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def pluck(freq, dur, amp=0.3):
    """가야금 느낌의 뜯는 소리: 배음 가산 합성 + 농현(늦은 비브라토)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(math.tau * 5.2 * t) * smoothstep(0.35, 0.9, t)
    bend = 1 + 0.012 * np.exp(-t / 0.04)
    phase = math.tau * np.cumsum(freq * vib * bend) / SR
    y = np.zeros(n, np.float32)
    for h, (a, dec) in enumerate([(1.0, 1.4), (0.55, 0.8), (0.32, 0.45), (0.2, 0.3), (0.12, 0.2), (0.07, 0.14)], 1):
        y += a * np.sin(h * phase) * np.exp(-t / dec)
    att = np.minimum(1, t / 0.004)
    rel = np.ones(n); rel[-int(0.05 * SR):] = np.linspace(1, 0, int(0.05 * SR))
    return (y * att * rel * amp).astype(np.float32)


def pad(freqs, dur, amp=0.08):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, np.float32)
    for f in freqs:
        for det in (-0.12, 0.0, 0.11):
            ff = f * 2 ** (det / 12)
            ph = np.random.default_rng(int(f)).random() * math.tau
            y += np.sin(math.tau * ff * t + ph) + 0.18 * np.sin(2 * math.tau * ff * t) + 0.05 * np.sin(3 * math.tau * ff * t)
    y *= env_adsr(n, 1.8, 0.5, 0.85, 2.2) * (1 + 0.08 * np.sin(math.tau * 0.23 * t))
    return (y / (len(freqs) * 3) * amp * 6).astype(np.float32)


def bell(freq, dur, amp=0.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    mod = np.sin(math.tau * freq * 3.5 * t) * 2.2 * np.exp(-t / 0.6)
    y = np.sin(math.tau * freq * t + mod) * np.exp(-t / 1.6)
    y += 0.3 * np.sin(math.tau * freq * 2.76 * t) * np.exp(-t / 0.5)
    return (y * np.minimum(1, t / 0.002) * amp).astype(np.float32)


def make_music(path):
    n = int((DUR + 0.5) * SR)
    L = np.zeros(n + SR * 6, np.float32)
    R = np.zeros_like(L)
    bpm = 70
    beat = 60 / bpm

    def add(sig, t, pan=0.0):
        i = int(t * SR)
        g_l, g_r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        L[i:i + len(sig)] += sig * g_l
        R[i:i + len(sig)] += sig * g_r

    # 화성 진행 (8박 단위): D - Bm - G - A(sus) - D
    chords = [
        (0, [50, 57, 62, 66, 64]),
        (8, [47, 54, 59, 62, 69]),
        (16, [43, 50, 55, 59, 69]),
        (24, [45, 52, 57, 62, 64]),
        (31, [50, 57, 62, 66, 69]),
    ]
    for i, (b, notes) in enumerate(chords):
        b2 = chords[i + 1][0] if i + 1 < len(chords) else 36
        d = (b2 - b) * beat + 2.2
        add(pad([midi_hz(m) for m in notes], d, 0.09), b * beat, 0.0)
        add(pad([midi_hz(notes[0] - 12)], d, 0.07), b * beat, 0.0)
    # 잔잔한 아르페지오 (8분음표)
    r = np.random.default_rng(7)
    for i, (b, notes) in enumerate(chords[:-1]):
        b2 = chords[i + 1][0]
        tones = sorted(notes[1:])
        pat = [0, 1, 2, 3, 2, 1, 2, 3]
        k = 0
        x = b + (1 if b == 0 else 0)
        while x < b2:
            m = tones[pat[k % len(pat)] % len(tones)]
            add(pluck(midi_hz(m), 2.2, 0.075 * (0.8 + 0.3 * r.random())), x * beat + r.normal() * 0.006, -0.35 + 0.7 * r.random())
            x += 0.5
            k += 1
    # 선율 (D 5음 음계)
    mel = [(2, 69, 1), (3, 71, .5), (3.5, 74, .5), (4, 76, 1.5), (5.5, 74, .5), (6, 71, 1), (7, 69, 2),
           (10, 74, 1), (11, 76, .5), (11.5, 78, .5), (12, 81, 1.5), (13.5, 78, .5), (14, 76, 1), (15, 74, 2),
           (18, 71, 1), (19, 74, .5), (19.5, 76, .5), (20, 78, 1), (21, 76, 1), (22, 74, 1), (23, 71, 1), (24, 69, 3),
           (26, 74, 1), (27, 78, 1), (28, 81, 1.5), (29.5, 78, .5), (30, 76, 1), (31, 74, 4)]
    for b, m, d in mel:
        add(pluck(midi_hz(m), max(1.6, d * beat + 1.2), 0.20), b * beat, 0.12)
        add(pluck(midi_hz(m + 12), max(1.0, d * beat + 0.6), 0.035), b * beat + 0.01, 0.4)
    # 풍경(종) 소리: 장면 전환 지점
    for tt, m, a in [(0.4, 86, 0.10), (6.3, 81, 0.07), (9.2, 86, 0.08), (15.4, 83, 0.07), (21.8, 86, 0.10), (24.3, 74, 0.12)]:
        add(bell(midi_hz(m), 4.0, a), tt, 0.25)

    # 리버브 (지수 감쇠 노이즈 IR 컨볼루션)
    def reverb(x, seed):
        rr = np.random.default_rng(seed)
        ir_n = int(2.8 * SR)
        tt = np.arange(ir_n) / SR
        ir = rr.normal(size=ir_n).astype(np.float32) * np.exp(-tt / 0.75)
        ir[: int(0.012 * SR)] = 0
        # 고역 감쇠
        k = 12
        ir = np.convolve(ir, np.ones(k) / k, mode="same")
        nfft = 1 << int(math.ceil(math.log2(len(x) + ir_n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
        return y / np.sqrt(np.sum(ir ** 2)) * 0.06

    L2 = L + reverb(L, 1) * 2.2
    R2 = R + reverb(R, 2) * 2.2
    out = np.stack([L2, R2], 1)[: int(DUR * SR)]
    t = np.arange(len(out)) / SR
    out *= (np.minimum(1, t / 0.6) * (1 - smoothstep(DUR - 3.0, DUR, t)))[:, None]
    out /= np.max(np.abs(out)) + 1e-9
    out = np.tanh(out * 1.1) / np.tanh(1.1) * 0.89
    pcm = (out * 32767).astype("<i2")
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- 렌더

SCENE = None


def render_chunk(args):
    idx, f0, f1, outdir = args
    path = os.path.join(outdir, f"seg_{idx:02d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-x264-params", "keyint=30", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        im = SCENE.frame(f / FPS)
        p.stdin.write(im.tobytes())
        if (f - f0) % 30 == 0:
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
    ap.add_argument("--preview", default=None, help="쉼표로 구분한 시각(초) 스틸 이미지만 출력")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    work = a.work or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(work, exist_ok=True)
    SCENE = Scene(a.fonts)
    if a.preview:
        for s in a.preview.split(","):
            SCENE.frame(float(s)).save(os.path.join(work, f"still_{float(s):05.1f}.png"))
        return
    wav = os.path.join(work, "music.wav")
    make_music(wav)
    print("음악 생성 완료", flush=True)
    nj = max(1, a.jobs)
    edges = [round(i * NFRAMES / nj) for i in range(nj + 1)]
    jobs = [(i, edges[i], edges[i + 1], work) for i in range(nj)]
    with Pool(nj) as pool:
        segs = pool.map(render_chunk, jobs)
    lst = os.path.join(work, "segs.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("완료:", a.out)


if __name__ == "__main__":
    main()
