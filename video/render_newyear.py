#!/usr/bin/env python3
"""신길튼튼365정형외과 · 2027 새해맞이 홍보 영상 (수묵 + 전통 + 세련된 모션그래픽)

1920x1080 · 30fps · 60초 · 국악풍 배경음악

장면
 1) 수묵 산수: 붉은 해와 구름, 운해 위 겹겹의 산, 벼랑 위 소나무, 홍매화, 까치 한 쌍
 2) 눈 덮인 한옥 대청: 병풍 앞에서 한복 입은 남녀가 세배
 3) 수묵 대나무: 잎이 하나씩 그려짐
 4) 설날 상차림: 떡국·전·과일·약과·식혜 앞에서 행복한 두 사람
 5) 엔딩: 산수 + 까치 + 인사말 + 병원명 + 낙관
장면 사이는 먹이 번지듯 전환된다.

사용:  python3 render_newyear.py --fonts <폰트폴더> --out <출력.mp4> [--preview 초,...]
"""
import argparse
import math
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render as R
from render import (W, H, FPS, smoothstep, sstep, ease_out_cubic, ease_out_back, fbm, value_noise,
                    Layer, make_paper, render_line, TextItem, Fonts, make_seal, plum_stroke, make_blossom_sprite, leaf_poly)
import render_seollal as RS
from render_seollal import draw_person, ease_io, bez

DUR = 60.0
NFRAMES = int(FPS * DUR)
DARK = (40, 34, 32)
RED = (186, 32, 44)
TEAL = (30, 92, 88)

# 장면 경계 (전환 시작, 전환 끝)
X12 = (11.6, 13.2)
X23 = (27.6, 29.2)
X34 = (37.6, 39.2)
X45 = (49.6, 51.2)


# ---------------------------------------------------------------- 공통 수묵 도구

def poly_mask(pts, blur=0):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im, np.float32) / 255


def streaks(seed, cell=40, stretch=6):
    """세로로 긴 준법(皴法) 느낌의 결."""
    h2 = max(8, H // stretch)
    n = fbm(W, h2, cell, 4, seed)
    return np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255


def mountain_range(base, amp, seed, col, alpha, t0, t1, x0=-100, x1=W + 100, fog=110):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.random.default_rng(seed)
    x = np.arange(W, dtype=np.float32)
    h = np.zeros(W, np.float32)
    for f, a in [(1 / 640, 1.0), (1 / 260, 0.55), (1 / 90, 0.14), (1 / 30, 0.04)]:
        h += a * np.sin(x * f * math.tau + r.random() * 10)
    h = h / 1.73
    peak = np.clip((h + 1) / 2, 0, 1) ** 2.0
    ridge = base - amp * peak
    taper = smoothstep(x0, x0 + 260, x) * smoothstep(x1, x1 - 260, x)
    ridge = base - (base - ridge) * taper
    d = yy - ridge[None, :]
    st = streaks(seed + 1, 60, 3)
    body = smoothstep(-1.5, 3, d) * np.exp(-np.clip(d, 0, None) / fog)
    edge = np.exp(-np.clip(d, 0, None) / 8) * smoothstep(-1.5, 2, d) * 0.5
    a = (body * (0.7 + 0.35 * st) + edge) * alpha
    a *= taper[None, :] * 0.9 + 0.1
    T = t0 + (t1 - t0) * (0.5 + 0.5 * np.sin(xx / W * 2.2)) * 0.6 + (t1 - t0) * 0.4 * fbm(W, H, 180, 2, seed + 5)
    return dict(a=np.clip(a, 0, 1).astype(np.float32), col=np.array(col, np.float32) / 255, T=T.astype(np.float32))


def apply_wash(img, w, t, dx=0):
    m = w["a"] * smoothstep(0, 1.4, t - w["T"])
    if dx:
        m = np.roll(m, dx, axis=1)
    img *= 1 - m[..., None] * (1 - w["col"])


def lighten(img, a, col=(0.985, 0.975, 0.955)):
    img += (np.array(col, np.float32) - img) * a[..., None]


def ink_dissolve(A, B, p, field):
    v = p * 1.25 - field
    m = smoothstep(0.0, 0.09, v)[..., None]
    edge = (np.exp(-((v - 0.01) / 0.03) ** 2) * 0.30 * math.sin(math.pi * min(max(p, 0), 1)))[..., None]
    out = A * (1 - m) + B * m
    out *= 1 - edge * (1 - np.array([0.42, 0.40, 0.38], np.float32))
    return out


# ---------------------------------------------------------------- 까치

def draw_magpie(d, x, y, s, flap, f=1):
    """옆에서 본 까치: 검은 머리·등, 흰 배와 날개 무늬, 긴 꼬리 (오른쪽으로 비행)."""
    def P(px, py):
        return (x + px * s, y + py * s)
    blk = (20, 22, 28, 255)
    wy = -38 * flap + 22 * (1 - flap)
    # 뒷날개
    d.polygon([P(2, -3), P(14, wy * 0.8), P(-6, wy * 0.72), P(-10, -1)], fill=(34, 46, 70, 255))
    # 꼬리
    d.polygon([P(-16, -3), P(-60, -9), P(-64, -3), P(-60, 3), P(-16, 4)], fill=blk)
    d.line([P(-22, 0), P(-60, -3)], fill=(60, 86, 120, 255), width=max(1, int(1.4 * s)))
    # 몸통 + 흰 배
    d.ellipse([P(-22, -9), P(20, 9)], fill=blk)
    d.ellipse([P(-6, -2), P(16, 9)], fill=(248, 248, 246, 255))
    # 머리 + 부리
    d.ellipse([P(13, -13), P(30, 3)], fill=blk)
    d.polygon([P(28, -7), P(40, -5), P(28, -2)], fill=(24, 24, 24, 255))
    d.ellipse([P(22, -9), P(25, -6)], fill=(230, 230, 230, 255))
    # 앞날개 (흰 띠)
    d.polygon([P(6, -4), P(18, wy), P(-4, wy * 0.92), P(-12, -2)], fill=blk)
    d.polygon([P(4, -5 + wy * 0.3), P(10, wy * 0.75), P(1, wy * 0.72), P(-4, -4 + wy * 0.3)], fill=(246, 246, 244, 255))


def magpie_pair(d, t, t0, t1, y0, s=1.0):
    if not (t0 < t < t1):
        return
    q = (t - t0) / (t1 - t0)
    for k, (dx, dy, ph) in enumerate(((0, 0, 0.0), (-120, 46, 1.7))):
        x = -120 + (W + 260) * q + dx
        y = y0 + dy - 50 * math.sin(q * math.pi) + 10 * math.sin(t * 2 + ph)
        flap = 0.5 + 0.5 * math.sin(t * 9 + ph * 3)
        draw_magpie(d, x, y, s * (1 - 0.12 * k), flap, f=1)


# ---------------------------------------------------------------- 장면 1 · 산수

def build_pine_cliff():
    L = Layer("pine")
    L.true_color = True
    r = np.random.default_rng(777)
    bark = [(96, 70, 52), (80, 58, 44), (112, 84, 62)]

    def stroke(pts, w0, w1, t0, speed):
        segs = list(zip(pts[:-1], pts[1:]))
        total = sum(math.dist(a, b) for a, b in segs)
        acc, out = 0.0, []
        for (ax, ay), (bx, by) in segs:
            dd = math.dist((ax, ay), (bx, by))
            n = max(2, int(dd / 2))
            for i in range(n):
                u = i / n
                x, y = ax + (bx - ax) * u, ay + (by - ay) * u
                s = (acc + dd * u) / total
                w = (w0 + (w1 - w0) * s) * (1 + 0.1 * math.sin((acc + dd * u) / 15))
                t = t0 + (acc + dd * u) / speed
                L.dab(t, x + r.normal() * 0.5, y + r.normal() * 0.5, w / 2, bark[r.integers(0, 3)], 0.92)
                if w > 12 and r.random() < 0.12:
                    ang = math.atan2(by - ay, bx - ax) + math.pi / 2
                    o = r.uniform(-0.4, 0.4) * w
                    cx_, cy_ = x + math.cos(ang) * o, y + math.sin(ang) * o
                    L.line(t + 0.05, [(cx_ - 7, cy_ - 3), (cx_, cy_ + 3), (cx_ + 7, cy_ - 1)], (38, 28, 22), 0.85, 2.4)
                if w > 30 and r.random() < 0.02:  # 옹이
                    L.dab(t + 0.1, x + r.normal() * w * 0.2, y, w * 0.18, (40, 30, 26), 0.9)
                out.append((x, y, t, w))
            acc += dd
        return out

    trunk = stroke([(40, 640), (90, 520), (60, 400), (130, 280), (210, 190), (300, 150)], 96, 36, 1.8, 360)
    br = [stroke([(300, 150), (430, 120), (560, 150), (700, 120), (780, 150)], 34, 8, trunk[-1][2], 320),
          stroke([(210, 190), (150, 100), (60, 60)], 24, 8, trunk[-1][2] - 0.2, 300),
          stroke([(130, 280), (260, 300), (380, 260)], 22, 7, trunk[len(trunk) // 2][2], 300),
          stroke([(560, 150), (600, 230), (660, 260)], 14, 5, trunk[-1][2] + 0.4, 280)]
    centers = [(780, 130, 70), (700, 100, 64), (620, 140, 72), (520, 110, 70), (440, 90, 64), (380, 150, 60),
               (660, 250, 58), (590, 210, 52), (360, 260, 60), (280, 300, 52), (60, 50, 66), (130, 90, 56),
               (220, 120, 60), (300, 80, 50), (480, 200, 54), (760, 200, 46), (420, 290, 46)]
    for ci, (cx, cy, rad) in enumerate(centers):
        t0 = 3.6 + ci * 0.16
        n = int(rad * 3.2)
        for k in range(n):
            a = r.uniform(-math.pi * 1.05, math.pi * 0.05) if r.random() < 0.75 else r.uniform(0, math.pi)
            ln = rad * r.uniform(0.55, 1.0)
            ex, ey = cx + math.cos(a) * ln, cy + math.sin(a) * ln * 0.72
            col = [(54, 78, 74), (66, 92, 86), (44, 64, 62), (120, 104, 70)][r.choice(4, p=[0.38, 0.32, 0.25, 0.05])]
            L.line(t0 + (k / n) * 0.35, [(cx + r.normal() * 3, cy + r.normal() * 3), (ex, ey)], col, 0.82, 2.2)
        L.dab(t0 + 0.05, cx, cy, 5, (40, 34, 28), 0.9)
    return L


def build_plum_corner(blossoms):
    L = Layer("plumc")
    L.true_color = True
    r = np.random.default_rng(1225)
    main = plum_stroke(L, [(1960, 1080), (1840, 940), (1800, 820), (1700, 700), (1660, 600)], 30, 8, 4.4, 300, r)
    tw1 = plum_stroke(L, [(1800, 820), (1700, 840), (1600, 800)], 12, 4, main[len(main) // 2][2], 260, r)
    tw2 = plum_stroke(L, [(1840, 940), (1920, 870)], 12, 4, main[len(main) // 4][2], 260, r)
    tw3 = plum_stroke(L, [(1700, 700), (1760, 620), (1780, 540)], 9, 3, main[len(main) * 3 // 4][2], 240, r)
    rr = np.random.default_rng(3)
    for br in (main, tw1, tw2, tw3):
        for q in br[::18]:
            if rr.random() < 0.7:
                kind = rr.choice(["full", "full", "half", "bud"])
                rad = 15 + rr.random() * 12 if kind != "bud" else 9
                blossoms.append(dict(x=q[0] + rr.normal() * 8, y=q[1] + rr.normal() * 8, t=q[2] + 0.4 + rr.random() * 0.4,
                                     spr=make_blossom_sprite(rad, rr, kind)))
    return L


def build_scene1():
    S = {}
    # 해 + 구름
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = 1640, 176
    d = np.hypot(xx - cx, yy - cy)
    k = np.clip((yy - (cy - 95)) / 190, 0, 1)
    S["sun_a"] = (smoothstep(96, 91, d) * (0.85 + 0.12 * fbm(W, H, 30, 3, 41))).astype(np.float32)
    S["sun_k"] = k.astype(np.float32)
    S["sun_d"] = d.astype(np.float32)
    cl = fbm(W, H, 60, 4, 42)
    band = (np.exp(-((yy - 205) / 18) ** 2) * smoothstep(1300, 1420, xx) + np.exp(-((yy - 150) / 12) ** 2) * smoothstep(1050, 1200, xx) * smoothstep(1700, 1600, xx)
            + np.exp(-((yy - 245) / 22) ** 2) * smoothstep(1480, 1580, xx))
    S["cloud"] = (smoothstep(0.45, 0.7, cl) * band * 0.8).astype(np.float32)
    # 산 (원경 → 근경)
    S["mounts"] = [
        mountain_range(330, 80, 51, (176, 182, 182), 0.26, 0.8, 2.4, 1100, W + 100, fog=40),
        mountain_range(620, 200, 52, (156, 162, 160), 0.36, 1.0, 2.8, 250, W + 100, fog=60),
        mountain_range(780, 230, 54, (116, 124, 120), 0.50, 1.6, 3.4, 380, 1300, fog=70),
        mountain_range(1010, 200, 55, (86, 94, 88), 0.62, 2.2, 4.0, 900, W + 100, fog=80),
    ]
    # 운해
    S["fog"] = fbm(W + 900, H, 220, 4, 56).astype(np.float32)
    S["fog_band"] = (np.exp(-((yy - 650) / 70) ** 2) * 0.95 + np.exp(-((yy - 830) / 80) ** 2) * 1.0 + np.exp(-((yy - 1000) / 70) ** 2) * 0.8).astype(np.float32)
    # 벼랑 바위
    pts = [(-20, 430), (40, 410), (120, 440), (170, 520), (150, 600), (210, 660), (260, 760), (320, 820), (380, 900), (420, 1000), (440, H + 20), (-20, H + 20)]
    m = poly_mask(pts, 3)
    rough = fbm(W, H, 14, 3, 57)
    m = smoothstep(0.35, 0.6, m * (0.75 + 0.5 * rough))
    st = streaks(58, 22, 5)
    edge = np.clip(m - np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(7)), np.float32) / 255, 0, 1)
    S["rock_a"] = np.clip(m * (0.35 + 0.55 * st), 0, 1).astype(np.float32)
    S["rock_edge"] = (edge * 0.9).astype(np.float32)
    S["moss"] = (m * smoothstep(0.62, 0.8, fbm(W, H, 50, 3, 59)) * 0.45).astype(np.float32)
    S["rock_T"] = (1.4 + (1 - yy / H) * 1.4 + 0.5 * fbm(W, H, 100, 2, 60)).astype(np.float32)
    # 바위 준법 획
    RK = Layer("rockink"); RK.true_color = True
    rr = np.random.default_rng(63)
    ys_, xs_ = np.where(m > 0.6)
    for _ in range(420):
        j = rr.integers(0, len(xs_))
        x, y = float(xs_[j]), float(ys_[j])
        L_ = rr.uniform(18, 60)
        a_ = math.pi / 2 + rr.normal() * 0.5
        pts_ = [(x, y), (x + math.cos(a_) * L_ * 0.5 + rr.normal() * 4, y + math.sin(a_) * L_ * 0.5), (x + math.cos(a_) * L_, y + math.sin(a_) * L_)]
        RK.line(1.6 + (1 - y / H) * 1.6, pts_, [(60, 50, 42), (80, 68, 56), (40, 34, 30)][rr.integers(0, 3)], 0.6, rr.uniform(2, 4))
    for _ in range(60):
        j = rr.integers(0, len(xs_))
        RK.dab(2.6, float(xs_[j]), float(ys_[j]), rr.uniform(3, 7), (30, 34, 28), 0.85)
    RK.rasterize(64, dry=0.5, bleed_r=6, bleed_a=0.15)
    S["rockink"] = RK
    # 소나무, 홍매화
    S["pine"] = build_pine_cliff()
    S["pine"].rasterize(61, dry=0.35, bleed_r=8, bleed_a=0.18)
    S["blossoms"] = []
    S["plum"] = build_plum_corner(S["blossoms"])
    S["plum"].rasterize(62, dry=0.4, bleed_r=9, bleed_a=0.2)
    return S


def render_scene1(S, paper, t, tl):
    """t: 실제 시간(안개 흐름), tl: 장면 내부 시간(드러남)."""
    img = paper.copy()
    # 해: 먹처럼 번지며 붉게
    p = ease_out_cubic((tl - 0.3) / 1.6)
    m = smoothstep(0, 16, 30 + 80 * p - S["sun_d"]) * S["sun_a"]
    col = np.array([0.93, 0.36, 0.22], np.float32) * (1 - S["sun_k"][..., None]) + np.array([0.86, 0.22, 0.20], np.float32) * S["sun_k"][..., None]
    img *= 1 - m[..., None] * (1 - col)
    for w in S["mounts"]:
        apply_wash(img, w, tl)
    off = int(14 * t) % 900
    fa = smoothstep(0.35, 0.7, S["fog"][:, off:off + W]) * S["fog_band"] * sstep(1.2, 4.0, tl)
    lighten(img, fa * 0.85)
    # 구름 (해 앞을 지나는 옅은 노을 구름)
    cl = np.roll(S["cloud"], int(6 * t), axis=1) * sstep(1.0, 2.5, tl)
    img = img * (1 - cl[..., None] * 0.55) + np.array([0.96, 0.84, 0.78], np.float32) * cl[..., None] * 0.55
    # 바위
    rv = smoothstep(0, 1.0, tl - S["rock_T"])
    img *= 1 - (S["rock_a"] * rv)[..., None] * (1 - np.array([0.47, 0.41, 0.36], np.float32))
    img *= 1 - (S["moss"] * rv)[..., None] * (1 - np.array([0.55, 0.62, 0.42], np.float32))
    img *= 1 - (S["rock_edge"] * rv)[..., None] * (1 - np.array([0.16, 0.14, 0.12], np.float32))
    S["rockink"].apply(img, tl)
    S["pine"].apply(img, tl)
    S["plum"].apply(img, tl)
    return img


def scene1_sprites(S, canvas, t, tl):
    for b in S["blossoms"]:
        k = (tl - b["t"]) / 0.6
        if k <= 0:
            continue
        s = ease_out_back(k, 2.0) if k < 1 else 1.0
        spr = b["spr"]
        im = b.get("_pil")
        if im is None:
            im = Image.fromarray((spr * 255).astype(np.uint8), "RGBA")
            b["_pil"] = im
        if s != 1.0:
            im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))), Image.BILINEAR)
        canvas.alpha_composite(im, (int(b["x"] - im.size[0] / 2), int(b["y"] - im.size[1] / 2)))


# ---------------------------------------------------------------- 장면 2 · 눈 덮인 한옥 대청

def build_scene2():
    B = Layer("hall_base"); B.true_color = True     # 면 (바탕)
    D = Layer("hall_detail"); D.true_color = True   # 선 (세부)
    t0 = 13.0
    cx = 960
    ridge = 168
    xl, xr = 170, 1750

    def eave(x):
        u = abs(x - cx) / ((xr - xl) / 2)
        return 340 - 62 * u ** 3

    def roof_top(x):
        return ridge + max(0, 520 - x) * 0.42 + max(0, x - 1400) * 0.42

    for i in range(60):
        u0, u1 = i / 60, (i + 1) / 60
        xa, xb = xl + (xr - xl) * u0, xl + (xr - xl) * u1
        B.poly(t0 + u0 * 0.9, [(xa, roof_top(xa)), (xb, roof_top(xb)), (xb, eave(xb)), (xa, eave(xa))], (84, 86, 90), 0.82)
    for i in range(80):
        u = (i + 0.5) / 80
        x = xl + 16 + (xr - xl - 32) * u
        D.line(t0 + 0.2 + u * 0.9, [(x, roof_top(x) + 4), (x, eave(x) - 4)], (36, 38, 42), 0.75, 4)
    D.line(t0 + 0.1, [(490, ridge - 30), (520, ridge - 8)] + [(x, ridge - 9) for x in range(540, 1390, 20)] + [(1400, ridge - 8), (1430, ridge - 30)], (30, 30, 34), 0.95, 16)
    D.line(t0 + 0.7, [(x, eave(x)) for x in range(xl, xr + 1, 10)], (28, 28, 32), 0.95, 10)
    for x in range(xl + 10, xr, 20):
        D.dab(t0 + 0.8, x, eave(x) + 6, 6, (40, 40, 44), 0.9)
    for x in range(xl + 50, xr - 50, 13):
        D.line(t0 + 0.9, [(x, eave(x) + 10), (x, eave(x) + 34)], (110, 78, 54), 0.6, 5)
    pillars = [330, 640, 1280, 1590]
    for i, px in enumerate(pillars):
        B.poly(t0 + 1.0 + i * 0.05, [(px - 17, eave(px) + 30), (px + 17, eave(px) + 30), (px + 17, 806), (px - 17, 806)], (110, 72, 48), 0.9)
    D.line(t0 + 1.1, [(310, 386), (1610, 386)], (100, 66, 44), 0.85, 12)
    # 대청 안: 병풍 + 마루
    B.poly(t0 + 1.2, [(658, 402), (1262, 402), (1262, 716), (658, 716)], (236, 222, 192), 0.5)
    B.poly(t0 + 1.25, [(657, 716), (1263, 716), (1280, 806), (640, 806)], (156, 108, 68), 0.8)
    for y in range(724, 806, 13):
        D.line(t0 + 1.35, [(650, y), (1270, y)], (112, 74, 48), 0.55, 2)
    r = np.random.default_rng(72)
    for k in range(6):
        x0 = 670 + k * 98
        D.line(t0 + 1.3, [(x0, 412), (x0 + 94, 412), (x0 + 94, 706), (x0, 706), (x0, 412)], (120, 86, 58), 0.8, 4)
        # 폭마다 작은 산수 (산 + 소나무)
        for j in range(3):
            bx = x0 + 18 + j * 28
            hh = r.uniform(40, 90)
            D.poly(t0 + 1.5 + k * 0.05, [(bx - 26, 640), (bx, 640 - hh), (bx + 26, 640)], (140, 146, 144), 0.55)
        if k % 2 == 0:
            px = x0 + 60
            D.line(t0 + 1.6, [(px, 690), (px + 4, 600), (px - 8, 540)], (96, 74, 56), 0.85, 4)
            for j in range(3):
                D.dab(t0 + 1.7, px - 14 + j * 14, 540 + j * 18, 14, (72, 96, 86), 0.55)
        D.line(t0 + 1.6, [(x0 + 10, 670), (x0 + 84, 668)], (150, 156, 154), 0.4, 3)
    # 양 옆 창호
    for (x0, x1) in ((348, 622), (1298, 1572)):
        B.poly(t0 + 1.2, [(x0, 410), (x1, 410), (x1, 790), (x0, 790)], (232, 222, 202), 0.35)
        for xx in np.arange(x0 + 22, x1, 26):
            D.line(t0 + 1.3, [(xx, 410), (xx, 790)], (120, 86, 60), 0.65, 2)
        for yy in np.arange(426, 790, 26):
            D.line(t0 + 1.35, [(x0, yy), (x1, yy)], (120, 86, 60), 0.65, 2)
        D.line(t0 + 1.4, [(x0, 410), (x1, 410), (x1, 790), (x0, 790), (x0, 410)], (96, 66, 46), 0.85, 5)
    # 기단 + 계단
    B.poly(t0 + 1.5, [(250, 806), (1670, 806), (1710, 866), (210, 866)], (156, 150, 142), 0.6)
    for x in range(260, 1680, 90):
        D.line(t0 + 1.6, [(x, 808), (x - 4, 864)], (96, 92, 86), 0.55, 2)
    for k in range(3):
        B.poly(t0 + 1.7, [(840 - k * 16, 866 + k * 14), (1080 + k * 16, 866 + k * 14), (1080 + k * 16, 880 + k * 14), (840 - k * 16, 880 + k * 14)], (140, 134, 126), 0.65)
    B.rasterize(71, dry=0.3, bleed_r=8, bleed_a=0.2)
    D.rasterize(76, dry=0.3, bleed_r=5, bleed_a=0.12)
    # 지붕·기단 위 눈: 불규칙한 덩어리
    snow = Image.new("L", (W, H), 0)
    ds = ImageDraw.Draw(snow)
    for _ in range(520):
        x = r.uniform(xl + 20, xr - 20)
        top = roof_top(x)
        y = top + (eave(x) - top) * (r.random() ** 2.2) * 0.85
        rw, rh = r.uniform(6, 20), r.uniform(2.5, 5)
        ds.ellipse([x - rw, y - rh, x + rw, y + rh], fill=int(r.uniform(170, 245)))
    for x in range(xl, xr, 6):
        ds.ellipse([x - 9, eave(x) - 12 - r.uniform(0, 5), x + 9, eave(x) - 1], fill=245)
    ds.line([(520, ridge - 17), (1400, ridge - 17)], fill=250, width=8)
    for x in range(220, 1700, 8):
        ds.ellipse([x - 11, 798 - r.uniform(0, 6), x + 11, 812], fill=240)
    snow = snow.filter(ImageFilter.GaussianBlur(2.0))
    sm = np.asarray(snow, np.float32) / 255 * (0.7 + 0.35 * fbm(W, H, 8, 2, 73))
    S = dict(base=B, detail=D, snow=np.clip(sm, 0, 1).astype(np.float32))
    S["mounts"] = [mountain_range(600, 220, 74, (150, 156, 154), 0.35, 12.6, 14.0, -100, 300, fog=80),
                   mountain_range(600, 200, 75, (150, 156, 154), 0.35, 12.6, 14.0, 1620, W + 100, fog=80)]
    yy = np.arange(H, dtype=np.float32)[:, None]
    S["ground"] = (smoothstep(880, 1000, yy) * 0.10 * np.ones((1, W), np.float32)).astype(np.float32)
    S["flakes"] = [(r.random() * W, r.random() * H, 20 + r.random() * 30, r.random() * 6) for _ in range(120)]
    return S


def render_scene2(S, paper, t, tl):
    img = paper.copy()
    for w in S["mounts"]:
        apply_wash(img, w, tl)
    S["base"].apply(img, tl)
    S["detail"].apply(img, tl)
    sn = S["snow"] * sstep(14.4, 15.6, tl)
    lighten(img, sn * 0.95, (0.99, 0.99, 1.0))
    img *= 1 - (S["ground"])[..., None] * (1 - np.array([0.7, 0.66, 0.6], np.float32))
    return img


def sit_bow(t, t0, lag=0.0):
    x = t - t0 - lag
    if x < 0 or x > 4.4:
        return 0.0, 0.0
    h = ease_io(x / 0.6) * (1 - ease_io((x - 3.8) / 0.6))
    b = ease_io((x - 0.7) / 0.9) * (1 - ease_io((x - 2.8) / 0.9))
    return h, b


def scene2_people(canvas, t, tl):
    a = sstep(14.6, 15.4, tl)
    if a <= 0.01:
        return
    S = 2
    sc = 0.85
    x0, y0, w, h = 640, 420, 640, 400
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for kind, cx, lag in (("man", 845, 0.0), ("woman", 1080, 0.2)):
        hh, bb = sit_bow(tl, 16.2, lag)
        draw_person(d, kind, (cx - x0) * S, (796 - y0) * S, S * sc, hh, 1.0, bb, 0.8 * math.sin(t * 1.6 + lag * 5))
    im = im.reduce(S)
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (x0, y0))


# ---------------------------------------------------------------- 장면 3 · 수묵 대나무 (잎이 하나씩)

def build_scene3():
    L = Layer("bamboo")
    L.true_color = True
    r = np.random.default_rng(88)
    t0 = 29.0
    stalks = [(250, 30, 0.04, (120, 124, 122), 0.0), (380, 20, -0.03, (150, 154, 152), 0.3),
              (1640, 34, -0.05, (110, 114, 112), 0.15), (1780, 22, 0.03, (150, 154, 152), 0.45)]
    leaves = []
    for (bx, bw, tilt, col, dt) in stalks:
        y = H + 30
        tt = t0 + dt
        nodes = []
        while y > -40:
            sl = r.uniform(150, 200)
            steps = int(sl / 2)
            for i in range(steps):
                s = i / steps
                yy = y - s * sl
                xx = bx + tilt * (H - yy)
                L.dab(tt + s * 0.14, xx, yy, bw / 2, col, 0.55 + 0.2 * abs(s - 0.5))
                if i % 2 == 0:
                    L.dab(tt + s * 0.14, xx + bw * 0.32, yy, bw * 0.14, darker_c(col, 0.7), 0.5)
                    L.dab(tt + s * 0.14, xx - bw * 0.3, yy, bw * 0.1, (250, 248, 240), 0.2)
            tt += 0.16
            y -= sl
            xx = bx + tilt * (H - y)
            L.line(tt, [(xx - bw * 0.6, y + 3), (xx, y - 2), (xx + bw * 0.6, y + 2)], (50, 52, 52), 0.9, 4)
            nodes.append((xx, y, tt))
            y -= 4
        for (nx, ny, nt) in nodes:
            if ny > 900 or ny < 30 or r.random() < 0.12:
                continue
            side = 1 if bx < W / 2 else -1
            ex, ey = nx + side * r.uniform(120, 200), ny + r.uniform(-30, 10)
            L.line(nt + 0.1, [(nx, ny), ((nx + ex) / 2, (ny + ey) / 2 - 12), (ex, ey)], (60, 62, 62), 0.85, 3)
            for k in range(r.integers(3, 6)):
                la = math.pi / 2 + side * (-0.4 + k * 0.32) + r.normal() * 0.12
                leaves.append((ex, ey, la, r.uniform(110, 170), r.uniform(18, 26), r.uniform(0.45, 0.9)))
    # 잎은 화면 전체에서 하나씩 차례로
    r.shuffle(leaves)
    for i, (x, y, a, ln, wd, al) in enumerate(leaves):
        tl0 = 29.8 + i * min(0.16, 6.5 / max(1, len(leaves)))
        for j, seg in enumerate(leaf_poly(x, y, a, ln, wd, nseg=6)):
            L.poly(tl0 + j * 0.03, seg, (52, 56, 56), al)
    L.rasterize(81, dry=0.5, bleed_r=6, bleed_a=0.15)
    return dict(layer=L, nleaves=len(leaves))


def darker_c(c, k):
    return tuple(int(v * k) for v in c)


def render_scene3(S, paper, t, tl):
    img = paper.copy()
    S["layer"].apply(img, tl)
    return img


# ---------------------------------------------------------------- 장면 4 · 설날 상차림

def build_scene4():
    L = Layer("room")
    L.true_color = True
    D = Layer("room_detail")
    D.true_color = True
    t0 = 39.0
    # 병풍 (8폭) — 수묵 산수
    for k in range(8):
        x0 = 120 + k * 210
        L.poly(t0 + k * 0.05, [(x0, 110), (x0 + 206, 110), (x0 + 206, 640), (x0, 640)], (232, 220, 196), 0.5)
        D.line(t0 + k * 0.05, [(x0, 110), (x0 + 206, 110), (x0 + 206, 640), (x0, 640), (x0, 110)], (120, 84, 56), 0.85, 6)
    r = np.random.default_rng(91)
    for k in range(18):
        bx = 150 + k * 95
        h = r.uniform(60, 160)
        D.poly(t0 + 0.4 + k * 0.03, [(bx - 70, 560), (bx, 560 - h), (bx + 70, 560)], (140, 148, 146), 0.5)
    for k in range(4):
        bx = 250 + k * 450
        D.line(t0 + 0.6, [(bx, 620), (bx + 10, 420), (bx - 20, 300)], (90, 70, 54), 0.85, 7)
        for j in range(4):
            D.dab(t0 + 0.7, bx - 30 + j * 24, 300 + j * 26, 26, (70, 94, 86), 0.6)
    D.dab(t0 + 0.5, 1500, 220, 46, (200, 70, 60), 0.75)
    # 바닥 (장판)
    L.poly(t0 + 0.2, [(-60, 640), (W + 60, 640), (W + 60, H + 60), (-60, H + 60)], (214, 180, 120), 0.45)
    for y in range(660, H, 60):
        D.line(t0 + 0.3, [(-60, y), (W + 60, y)], (180, 140, 90), 0.4, 2)
    L.rasterize(92, dry=0.3, bleed_r=8, bleed_a=0.15)
    D.rasterize(93, dry=0.35, bleed_r=6, bleed_a=0.15)
    return dict(layer=L, detail=D)


def render_scene4(S, paper, t, tl):
    img = paper.copy()
    S["layer"].apply(img, tl)
    S["detail"].apply(img, tl)
    return img


def draw_table(canvas, t, tl):
    a = sstep(40.2, 41.0, tl)
    if a <= 0.01:
        return
    S = 2
    x0, y0, w, h = 0, 600, W, 480
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    Z = 1.6  # 상차림 확대 (화면 중심 기준)

    def P(x, y):
        X = 960 + (x - 960) * Z
        Y = 830 + (y - 830) * Z
        return ((X - x0) * S, (Y - y0) * S)

    def E(cx, cy, rx, ry, fill, outline=None, width=2):
        d.ellipse([P(cx - rx, cy - ry), P(cx + rx, cy + ry)], fill=fill, outline=outline, width=int(width * S * Z) if outline else 0)

    # 교자상
    d.polygon([P(500, 806), P(1420, 806), P(1470, 900), P(450, 900)], fill=(140, 50, 34, 255), outline=(80, 26, 16, 255))
    d.polygon([P(520, 812), P(1400, 812), P(1420, 830), P(500, 830)], fill=(170, 74, 52, 120))
    d.rectangle([P(450, 900), P(1470, 920)], fill=(104, 34, 22, 255))
    for lx in (480, 1440):
        d.polygon([P(lx - 12, 920), P(lx + 12, 920), P(lx + 8, 990), P(lx - 8, 990)], fill=(96, 30, 20, 255))
    # 음식 등장 순서
    def pop(t_):
        k = (tl - t_) / 0.45
        return 0 if k <= 0 else (ease_out_back(k, 2.2) if k < 1 else 1.0)

    # 떡국 두 그릇 (놋그릇)
    for i, bx in enumerate((760, 1160)):
        s = pop(41.0 + i * 0.25)
        if s <= 0:
            continue
        E(bx, 842, 70 * s, 22 * s, (196, 158, 76, 255), (120, 90, 40, 255))
        E(bx, 838, 58 * s, 16 * s, (246, 242, 230, 255))
        rr = np.random.default_rng(i)
        for j in range(9):
            ox, oy = rr.uniform(-44, 44) * s, rr.uniform(-10, 8) * s
            E(bx + ox, 838 + oy, 9 * s, 4 * s, (255, 255, 252, 255), (200, 196, 186, 255), 1)
        for j in range(6):
            ox = rr.uniform(-30, 30) * s
            d.line([P(bx + ox, 832), P(bx + ox + 12 * s, 834)], fill=(242, 196, 50, 255), width=3 * S)
            d.line([P(bx + ox + 8, 840), P(bx + ox + 18 * s, 842)], fill=(250, 250, 240, 255), width=3 * S)
        for j in range(5):
            E(bx + rr.uniform(-24, 24) * s, 836 + rr.uniform(-6, 6) * s, 3 * s, 2 * s, (60, 130, 60, 255))
        d.line([P(bx - 30, 830), P(bx + 26, 846)], fill=(40, 40, 40, 255), width=2 * S)
    # 전
    s = pop(41.6)
    if s > 0:
        E(960, 872, 92 * s, 22 * s, (240, 236, 226, 255), (170, 160, 150, 255))
        for j, (ox, oy) in enumerate(((-46, -2), (0, -6), (46, -2), (-22, 8), (24, 8))):
            E(960 + ox * s, 868 + oy * s, 26 * s, 10 * s, (222, 172, 70, 255), (160, 110, 40, 255), 1)
            E(960 + ox * s - 6, 866 + oy * s, 6 * s, 3 * s, (90, 150, 60, 255))
            E(960 + ox * s + 7, 868 + oy * s, 4 * s, 2 * s, (200, 60, 50, 255))
    # 과일 (사과·배·감)
    s = pop(41.9)
    if s > 0:
        E(560, 856, 70 * s, 18 * s, (230, 226, 216, 255), (150, 140, 130, 255))
        for (ox, oy, c, rr_) in ((-30, -8, (200, 40, 40), 18), (0, -10, (226, 200, 120), 20), (30, -8, (236, 130, 40), 17),
                                 (-14, -30, (190, 30, 40), 16), (16, -30, (238, 140, 46), 16), (0, -48, (220, 196, 110), 15)):
            E(560 + ox * s, 850 + oy * s, rr_ * s, rr_ * s, c + (255,), darker_c(c, 0.7) + (255,), 1)
            E(560 + ox * s - rr_ * 0.35 * s, 850 + oy * s - rr_ * 0.4 * s, rr_ * 0.25 * s, rr_ * 0.18 * s, (255, 250, 240, 150))
    # 약과
    s = pop(42.2)
    if s > 0:
        E(1360, 858, 66 * s, 18 * s, (230, 226, 216, 255), (150, 140, 130, 255))
        for j in range(7):
            ang = j * math.tau / 7
            cx, cy = 1360 + math.cos(ang) * 28 * s, 848 + math.sin(ang) * 8 * s
            E(cx, cy, 13 * s, 7 * s, (160, 92, 30, 255), (110, 60, 20, 255), 1)
        E(1360, 836, 14 * s, 8 * s, (168, 100, 36, 255), (110, 60, 20, 255), 1)
    # 식혜
    s = pop(42.5)
    if s > 0:
        E(1250, 878, 34 * s, 12 * s, (236, 232, 220, 255), (150, 140, 130, 255))
        E(1250, 875, 26 * s, 8 * s, (240, 228, 196, 255))
        for j in range(4):
            E(1242 + j * 5, 875, 2, 1.5, (250, 250, 240, 255))
        E(1254, 872, 3, 3, (200, 60, 50, 255))
    im = im.reduce(S)
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (x0, y0))
    # 떡국 김
    if tl > 41.5:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(ov)
        for bx in (760, 1160):
            for j in range(3):
                ph = t * 1.2 + j * 2.1 + bx
                X = 960 + (bx - 960) * 1.6
                pts = [(X - 24 + j * 24 + 8 * math.sin(ph + k * 0.6), 800 - k * 12) for k in range(10)]
                dd.line(pts[:6], fill=(255, 255, 255, 110), width=5)
        ov = ov.filter(ImageFilter.GaussianBlur(3))
        canvas.alpha_composite(ov)


def scene4_people(canvas, t, tl):
    a = sstep(39.8, 40.6, tl)
    if a <= 0.01:
        return
    S = 2
    sc = 1.0
    x0, y0, w, h = 560, 360, 800, 520
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for kind, cx, ph in (("man", 800, 0.0), ("woman", 1120, 1.2)):
        laugh = 2.5 * math.sin(t * 6 + ph) * max(0, math.sin(t * 0.9 + ph))
        nod = 0.10 * max(0, math.sin(t * 1.4 + ph)) ** 4
        draw_person(d, kind, (cx - x0) * S, (860 - y0) * S, S * sc, 0.0, 1.0, nod, laugh)
    im = im.reduce(S)
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (x0, y0))


# ---------------------------------------------------------------- 전체 장면

class Scene:
    def __init__(self, fonts_dir):
        print("장면 생성 중...", flush=True)
        self.paper = make_paper()
        self.s1 = build_scene1()
        self.s2 = build_scene2()
        self.s3 = build_scene3()
        self.s4 = build_scene4()
        self.fonts = Fonts(fonts_dir)
        self.seal = make_seal(self.fonts, 84)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dd = np.hypot(xx - W / 2, yy - H / 2) / 1100
        f = 0.55 * fbm(W, H, 220, 4, 4242) + 0.45 * dd
        self.tfield = ((f - f.min()) / (f.max() - f.min())).astype(np.float32)
        self.texts = self.build_texts()
        r = np.random.default_rng(5)
        self.flecks = [dict(x=r.random() * W, y=r.random() * H, vy=-(6 + r.random() * 12), ph=r.random() * 6) for _ in range(50)]
        self.fspr = RS.Scene.__new__(RS.Scene)
        g = np.mgrid[0:9, 0:9].astype(np.float32) - 4
        rr = np.hypot(g[0], g[1]) / 4.5
        arr = np.zeros((9, 9, 4), np.uint8)
        arr[..., 0], arr[..., 1], arr[..., 2] = 210, 160, 64
        arr[..., 3] = (np.clip(np.exp(-rr ** 2 * 4) + 0.8 * np.exp(-rr ** 2 * 30), 0, 1) * 255).astype(np.uint8)
        self.fleck = Image.fromarray(arr, "RGBA")
        print("준비 완료", flush=True)

    def build_texts(self):
        F = self.fonts
        T = []
        cx = W / 2
        # 1 산수
        T.append(TextItem(render_line(F, [("2027 정미년", RED)], "body", 38, 10), cx, 300, 6.0, 11.0, rise=10))
        T.append(TextItem(render_line(F, [("희망찬 새해가 밝았습니다", DARK)], "title", 72, 4), cx, 390, 6.6, 11.1))
        # 2 한옥
        T.append(TextItem(render_line(F, [("마음을 담아 새해 인사를 올립니다", DARK)], "body", 56, 4), cx, 70, 14.2, 27.0))
        T.append(TextItem(render_line(F, [("새해 ", DARK), ("복", RED), (" 많이 받으세요", DARK)], "brush", 130, 6), cx, 968, 17.6, 27.1,
                          d_in=1.2, stagger=0.11, rise=20))
        # 3 대나무
        T.append(TextItem(render_line(F, [("올 한 해도 대나무처럼", DARK)], "body", 60, 3), cx, 450, 30.2, 37.0))
        T.append(TextItem(render_line(F, [("곧고 ", DARK), ("푸르게", TEAL), (", 건강하고 ", DARK), ("튼튼하게", RED)], "body", 60, 3), cx, 550, 31.0, 37.1))
        # 4 상차림
        T.append(TextItem(render_line(F, [("온 가족이 둘러앉아 ", DARK), ("웃음꽃", RED), (" 피는 설날", DARK)], "body", 52, 3), cx, 60, 40.6, 49.0))
        T.append(TextItem(render_line(F, [("신길튼튼365정형외과", TEAL), ("가 여러분의 건강을 응원합니다", DARK)], "body", 40, 2), cx, 1030, 43.0, 49.1))
        # 5 엔딩
        T.append(TextItem(render_line(F, [("새해 ", DARK), ("복", RED), (" 많이 받으세요", DARK)], "brush", 150, 6), cx, 330, 51.6, 99,
                          d_in=1.2, stagger=0.11, rise=22))
        T.append(TextItem(render_line(F, [("신길튼튼365정형외과 임직원 일동", DARK)], "title", 50, 3), cx, 470, 53.0, 99))
        T.append(TextItem(render_line(F, [("서울특별시 영등포구 가마산로 483  3A01호~3A05호", (90, 80, 72))], "reg", 28, 1), cx, 530, 53.8, 99, rise=8))
        return T

    def bg(self, which, t):
        if which == 1:
            return render_scene1(self.s1, self.paper, t, t)
        if which == 2:
            return render_scene2(self.s2, self.paper, t, t)
        if which == 3:
            return render_scene3(self.s3, self.paper, t, t)
        if which == 4:
            return render_scene4(self.s4, self.paper, t, t)
        return render_scene1(self.s1, self.paper, t, 30.0)

    def overlays(self, which, canvas, t):
        d = ImageDraw.Draw(canvas)
        if which == 1:
            scene1_sprites(self.s1, canvas, t, t)
            magpie_pair(d, t, 5.0, 12.5, 250, 1.7)
        elif which == 2:
            scene2_people(canvas, t, t)
            for (x, y, v, ph) in self.s2["flakes"]:
                yy = (y + v * t) % H
                xx = (x + 14 * math.sin(t * 0.8 + ph)) % W
                d.ellipse([xx - 2, yy - 2, xx + 2, yy + 2], fill=(255, 255, 255, 200))
        elif which == 4:
            scene4_people(canvas, t, t)
            draw_table(canvas, t, t)
        elif which == 5:
            scene1_sprites(self.s1, canvas, t, 30.0)
            magpie_pair(d, t, 50.5, 58.5, 200, 1.7)

    def compose(self, which, t):
        img = self.bg(which, t)
        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        self.overlays(which, canvas, t)
        # 장면마다 아주 느린 카메라 움직임
        z = 1.0 + 0.035 * ((t * 0.07) % 1.0 if False else sstep(0, 1, (t % 12) / 12))
        starts = {1: 0, 2: X12[0], 3: X23[0], 4: X34[0], 5: X45[0]}
        u = sstep(0, 1, (t - starts[which]) / 14)
        z = 1.0 + 0.035 * u
        cw, ch = W / z, H / z
        ox, oy = (W - cw) * (0.5 if which != 1 else 0.3), (H - ch) * 0.5
        canvas = canvas.resize((W, H), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))
        return canvas

    def frame(self, t):
        bounds = [(X12, 1, 2), (X23, 2, 3), (X34, 3, 4), (X45, 4, 5)]
        which = 1 if t < X12[0] else 2 if t < X23[0] else 3 if t < X34[0] else 4 if t < X45[0] else 5
        canvas = None
        for (rng, a, b) in bounds:
            if rng[0] <= t < rng[1]:
                A = np.asarray(self.compose(a, t).convert("RGB"), np.float32) / 255
                B = np.asarray(self.compose(b, t).convert("RGB"), np.float32) / 255
                out = ink_dissolve(A, B, sstep(rng[0], rng[1], t), self.tfield)
                canvas = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
                break
        if canvas is None:
            canvas = self.compose(which, t)
        # 금빛 티끌
        for f in self.flecks:
            x = (f["x"] + 12 * math.sin(t * 0.5 + f["ph"])) % W
            y = (f["y"] + f["vy"] * t) % H
            a = (0.25 + 0.6 * math.sin(t * 1.2 + f["ph"]) ** 2) * sstep(2, 4, t)
            arr = np.asarray(self.fleck).copy()
            arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
            canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (int(x), int(y)))
        for tx in self.texts:
            tx.draw(canvas, t)
        if t > 54.6:
            k = (t - 54.6) / 0.45
            s = 1.6 - 0.6 * ease_out_cubic(k) if k < 1 else 1.0
            sz = max(4, int(84 * s))
            im = self.seal.resize((sz, sz), Image.BILINEAR).rotate(-4, Image.BICUBIC, expand=True)
            arr = np.asarray(im).copy()
            arr[..., 3] = (arr[..., 3] * min(1, k * 1.8) * 0.92).astype(np.uint8)
            canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (int(1530 - im.size[0] / 2), int(330 - im.size[1] / 2)))
        out = canvas.convert("RGB")
        f_in = sstep(0, 0.8, t)
        f_out = 1 - sstep(DUR - 1.4, DUR - 0.05, t)
        bgc = Image.new("RGB", (W, H), (250, 247, 240))
        if f_in < 1:
            out = Image.blend(bgc, out, f_in)
        if f_out < 1:
            out = Image.blend(bgc, out, f_out)
        return out


SCENE = None


def render_chunk(args):
    idx, f0, f1, outdir = args
    path = os.path.join(outdir, f"ny_seg_{idx:02d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
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
            SCENE.frame(float(s)).save(os.path.join(work, f"ny_{float(s):05.1f}.png"))
        return
    wav = os.path.join(work, "ny_music.wav")
    RS.BOW1 = 15.0   # 세배 종소리: 17.0초
    RS.BOW2 = 39.0
    RS.make_music(wav)
    print("음악 생성 완료", flush=True)
    nj = max(1, a.jobs)
    edges = [round(i * NFRAMES / nj) for i in range(nj + 1)]
    with Pool(nj) as pool:
        segs = pool.map(render_chunk, [(i, edges[i], edges[i + 1], work) for i in range(nj)])
    lst = os.path.join(work, "ny_segs.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("완료:", a.out)


if __name__ == "__main__":
    main()
