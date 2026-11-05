#!/usr/bin/env python3
"""신길튼튼365정형외과 · 새해 복 많이 받으세요 (설날 세배 수묵 모션그래픽)

1920x1080 · 30fps · 60초 · 국악풍 배경음악(직접 합성)

한지 위에 수묵으로 붉은 해·먼 산·소나무·양반집(한옥)이 그려지고,
고운 한복을 입은 남녀가 세배(큰절)를 올리며 새해 인사를 전한다.

사용:  python3 render_seollal.py --fonts <폰트폴더> --out <출력.mp4> [--preview 초,...]
필요 폰트: GowunBatang-Bold/Regular.ttf, Myeongjo-EB.ttf, Brush.ttf (render.py 와 동일)
"""
import argparse
import math
import os
import subprocess
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render as R
from render import (W, H, FPS, smoothstep, sstep, ease_out_cubic, ease_out_back, fbm, value_noise,
                    Layer, make_paper, render_line, TextItem, Fonts, make_seal, pluck, pad, bell, midi_hz, SR)

DUR = 60.0
NFRAMES = int(FPS * DUR)

INK = (36, 32, 30)
DARK = (40, 34, 32)
RED = (186, 32, 44)
GOLD = (176, 128, 40)


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def in_text_zone(x, y):
    return 440 < x < 1480 and 70 < y < 360


# ---------------------------------------------------------------- 수묵 배경 요소

def build_mountains_far():
    """한옥 뒤로 겹겹이 옅게 번지는 먼 산 (아래로 갈수록 안개처럼 사라짐)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for k, (base, amp, alpha, col, seed) in enumerate([(470, 150, 0.20, (120, 140, 140), 5), (540, 110, 0.28, (96, 116, 118), 6)]):
        r = np.random.default_rng(seed)
        x = np.arange(W, dtype=np.float32)
        h = np.zeros(W, np.float32)
        for f, a in [(1 / 650, 1.0), (1 / 240, 0.45), (1 / 80, 0.12), (1 / 26, 0.04)]:
            h += a * np.sin(x * f * math.tau + r.random() * 10)
        ridge = base - amp * (0.6 + 0.4 * h / 1.6)
        d = yy - ridge[None, :]
        a = smoothstep(-2, 5, d) * np.exp(-np.clip(d, 0, None) / 90) * alpha
        a *= 0.7 + 0.45 * fbm(W, H, 70, 3, seed + 3)
        T = 1.0 + 3.0 * (xx / W) + 0.8 * fbm(W, H, 150, 2, seed + 7) + k * 0.6
        out.append((a.astype(np.float32), np.array(col, np.float32) / 255, T.astype(np.float32)))
    return out


def build_ground():
    L = Layer("ground")
    for i in range(24):
        x0 = -60 + i * 85
        L.poly(3.6 + i * 0.04, [(x0, 832), (x0 + 90, 832), (x0 + 90, H + 60), (x0, H + 60)], (196, 186, 166), 0.30)
    return L


def build_hanok():
    """팔작지붕 한옥 정면 + 기단 + 담장 (수묵 담채)."""
    L = Layer("hanok")
    r = np.random.default_rng(483)
    cx = 1100
    ridge_y = 382
    x_l, x_r = 540, 1660

    def eave_y(x):
        u = abs(x - cx) / ((x_r - x_l) / 2)
        return 532 - 74 * u ** 3

    t0 = 2.0
    # 지붕면
    for i in range(48):
        u0, u1 = i / 48, (i + 1) / 48
        xa, xb = x_l + (x_r - x_l) * u0, x_l + (x_r - x_l) * u1
        ta = 760 + (1440 - 760) * u0
        tb = 760 + (1440 - 760) * u1
        ya = ridge_y + (0 if 760 <= xa <= 1440 else 0)
        top_a = ridge_y + max(0, (760 - xa)) * (80 / 220) + max(0, xa - 1440) * (80 / 220)
        top_b = ridge_y + max(0, (760 - xb)) * (80 / 220) + max(0, xb - 1440) * (80 / 220)
        L.poly(t0 + u0 * 1.4, [(xa, top_a), (xb, top_b), (xb, eave_y(xb)), (xa, eave_y(xa))], (66, 72, 80), 0.78)
    # 기왓골
    for i in range(62):
        u = (i + 0.5) / 62
        x = x_l + 14 + (x_r - x_l - 28) * u
        top = ridge_y + max(0, (760 - x)) * (80 / 220) + max(0, x - 1440) * (80 / 220) + 6
        L.line(t0 + 0.2 + u * 1.4, [(x + (x - cx) * 0.02, top), (x, eave_y(x) - 6)], (34, 38, 44), 0.6, 2.2)
    # 용마루 (양끝 치켜올림)
    pts = [(730, ridge_y - 26), (760, ridge_y - 6)] + [(x, ridge_y - 8) for x in range(780, 1430, 20)] + [(1440, ridge_y - 6), (1470, ridge_y - 26)]
    L.line(t0 + 0.1, pts, (26, 26, 30), 0.95, 13)
    # 처마선 + 막새
    ep = [(x, eave_y(x)) for x in range(x_l, x_r + 1, 10)]
    L.line(t0 + 0.8, ep, (24, 24, 28), 0.95, 9)
    for x in range(x_l + 8, x_r, 18):
        L.dab(t0 + 0.9 + (x - x_l) / 1400, x, eave_y(x) + 5, 5, (30, 30, 34), 0.9)
    # 서까래 그늘
    for x in range(x_l + 40, x_r - 40, 12):
        L.line(t0 + 1.0, [(x, eave_y(x) + 8), (x + (x - cx) * 0.01, eave_y(x) + 30)], (90, 66, 48), 0.55, 4)
    # 기둥과 창방
    t1 = 3.4
    pillars = [640, 820, 1000, 1200, 1380, 1560]
    for i, px in enumerate(pillars):
        y0 = eave_y(px) + 24
        L.poly(t1 + i * 0.08, [(px - 10, y0), (px + 10, y0), (px + 10, 792), (px - 10, 792)], (124, 80, 52), 0.82)
    for yb in (566, 600):
        L.line(t1 + 0.3, [(620, yb), (1580, yb)], (112, 74, 50), 0.8, 8)
    # 단청 띠 (은은한 청록·주홍)
    for i in range(len(pillars) - 1):
        x0, x1 = pillars[i] + 10, pillars[i + 1] - 10
        L.poly(t1 + 0.5 + i * 0.05, [(x0, 572), (x1, 572), (x1, 594), (x0, 594)], (64, 130, 118), 0.5)
        for xx in np.arange(x0 + 14, x1 - 10, 34):
            L.dab(t1 + 0.6 + i * 0.05, xx, 583, 6, (196, 82, 60), 0.6)
    # 창호 (문살)
    t2 = 4.2
    for i in range(len(pillars) - 1):
        x0, x1 = pillars[i] + 12, pillars[i + 1] - 12
        if i == 2:  # 가운데 대청: 열린 공간
            L.poly(t2, [(x0, 606), (x1, 606), (x1, 790), (x0, 790)], (70, 56, 46), 0.55)
            for yy in range(720, 790, 14):
                L.line(t2 + 0.2, [(x0, yy), (x1, yy)], (120, 90, 64), 0.5, 2)
            continue
        L.poly(t2 + i * 0.1, [(x0, 612), (x1, 612), (x1, 786), (x0, 786)], (226, 214, 190), 0.25)
        n = 6
        for k in range(1, n):
            xx = x0 + (x1 - x0) * k / n
            L.line(t2 + 0.1 + i * 0.1, [(xx, 612), (xx, 786)], (120, 86, 60), 0.6, 2)
        for yy in np.arange(626, 786, 22):
            L.line(t2 + 0.15 + i * 0.1, [(x0, yy), (x1, yy)], (120, 86, 60), 0.6, 2)
        L.line(t2 + 0.2 + i * 0.1, [(x0, 612), (x1, 612), (x1, 786), (x0, 786), (x0, 612)], (96, 66, 46), 0.8, 4)
    # 기단 (장대석)
    t3 = 4.6
    L.poly(t3, [(600, 792), (1600, 792), (1630, 836), (570, 836)], (150, 146, 138), 0.55)
    for x in range(610, 1600, 80):
        L.line(t3 + 0.1, [(x, 794), (x - 3, 834)], (90, 86, 80), 0.55, 2)
    L.line(t3 + 0.1, [(590, 814), (1610, 814)], (90, 86, 80), 0.5, 2)
    # 계단
    for k in range(3):
        L.poly(t3 + 0.2 + k * 0.05, [(1040 - k * 8, 836 + k * 10), (1160 + k * 8, 836 + k * 10), (1160 + k * 8, 846 + k * 10), (1040 - k * 8, 846 + k * 10)], (130, 126, 120), 0.6)
    # 오른쪽 담장 (기와 얹은 토담)
    t4 = 5.0
    L.poly(t4, [(1660, 666), (W + 60, 666), (W + 60, 836), (1660, 836)], (206, 186, 150), 0.42)
    for yy in range(690, 830, 30):
        for xx in range(1670 + (yy // 30 % 2) * 20, W + 40, 44):
            L.line(t4 + 0.2, [(xx, yy), (xx + 34, yy + 2)], (150, 130, 104), 0.5, 2)
    L.poly(t4 + 0.1, [(1650, 646), (W + 60, 640), (W + 60, 670), (1644, 674)], (48, 52, 58), 0.85)
    for xx in range(1656, W + 40, 16):
        L.line(t4 + 0.15, [(xx, 644), (xx, 670)], (26, 28, 32), 0.6, 2)
    return L


def build_pine():
    """붉은 줄기의 적송 + 층층이 펼쳐진 솔잎 뭉치."""
    L = Layer("pine")
    r = np.random.default_rng(1004)
    bark = [(124, 70, 46), (106, 60, 40), (140, 84, 54)]

    def stroke(pts, w0, w1, t0, speed):
        segs = list(zip(pts[:-1], pts[1:]))
        total = sum(math.dist(a, b) for a, b in segs)
        acc, out = 0.0, []
        for (ax, ay), (bx, by) in segs:
            d = math.dist((ax, ay), (bx, by))
            n = max(2, int(d / 2))
            for i in range(n):
                u = i / n
                x, y = ax + (bx - ax) * u, ay + (by - ay) * u
                s = (acc + d * u) / total
                w = (w0 + (w1 - w0) * s) * (1 + 0.08 * math.sin((acc + d * u) / 17))
                t = t0 + (acc + d * u) / speed
                L.dab(t, x + r.normal() * 0.4, y + r.normal() * 0.4, w / 2, bark[r.integers(0, 3)], 0.92)
                # 껍질 무늬 (먹선)
                if w > 14 and r.random() < 0.08:
                    ang = math.atan2(by - ay, bx - ax) + math.pi / 2
                    ox = math.cos(ang) * r.uniform(-w * 0.35, w * 0.35)
                    oy = math.sin(ang) * r.uniform(-w * 0.35, w * 0.35)
                    L.line(t + 0.05, [(x + ox - 6, y + oy - 2), (x + ox + 4, y + oy + 3), (x + ox + 9, y + oy)], (50, 30, 24), 0.85, 2.2)
                out.append((x, y, t, w))
            acc += d
        # 가장자리 먹선
        return out

    trunk = stroke([(70, 1110), (150, 900), (112, 760), (210, 600), (300, 470), (350, 360)], 74, 30, 1.0, 330)
    b1 = stroke([(212, 600), (130, 540), (40, 520)], 26, 8, trunk[len(trunk) * 3 // 5][2], 300)
    b2 = stroke([(300, 470), (430, 450), (560, 470), (690, 445)], 28, 8, trunk[len(trunk) * 4 // 5][2], 300)
    b3 = stroke([(350, 360), (300, 270), (330, 170)], 24, 9, trunk[-1][2], 300)
    b4 = stroke([(350, 360), (420, 330), (440, 380)], 16, 6, trunk[-1][2] + 0.1, 280)
    clusters = [(40, 500, 95, 0), (150, 520, 70, 0), (330, 160, 105, 0), (250, 250, 80, 0), (430, 330, 70, 0),
                (470, 440, 90, 0), (620, 455, 95, 0), (720, 440, 70, 0), (110, 700, 60, 0), (380, 230, 60, 0)]
    for ci, (cx, cy, rad, _) in enumerate(clusters):
        t0 = 3.4 + ci * 0.22
        # 연한 바탕 담채
        L.poly(t0 + 0.9, [(cx + rad * math.cos(a), cy + rad * 0.40 * math.sin(a) - rad * 0.05) for a in np.linspace(0, math.tau, 24)], (96, 128, 92), 0.28)
        # 솔잎: 위로 펼쳐지는 부채꼴 획
        n = int(rad * 7)
        for k in range(n):
            ox = r.uniform(-rad, rad)
            oy = r.uniform(-rad * 0.15, rad * 0.25)
            bx, by = cx + ox, cy + oy
            ang = -math.pi / 2 + (ox / rad) * 1.15 + r.normal() * 0.18
            ln = r.uniform(18, 36) * (1 - 0.35 * abs(ox) / rad)
            ex, ey = bx + math.cos(ang) * ln, by + math.sin(ang) * ln
            if in_text_zone(ex, ey):
                continue
            col = [(30, 58, 44), (40, 74, 52), (24, 48, 38)][r.integers(0, 3)]
            L.line(t0 + 0.05 + k * 0.0012, [(bx, by), (ex, ey)], col, 0.88, 3.0)
        # 아래 그늘선
        L.line(t0 + 0.3, [(cx - rad * 0.9, cy + rad * 0.22), (cx, cy + rad * 0.3), (cx + rad * 0.9, cy + rad * 0.2)], (30, 52, 40), 0.7, 3)
    return L


def build_sun():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = 1700, 190
    d = np.hypot(xx - cx, yy - cy) * (1 + 0.05 * (fbm(W, H, 90, 3, 31) - 0.5))
    a = smoothstep(96, 90, d) * (0.78 + 0.18 * fbm(W, H, 30, 3, 32))
    glow = smoothstep(260, 80, d) * 0.10
    return d, a.astype(np.float32), glow.astype(np.float32)


# ---------------------------------------------------------------- 학 (두루미)

def draw_crane(d, x, y, s, flap, facing=-1):
    """간단한 선묘 학: 흰 몸, 검은 목과 날개깃, 붉은 정수리."""
    f = facing
    body = [(x - 26 * s * f, y), (x + 18 * s * f, y - 6 * s), (x + 34 * s * f, y + 2 * s), (x + 10 * s * f, y + 8 * s)]
    wing_up = flap * 38 * s
    for side in (0, 1):
        yy = -wing_up if side == 0 else -wing_up * 0.6
        wing = [(x - 4 * s * f, y - 2 * s), (x + 8 * s * f, y + yy - 4 * s), (x + 36 * s * f, y + yy - 12 * s),
                (x + 22 * s * f, y + yy * 0.6), (x + 6 * s * f, y + 4 * s)]
        d.polygon(wing, fill=(250, 248, 242, 235), outline=(60, 56, 54, 220))
        tip = [(x + 36 * s * f, y + yy - 12 * s), (x + 24 * s * f, y + yy * 0.75 - 4 * s), (x + 30 * s * f, y + yy * 0.7 + 2 * s)]
        d.polygon(tip, fill=(30, 28, 28, 240))
    d.polygon(body, fill=(252, 250, 246, 245), outline=(60, 56, 54, 220))
    # 목과 머리
    hx, hy = x - 52 * s * f, y - 10 * s
    d.line([(x - 22 * s * f, y), (x - 40 * s * f, y - 6 * s), (hx, hy)], fill=(28, 26, 26, 240), width=max(2, int(4 * s)))
    d.ellipse([hx - 4 * s, hy - 4 * s, hx + 4 * s, hy + 4 * s], fill=(250, 248, 242, 255))
    d.ellipse([hx - 2.5 * s, hy - 5 * s, hx + 2 * s, hy - 1 * s], fill=(200, 30, 40, 255))
    d.line([(hx, hy), (hx - 14 * s * f, hy + 3 * s)], fill=(90, 80, 60, 255), width=max(1, int(2 * s)))
    # 다리
    d.line([(x + 30 * s * f, y + 4 * s), (x + 60 * s * f, y + 10 * s)], fill=(40, 36, 34, 220), width=max(1, int(2 * s)))


# ---------------------------------------------------------------- 한복 인물

SK = (246, 222, 200)
HAIR = (30, 26, 30)


def bow_pose(t, t0, lag=0.0):
    """세배 동작 곡선: (손 올림 h, 무릎 꿇기 k, 엎드림 b)."""
    x = t - t0 - lag
    if x < 0 or x > 4.8:
        return 0.0, 0.0, 0.0
    h = ease_io(x / 0.7) * (1 - ease_io((x - 4.1) / 0.7))
    k = ease_io((x - 0.7) / 0.9) * (1 - ease_io((x - 3.6) / 0.8))
    b = ease_io((x - 1.5) / 0.8) * (1 - ease_io((x - 2.9) / 0.7))
    return h, k, b


def bez(p0, p1, p2, n=10):
    return [((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
             (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]) for u in np.linspace(0, 1, n)]


def darker(c, k=0.7):
    return tuple(int(v * k) for v in c)


def draw_person(d, kind, cx, fy, S, h, k, b, breathe):
    """정면 인물. cx,fy: 발 중심, S: 배율(2배 렌더용)."""
    def P(x, y):
        return (cx + x * S, fy + y * S)

    def poly(pts, col, a=255, ow=2):
        d.polygon([P(*p) for p in pts], fill=col + (a,), outline=darker(col, 0.62) + (255,))

    woman = kind == "woman"
    L0 = 262 if woman else 250
    drop = 0.58 * L0 * k
    yw = -L0 + drop + breathe
    sh = 40 if woman else 52
    shoulder_y = yw - sh * (1 - 0.65 * b)
    sw = 50 if woman else 56
    head_r = 34 if woman else 36
    head_cy = shoulder_y - (14 + head_r) * (1 - 1.15 * b) + 4 * b
    hand_low = (0, yw + 26)
    hand_hi = (0, head_cy - head_r * 0.35)
    hand_floor = (0, -8)
    if woman:
        hp = (0, hand_low[1] * (1 - h) + hand_hi[1] * h)
    else:
        hp = (0, (hand_low[1] * (1 - h) + hand_hi[1] * h) * (1 - b) + hand_floor[1] * b)
    # 그림자
    d.ellipse([P(-120 - 40 * k, -10), P(120 + 40 * k, 14)], fill=(60, 50, 40, 60))
    if woman:
        # 치마 (다홍) — 무릎 꿇으면 넓고 낮게 퍼진다
        CH = (204, 46, 62)
        top = yw - 4
        wt, wb = 44 + 14 * k, 108 + 60 * k
        pts = [(-wt, top), (wt, top)] + bez((wt, top), (wb * 0.9, top * 0.45), (wb, -2), 12)[1:] + \
              bez((wb, -2), (0, 8 + 10 * k), (-wb, -2), 14)[1:] + bez((-wb, -2), (-wb * 0.9, top * 0.45), (-wt, top), 12)[1:]
        poly(pts, CH)
        for px in (-0.55, -0.2, 0.2, 0.55):
            d.line([P(px * wt, top + 10), P(px * wb * 0.92, -10)], fill=darker(CH, 0.8) + (200,), width=max(1, int(1.6 * S)))
        # 저고리 (연두) + 동정 + 고름
        JG = (124, 172, 98)
        poly([(-sw, shoulder_y + 6), (-sw + 6, shoulder_y - 2), (sw - 6, shoulder_y - 2), (sw, shoulder_y + 6),
              (sw - 4, yw + 6), (-sw + 4, yw + 6)], JG)
        d.line([P(-14, shoulder_y - 2), P(4, yw + 2)], fill=(250, 248, 240, 255), width=max(2, int(5 * S)))
        d.line([P(14, shoulder_y - 2), P(-2, yw - 8)], fill=(250, 248, 240, 255), width=max(2, int(5 * S)))
        if b < 0.6:
            d.line([P(6, yw - 4), P(18, yw + 60 - 20 * k)], fill=(196, 36, 56, 255), width=max(2, int(7 * S)))
            d.line([P(10, yw - 4), P(28, yw + 52 - 20 * k)], fill=(196, 36, 56, 255), width=max(2, int(6 * S)))
    else:
        # 두루마기 (쪽빛)
        CO = (56, 84, 140)
        hem = -36 + 32 * k
        wb = 88 + 48 * k
        pts = [(-sw, shoulder_y + 8), (-sw + 8, shoulder_y), (sw - 8, shoulder_y), (sw, shoulder_y + 8),
               (sw + 2, yw), (wb, hem), (-wb, hem), (-sw - 2, yw)]
        poly(pts, CO)
        d.line([P(0, shoulder_y + 20), P(4, hem)], fill=darker(CO, 0.72) + (255,), width=max(1, int(2 * S)))
        d.line([P(-16, shoulder_y), P(6, shoulder_y + 46)], fill=(250, 248, 240, 255), width=max(2, int(6 * S)))
        d.line([P(16, shoulder_y), P(-2, shoulder_y + 38)], fill=(250, 248, 240, 255), width=max(2, int(6 * S)))
        if b < 0.6:
            d.line([P(8, shoulder_y + 44), P(16, shoulder_y + 96)], fill=(214, 220, 236, 255), width=max(2, int(6 * S)))
        if k < 0.5:
            for sx in (-1, 1):  # 버선
                d.ellipse([P(sx * 22 - 16, hem - 2), P(sx * 22 + 16, 2)], fill=(250, 248, 242, 255), outline=(150, 140, 130, 255))
    # 목
    if b < 0.7 and head_cy + head_r - 6 < shoulder_y + 2:
        d.rectangle([P(-9, head_cy + head_r - 6), P(9, shoulder_y + 2)], fill=SK + (255,))
    # 머리
    face_a = 1 - sstep(0.25, 0.75, b)
    hr = head_r
    if face_a > 0.02:
        d.ellipse([P(-hr * 0.86, head_cy - hr), P(hr * 0.86, head_cy + hr)], fill=SK + (255,), outline=darker(SK, 0.8) + (255,))
        # 눈(웃는 눈), 볼, 입
        for sx in (-1, 1):
            ex, ey = sx * hr * 0.36, head_cy + hr * 0.08
            d.arc([P(ex - 7, ey - 5), P(ex + 7, ey + 5)], 200, 340, fill=(50, 40, 40, 255), width=max(1, int(2 * S)))
            d.ellipse([P(sx * hr * 0.5 - 7, head_cy + hr * 0.32 - 4), P(sx * hr * 0.5 + 7, head_cy + hr * 0.32 + 4)], fill=(240, 150, 150, 150))
        d.arc([P(-6, head_cy + hr * 0.38), P(6, head_cy + hr * 0.56)], 20, 160, fill=(190, 70, 70, 255), width=max(1, int(2 * S)))
        if not woman:
            for sx in (-1, 1):
                d.line([P(sx * hr * 0.2, head_cy - hr * 0.14), P(sx * hr * 0.55, head_cy - hr * 0.18)], fill=(40, 34, 34, 255), width=max(1, int(2.4 * S)))
    # 머리카락 (엎드리면 정수리 쪽이 보이도록 아래로 덮음)
    cover = hr * (0.15 + 1.1 * (1 - face_a))
    if woman:
        # 가르마를 탄 단정한 머리: 정수리에서 관자놀이로 내려오는 머리선
        if face_a > 0.02:
            outer = [(hr * 0.98 * math.cos(a), head_cy + hr * 1.02 * math.sin(a) - 2) for a in np.linspace(math.pi * 1.02, math.pi * 1.98, 18)]
            right_line = bez((hr * 0.9, head_cy + hr * 0.05), (hr * 0.55, head_cy - hr * 0.7), (0, head_cy - hr * 0.62), 10)
            left_line = bez((0, head_cy - hr * 0.62), (-hr * 0.55, head_cy - hr * 0.7), (-hr * 0.9, head_cy + hr * 0.05), 10)
            pts = outer + [(hr * 0.98, head_cy + hr * 0.1)] + right_line + left_line[1:] + [(-hr * 0.98, head_cy + hr * 0.1)]
            d.polygon([P(*q) for q in pts], fill=HAIR + (int(255 * face_a),))
            d.line([P(0, head_cy - hr * 1.0), P(0, head_cy - hr * 0.62)], fill=(70, 62, 62, int(255 * face_a)), width=max(1, int(1.4 * S)))
        if face_a < 0.98:
            ia = int(255 * (1 - face_a))
            d.ellipse([P(-hr * 0.95, head_cy - hr * 1.0), P(hr * 0.95, head_cy + hr * 0.95)], fill=HAIR + (ia,))
            d.line([P(0, head_cy - hr * 0.9), P(0, head_cy + hr * 0.6)], fill=(70, 62, 62, ia), width=max(1, int(1.4 * S)))
            # 쪽머리와 비녀 (엎드리면 보인다)
            d.ellipse([P(-hr * 0.42, head_cy - hr * 1.25), P(hr * 0.42, head_cy - hr * 0.78)], fill=HAIR + (ia,))
            d.line([P(-hr * 0.75, head_cy - hr * 1.0), P(hr * 0.75, head_cy - hr * 1.0)], fill=(214, 172, 80, ia), width=max(2, int(4 * S)))
    else:
        d.chord([P(-hr * 0.9, head_cy - hr * 1.02), P(hr * 0.9, head_cy - hr + 2 * (hr * 0.18 + cover))], 180, 360, fill=HAIR + (255,))
        if face_a < 0.98:
            d.ellipse([P(-hr * 0.9, head_cy - hr * 0.95), P(hr * 0.9, head_cy - hr * 0.95 + 2 * cover)], fill=HAIR + (int(255 * (1 - face_a)),))
        # 갓: 엎드리면 챙이 위에서 본 타원으로 커진다
        by = head_cy - hr * 0.88
        bh = 7 + 34 * b
        d.ellipse([P(-72, by - bh), P(72, by + bh)], fill=(22, 20, 22, 205), outline=(10, 10, 10, 255))
        if b < 0.6:
            d.polygon([P(-24, by - 2), P(24, by - 2), P(20, by - 44 * (1 - b)), P(-20, by - 44 * (1 - b))], fill=(18, 16, 18, 235))
            d.line([P(-hr * 0.8, by + 2), P(-6, head_cy + hr + 8)], fill=(40, 30, 30, 200), width=max(1, int(1.4 * S)))
            d.line([P(hr * 0.8, by + 2), P(6, head_cy + hr + 8)], fill=(40, 30, 30, 200), width=max(1, int(1.4 * S)))
        else:
            d.ellipse([P(-22, by - 18 * b), P(22, by + 10 * b)], fill=(14, 12, 14, 240))
    # 소매 (공수 → 이마 → 엎드림)
    SL = (124, 172, 98) if woman else (56, 84, 140)
    CUFF = (112, 60, 124) if woman else (250, 248, 240)
    hx, hy = hp
    for sx in (-1, 1):
        sxo = (shoulder_y + hy) / 2
        outer = bez((sx * (sw + 4), shoulder_y + 6), (sx * (sw + 34), sxo + 26), (sx * 12, hy + 16), 12)
        inner = bez((sx * 4, hy - 8), (sx * (sw - 4), sxo + 4), (sx * (sw - 18), shoulder_y + 26), 12)
        poly(outer + inner, SL)
        poly([(sx * 2, hy - 10), (sx * 16, hy - 12), (sx * 18, hy + 18), (sx * 2, hy + 16)], CUFF)
    d.ellipse([P(-12, hy - 6), P(12, hy + 12)], fill=SK + (255,), outline=darker(SK, 0.8) + (255,))


# ---------------------------------------------------------------- 복주머니

def pouch_sprite(fonts, size=120):
    S = 3
    s = size * S
    im = Image.new("RGBA", (s, int(s * 1.25)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([s * 0.08, s * 0.32, s * 0.92, s * 1.2], fill=(200, 36, 52, 255), outline=(130, 16, 30, 255), width=S * 2)
    d.polygon([(s * 0.3, s * 0.36), (s * 0.7, s * 0.36), (s * 0.82, s * 0.16), (s * 0.18, s * 0.16)], fill=(214, 52, 66, 255))
    d.rectangle([s * 0.26, s * 0.32, s * 0.74, s * 0.4], fill=(222, 178, 70, 255))
    d.line([(s * 0.5, s * 0.36), (s * 0.4, s * 0.02), (s * 0.6, s * 0.02), (s * 0.5, s * 0.36)], fill=(222, 178, 70, 255), width=S * 3)
    f = fonts.get("title", int(s * 0.38))
    bb = d.textbbox((0, 0), "복", font=f)
    d.ellipse([s * 0.28, s * 0.56, s * 0.72, s * 1.0], fill=(236, 196, 92, 255))
    d.text((s * 0.5 - (bb[2] - bb[0]) / 2 - bb[0], s * 0.78 - (bb[3] - bb[1]) / 2 - bb[1]), "복", font=f, fill=(170, 24, 36, 255))
    return im.resize((size, int(size * 1.25)), Image.LANCZOS)


# ---------------------------------------------------------------- 장면

BOW1 = 15.0
BOW2 = 47.6


class Scene:
    def __init__(self, fonts_dir):
        print("수묵 레이어 생성 중...", flush=True)
        self.paper = make_paper()
        self.mount = build_mountains_far()
        self.ground = build_ground(); self.ground.true_color = True; self.ground.rasterize(12, dry=0.25, bleed_r=10, bleed_a=0.1)
        self.hanok = build_hanok(); self.hanok.true_color = True; self.hanok.rasterize(13, dry=0.3, bleed_r=8, bleed_a=0.22)
        self.pine = build_pine(); self.pine.true_color = True; self.pine.rasterize(14, dry=0.45, bleed_r=10, bleed_a=0.28)
        self.sun_d, self.sun_a, self.sun_glow = build_sun()
        self.fonts = Fonts(fonts_dir)
        self.texts = self.build_texts()
        self.seal = make_seal(self.fonts, 86)
        self.pouch = pouch_sprite(self.fonts, 110)
        r = np.random.default_rng(9)
        self.flecks = [dict(x=r.random() * W, y=r.random() * H, vy=-(8 + r.random() * 14), amp=10 + r.random() * 20,
                            fr=0.1 + r.random() * 0.2, ph=r.random() * math.tau, s=r.integers(0, 2)) for _ in range(60)]
        self.fleck_spr = []
        for sz in (7, 11):
            g = np.mgrid[0:sz, 0:sz].astype(np.float32) - (sz - 1) / 2
            rr = np.hypot(g[0], g[1]) / (sz / 2)
            arr = np.zeros((sz, sz, 4), np.uint8)
            arr[..., 0], arr[..., 1], arr[..., 2] = 214, 160, 60
            arr[..., 3] = (np.clip(np.exp(-rr ** 2 * 4) + 0.8 * np.exp(-rr ** 2 * 30), 0, 1) * 255).astype(np.uint8)
            self.fleck_spr.append(Image.fromarray(arr, "RGBA"))
        self.burst = [(r.uniform(0, math.tau), r.uniform(120, 320), r.integers(0, 2)) for _ in range(40)]
        print("준비 완료", flush=True)

    def build_texts(self):
        F = self.fonts
        T = []
        cx = W / 2
        T.append(TextItem(render_line(F, [("2027 정미년", RED)], "body", 40, 8), cx, 168, 8.0, 14.0, rise=10))
        T.append(TextItem(render_line(F, [("새해가 밝았습니다", DARK)], "body", 64, 4), cx, 252, 8.6, 14.1))
        T.append(TextItem(render_line(F, [("새해 ", DARK), ("복", RED), (" 많이 받으세요", DARK)], "brush", 150, 6), cx, 205, 16.8, 25.4,
                          d_in=1.2, stagger=0.12, rise=22))
        T.append(TextItem(render_line(F, [("올 한 해 늘 ", DARK), ("건강", RED), ("하시고", DARK)], "body", 60, 2), cx, 170, 26.0, 32.4))
        T.append(TextItem(render_line(F, [("웃음 가득한 나날 되시길 바랍니다", DARK)], "body", 52, 2), cx, 262, 26.6, 32.5))
        T.append(TextItem(render_line(F, [("아픈 곳 없이 ", DARK), ("튼튼한", RED), (" 한 해", DARK)], "body", 60, 2), cx, 170, 33.0, 39.4))
        T.append(TextItem(render_line(F, [("신길튼튼365정형외과", (30, 92, 88)), ("가 늘 곁에 있겠습니다", DARK)], "body", 50, 2), cx, 262, 33.6, 39.5))
        T.append(TextItem(render_line(F, [("가정마다 행복과 평안이", DARK)], "body", 60, 2), cx, 170, 40.0, 46.4))
        T.append(TextItem(render_line(F, [("가득하시길 기원합니다", DARK)], "body", 56, 2), cx, 262, 40.6, 46.5))
        # 엔딩
        T.append(TextItem(render_line(F, [("새해 ", DARK), ("복", RED), (" 많이 받으세요", DARK)], "brush", 136, 6), cx, 160, 49.0, 99,
                          d_in=1.2, stagger=0.11, rise=22))
        T.append(TextItem(render_line(F, [("신길튼튼365정형외과 임직원 일동", DARK)], "title", 46, 3), cx, 276, 50.6, 99))
        T.append(TextItem(render_line(F, [("서울특별시 영등포구 가마산로 483  3A01호~3A05호", (90, 80, 72))], "reg", 28, 1), cx, 330, 51.4, 99, rise=8))
        return T

    def figures(self, canvas, t):
        a = sstep(6.4, 7.4, t)
        if a <= 0.01:
            return
        S = 2
        x0, y0, w, h = 560, 470, 800, 600
        im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for kind, cx, lag, phase in (("woman", 830, 0.0, 0.0), ("man", 1090, 0.18, 1.3)):
            hh, kk, bb = bow_pose(t, BOW1, lag)
            if (hh, kk, bb) == (0.0, 0.0, 0.0):
                hh, kk, bb = bow_pose(t, BOW2, lag)
            br = 1.2 * math.sin(t * 1.6 + phase)
            draw_person(d, kind, (cx - x0) * S, (1010 - y0) * S, S, hh, kk, bb, br)
        im = im.reduce(S)
        # 수채 질감
        arr = np.asarray(im).astype(np.float32)
        tex = self._figtex if hasattr(self, "_figtex") else None
        if tex is None:
            self._figtex = 0.93 + 0.1 * fbm(w, h, 18, 3, 77)[..., None]
            tex = self._figtex
        arr[..., :3] *= tex
        arr[..., 3] *= a
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        canvas.alpha_composite(im, (x0, y0 + int(20 * (1 - ease_out_cubic((t - 6.4) / 1.0)))))

    def frame(self, t):
        img = self.paper.copy()
        # 붉은 해: 먹 번지듯 퍼짐
        if t > 0.4:
            p = ease_out_cubic((t - 0.4) / 2.0)
            m = smoothstep(-4, 10, 30 + 80 * p - self.sun_d * (1.0 / max(0.05, 1.0)) + 0)
            m = smoothstep(0, 14, (40 + 80 * p) - (self.sun_d - 0))
            a = np.clip(self.sun_a * np.minimum(1, m) * 1.0, 0, 1)
            img *= 1 - a[..., None] * (1 - np.array([0.80, 0.16, 0.18], np.float32))
            img *= 1 - (self.sun_glow * sstep(1.0, 3.0, t))[..., None] * (1 - np.array([1.0, 0.7, 0.6], np.float32))
        for a_, col_, T_ in self.mount:
            m_ = a_ * smoothstep(0, 1.6, t - T_)
            img *= 1 - m_[..., None] * (1 - col_)
        self.ground.apply(img, t)
        self.hanok.apply(img, t)
        self.pine.apply(img, t)
        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        # 학
        d = ImageDraw.Draw(canvas)
        for (t0, y0, s, sp) in ((1.6, 330, 1.0, 150), (2.4, 300, 0.8, 140), (27.0, 340, 0.9, 130)):
            tt = t - t0
            if 0 < tt < 16:
                x = W + 80 - sp * tt
                y = y0 - 10 * math.sin(tt * 0.7) - tt * 3
                flap = 0.5 + 0.5 * math.sin(tt * 5.5 + t0)
                draw_crane(d, x, y, s, flap, facing=-1)
        self.figures(canvas, t)
        # 복주머니: 두 사람 사이에서 떠올라 금빛으로 터진다
        if 33.0 < t < 40.5:
            q = (t - 33.0) / 6.0
            y = 800 - 330 * ease_out_cubic(q)
            sc = ease_out_back(min(1, (t - 33.0) / 0.6), 2.0) * (1 + 0.03 * math.sin(t * 5))
            if t < 39.0:
                im = self.pouch.resize((max(1, int(self.pouch.size[0] * sc)), max(1, int(self.pouch.size[1] * sc))), Image.BILINEAR)
                im = im.rotate(6 * math.sin(t * 2.2), Image.BICUBIC, expand=True)
                canvas.alpha_composite(im, (int(960 - im.size[0] / 2), int(y - im.size[1] / 2)))
            else:
                bt = t - 39.0
                for (ang, sp, k) in self.burst:
                    rr = sp * ease_out_cubic(bt / 1.2)
                    x = 960 + math.cos(ang) * rr
                    yy = 470 + math.sin(ang) * rr + 60 * bt * bt
                    a = 1 - bt / 1.5
                    if a > 0:
                        arr = np.asarray(self.fleck_spr[k]).copy()
                        arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
                        canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (int(x), int(yy)))
        # 금빛 티끌
        for f in self.flecks:
            if t < 6:
                break
            tt = t - 6
            x = (f["x"] + f["amp"] * math.sin(f["fr"] * math.tau * tt + f["ph"])) % W
            y = (f["y"] + f["vy"] * tt) % H
            a = sstep(0, 2, tt) * (0.3 + 0.7 * math.sin(tt * 1.3 + f["ph"]) ** 2) * 0.7
            arr = np.asarray(self.fleck_spr[f["s"]]).copy()
            arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
            canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (int(x), int(y)))
        # 카메라: 아주 느린 줌
        z = 1.0 + 0.04 * sstep(0, DUR, t)
        cw, ch = W / z, H / z
        ox, oy = (W - cw) / 2, (H - ch) * 0.8
        canvas = canvas.resize((W, H), Image.BICUBIC, box=(ox, oy, ox + cw, oy + ch))
        for tx in self.texts:
            tx.draw(canvas, t)
        # 낙관
        if t > 52.2:
            k = (t - 52.2) / 0.45
            s = 1.6 - 0.6 * ease_out_cubic(k) if k < 1 else 1.0
            sz = max(4, int(86 * s))
            im = self.seal.resize((sz, sz), Image.BILINEAR).rotate(-4, Image.BICUBIC, expand=True)
            arr = np.asarray(im).copy()
            arr[..., 3] = (arr[..., 3] * min(1, k * 1.8) * 0.92).astype(np.uint8)
            canvas.alpha_composite(Image.fromarray(arr, "RGBA"), (int(1440 - im.size[0] / 2), int(160 - im.size[1] / 2)))
        out = canvas.convert("RGB")
        f_in = sstep(0, 0.8, t)
        f_out = 1 - sstep(DUR - 1.4, DUR - 0.05, t)
        bg = Image.new("RGB", (W, H), (250, 247, 240))
        if f_in < 1:
            out = Image.blend(bg, out, f_in)
        if f_out < 1:
            out = Image.blend(bg, out, f_out)
        return out


# ---------------------------------------------------------------- 국악풍 음악

def janggu_low(amp=0.5):
    n = int(0.45 * SR); t = np.arange(n) / SR
    f = 70 + 60 * np.exp(-t / 0.04)
    y = np.sin(math.tau * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    return (y * amp).astype(np.float32)


def janggu_high(amp=0.25, seed=0):
    n = int(0.12 * SR); t = np.arange(n) / SR
    x = np.random.default_rng(seed).normal(size=n).astype(np.float32)
    x = np.diff(x, prepend=0)
    y = x / (np.max(np.abs(x)) + 1e-9) * np.exp(-t / 0.035) + 0.5 * np.sin(math.tau * 520 * t) * np.exp(-t / 0.05)
    return (y * amp).astype(np.float32)


def flute(freq, dur, amp=0.12):
    n = int(dur * SR); t = np.arange(n) / SR
    vib = 1 + 0.008 * np.sin(math.tau * 5.0 * t) * smoothstep(0.2, 0.8, t)
    ph = math.tau * np.cumsum(freq * vib) / SR
    y = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    breath = np.random.default_rng(int(freq)).normal(size=n).astype(np.float32) * 0.05
    env = smoothstep(0, 0.12, t) * (1 - smoothstep(dur - 0.25, dur, t))
    return ((y + breath) * env * amp).astype(np.float32)


def make_music(path):
    n = int((DUR + 3) * SR)
    Lc = np.zeros(n, np.float32); Rc = np.zeros(n, np.float32)

    def add(sig, t, pan_=0.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[: n - i]
        gl, gr = math.cos((pan_ + 1) * math.pi / 4), math.sin((pan_ + 1) * math.pi / 4)
        Lc[i:i + len(sig)] += sig * gl
        Rc[i:i + len(sig)] += sig * gr

    bpm = 84
    bt = 60 / bpm
    # 바탕 지속음 (D + A)
    for k in range(0, 64, 8):
        add(pad([midi_hz(50), midi_hz(57), midi_hz(62)], 8 * bt + 2.0, 0.08), k * bt)
    # 장구 (덩 · 기덕 · 쿵 · 더러러)
    for bar in range(4, 17):
        b0 = bar * 4 * bt
        soft = 0.6 if bar >= 14 else 1.0
        add(janggu_low(0.45 * soft), b0); add(janggu_high(0.22 * soft, bar), b0)
        add(janggu_high(0.16 * soft, bar + 50), b0 + 1.5 * bt); add(janggu_high(0.14 * soft, bar + 80), b0 + 1.75 * bt)
        add(janggu_low(0.38 * soft), b0 + 2 * bt)
        for j in range(3):
            add(janggu_high(0.12 * soft, bar * 3 + j), b0 + 3 * bt + j * bt / 4)
    # 가야금 선율 (D 5음 음계)
    P = [(0, 74, 1), (1, 76, .5), (1.5, 78, .5), (2, 81, 1), (3, 78, 1), (4, 76, 1.5), (5.5, 74, .5), (6, 71, 1), (7, 74, 1),
         (8, 76, 1), (9, 78, .5), (9.5, 81, .5), (10, 83, 1), (11, 81, 1), (12, 78, 1), (13, 76, 1), (14, 74, 2)]
    P2 = [(0, 81, 1), (1, 83, .5), (1.5, 86, .5), (2, 83, 1), (3, 81, 1), (4, 78, 1.5), (5.5, 76, .5), (6, 78, 1), (7, 81, 1),
          (8, 83, 1), (9, 81, .5), (9.5, 78, .5), (10, 76, 1), (11, 78, 1), (12, 74, 1), (13, 76, 1), (14, 74, 2)]
    for (b0, ph, a) in ((16, P, 0.18), (32, P2, 0.18), (48, P, 0.17), (64, P2[:9], 0.15)):
        for (b, m, dd) in ph:
            add(pluck(midi_hz(m), max(1.4, dd * bt + 1.0), a), (b0 + b) * bt, 0.15)
            add(pluck(midi_hz(m - 12), 1.2, a * 0.35), (b0 + b) * bt + 0.01, -0.3)
    # 대금풍 긴 선율: 도입과 끝
    for (b, m, dd) in [(1, 69, 3), (4, 74, 2), (6, 76, 2), (8, 78, 4), (12, 74, 4),
                       (66, 78, 3), (69, 76, 2), (71, 74, 5)]:
        add(flute(midi_hz(m), dd * bt + 0.3, 0.11), b * bt, -0.1)
    # 종소리: 세배 · 제목 · 낙관
    for tt, m, a in [(0.4, 86, 0.08), (BOW1 + 2.0, 74, 0.12), (16.8, 81, 0.09), (33.0, 86, 0.07), (39.0, 90, 0.07),
                     (BOW2 + 2.0, 74, 0.12), (49.0, 81, 0.09), (52.2, 69, 0.12)]:
        add(bell(midi_hz(m), 4.0, a), tt, 0.2)

    def reverb(x, seed):
        rr = np.random.default_rng(seed)
        ir_n = int(2.4 * SR)
        t_ = np.arange(ir_n) / SR
        ir = rr.normal(size=ir_n).astype(np.float32) * np.exp(-t_ / 0.6)
        ir[: int(0.012 * SR)] = 0
        ir = np.convolve(ir, np.ones(10) / 10, mode="same")
        nfft = 1 << int(math.ceil(math.log2(len(x) + ir_n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
        return y / np.sqrt(np.sum(ir ** 2)) * 0.06

    L2 = Lc + reverb(Lc, 1) * 1.8
    R2 = Rc + reverb(Rc, 2) * 1.8
    out = np.stack([L2, R2], 1)[: int(DUR * SR)]
    t = np.arange(len(out)) / SR
    out *= (np.minimum(1, t / 0.6) * (1 - smoothstep(DUR - 3.0, DUR, t)))[:, None]
    out /= np.max(np.abs(out)) + 1e-9
    out = np.tanh(out * 1.15) / np.tanh(1.15) * 0.89
    pcm = (out * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


SCENE = None


def render_chunk(args):
    idx, f0, f1, outdir = args
    path = os.path.join(outdir, f"seollal_seg_{idx:02d}.mp4")
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
            SCENE.frame(float(s)).save(os.path.join(work, f"seollal_{float(s):05.1f}.png"))
        return
    wav = os.path.join(work, "seollal_music.wav")
    make_music(wav)
    print("음악 생성 완료", flush=True)
    nj = max(1, a.jobs)
    edges = [round(i * NFRAMES / nj) for i in range(nj + 1)]
    with Pool(nj) as pool:
        segs = pool.map(render_chunk, [(i, edges[i], edges[i + 1], work) for i in range(nj)])
    lst = os.path.join(work, "seollal_segs.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("완료:", a.out)


if __name__ == "__main__":
    main()
