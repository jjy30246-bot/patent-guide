#!/usr/bin/env python3
"""신길튼튼365정형외과 · Merry Christmas & Happy New Year — 모던 버전

render_city.py 의 문구·음악·타이밍을 그대로 쓰고, 화면만 현대적인 디자인으로 바꾼다.
- 천천히 흐르는 파스텔 메시 그라디언트 + 필름 그레인
- 반투명 유리 같은 흰색 스카이라인
- 통통 튀며(wobble) 떠다니는 3D 구·링·필·반짝이
- 탄성 있게 튀어나오는 유리 카드 위의 문구
- 선과 점으로 터지는 모던 스타버스트 불꽃

사용:  python3 render_modern.py --fonts <폰트폴더> --out <출력.mp4> [--preview 초,...]
"""
import argparse
import math
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import render_city as rc
from render_city import (W, H, FPS, DUR, NFRAMES, T_COUNT, T_DROP, T_DAWN, T_END, BEAT,
                         GOLD, CORAL, MINT, PINK, SKY, NAVY, fade, pan, rise_off, make_music)
from render import smoothstep, sstep, ease_out_cubic, ease_out_back, value_noise

LAV = (150, 136, 240)


def wobble(x, amp=0.08, k=5.0, w=16.0):
    """탄성 흔들림: 등장 직후 몇 번 출렁이다 멈춘다."""
    if x <= 0:
        return 1.0
    return 1 + amp * math.exp(-k * x) * math.sin(w * x)


# ---------------------------------------------------------------- 3D 느낌 오브젝트

def sphere_sprite(D, col):
    S = 3
    n = int(D * 1.3 * S)
    g = np.mgrid[0:n, 0:n].astype(np.float32) - (n - 1) / 2
    R = D / 2 * S
    gx, gy = g[1] / R, g[0] / R
    d = np.hypot(gx, gy)
    a = smoothstep(1.0, 0.97, d)
    hl = np.exp(-(((gx + 0.38) ** 2 + (gy + 0.42) ** 2) / 0.06))
    rim = np.exp(-(((gx - 0.35) ** 2 + (gy - 0.5) ** 2) / 0.25)) * 0.25
    c = np.array(col, np.float32)
    light = np.clip(c * 1.25 + 30, 0, 255)
    dark = c * 0.62
    k = np.clip((gx + gy + 1.4) / 2.8, 0, 1)[..., None]
    rgb = light * (1 - k) + dark * k
    rgb = rgb * (1 - hl[..., None] * 0.85) + 255 * hl[..., None] * 0.85
    rgb = rgb + rim[..., None] * 255 * 0.5
    arr = np.zeros((n, n, 4), np.float32)
    arr[..., :3] = rgb
    arr[..., 3] = a * 255
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    return im.resize((n // S, n // S), Image.LANCZOS)


def ring_sprite(D, col, tilt=0.42):
    S = 3
    n = int(D * 1.2 * S)
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = n / 2
    rx, ry = D / 2 * S, D / 2 * S * tilt
    th = D * 0.16 * S
    c1 = tuple(int(min(255, v * 0.7)) for v in col)
    c2 = tuple(int(min(255, v * 1.0)) for v in col)
    c3 = tuple(int(min(255, v * 1.25 + 40)) for v in col)
    d.ellipse([c - rx, c - ry, c + rx, c + ry], outline=c1 + (255,), width=int(th))
    d.ellipse([c - rx + th * 0.15, c - ry - th * 0.12, c + rx - th * 0.15, c + ry - th * 0.12], outline=c2 + (255,), width=int(th * 0.62))
    d.ellipse([c - rx + th * 0.3, c - ry - th * 0.22, c + rx - th * 0.3, c + ry - th * 0.22], outline=c3 + (230,), width=int(th * 0.22))
    return im.resize((n // S, n // S), Image.LANCZOS)


def pill_sprite(L, Wd, col):
    S = 3
    im = Image.new("RGBA", (int(L * S), int(Wd * S)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, L * S - 1, Wd * S - 1], radius=Wd * S / 2, fill=tuple(int(v * 0.78) for v in col) + (255,))
    d.rounded_rectangle([L * S * 0.03, Wd * S * 0.06, L * S * 0.97, Wd * S * 0.72], radius=Wd * S * 0.33, fill=col + (255,))
    d.rounded_rectangle([L * S * 0.12, Wd * S * 0.14, L * S * 0.6, Wd * S * 0.32], radius=Wd * S * 0.1,
                        fill=tuple(int(min(255, v * 1.2 + 60)) for v in col) + (200,))
    return im.resize((int(L), int(Wd)), Image.LANCZOS)


def sparkle_sprite(D, col):
    S = 4
    n = int(D * S)
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = n / 2
    pts = []
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        r = n / 2 if k % 2 == 0 else n * 0.11
        pts.append((c + math.cos(a) * r, c + math.sin(a) * r))
    d.polygon(pts, fill=col + (255,))
    return im.resize((int(D), int(D)), Image.LANCZOS)


def shadow_sprite(w, h, alpha=0.16, blur=14):
    pad = blur * 3
    im = Image.new("L", (int(w + pad * 2), int(h + pad * 2)), 0)
    ImageDraw.Draw(im).ellipse([pad, pad, pad + w, pad + h], fill=int(255 * alpha))
    im = im.filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", im.size, NAVY + (0,))
    out.putalpha(im)
    return out


class ModernScene(rc.Scene):
    def __init__(self, fonts_dir):
        super().__init__(fonts_dir)
        r = np.random.default_rng(2027)
        # 메시 그라디언트 블롭 (1/4 해상도에서 합성)
        self.qw, self.qh = W // 4, H // 4
        qy, qx = np.mgrid[0:self.qh, 0:self.qw].astype(np.float32)
        self.qx, self.qy = qx, qy
        self.blobs = [
            dict(c=(196, 186, 255), cd=(255, 196, 176), x=0.15, y=0.2, r=0.45, fx=0.05, fy=0.07, ph=0.0),
            dict(c=(170, 214, 255), cd=(255, 214, 160), x=0.85, y=0.15, r=0.5, fx=0.04, fy=0.05, ph=1.3),
            dict(c=(255, 204, 196), cd=(255, 170, 170), x=0.7, y=0.75, r=0.5, fx=0.06, fy=0.04, ph=2.1),
            dict(c=(190, 240, 222), cd=(255, 226, 190), x=0.25, y=0.8, r=0.45, fx=0.05, fy=0.06, ph=3.7),
            dict(c=(255, 206, 232), cd=(250, 190, 220), x=0.5, y=0.45, r=0.38, fx=0.07, fy=0.05, ph=5.0),
        ]
        self.grain = [((value_noise(W, H, 1.0, 600 + k) - 0.5) * 0.035).astype(np.float32)[..., None] for k in range(3)]
        # 스카이라인 그림자
        self.city_shadow = []
        for L in self.city:
            m = (np.clip(L["mask"], 0, 1) * 255).astype(np.uint8)
            sh = np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(16)), np.float32) / 255
            self.city_shadow.append(np.roll(sh, -10, axis=0))
        # 떠다니는 오브젝트
        spec = [
            ("sphere", CORAL, 230, 250, 130, 0.6), ("sphere", GOLD, 1700, 210, 96, 0.5),
            ("sphere", MINT, 1590, 790, 72, 0.9), ("sphere", LAV, 330, 760, 60, 0.8),
            ("ring", SKY, 1735, 560, 160, 0.7), ("ring", PINK, 160, 520, 118, 0.7),
            ("ring", GOLD, 1430, 140, 120, 0.4), ("sparkle", MINT, 520, 120, 58, 0.5),
            ("sparkle", GOLD, 1300, 120, 52, 0.3), ("sparkle", CORAL, 640, 860, 40, 1.0),
            ("sparkle", LAV, 1860, 900, 46, 1.0), ("sphere", PINK, 1250, 900, 44, 1.0),
            ("sparkle", MINT, 90, 140, 36, 0.4), ("sphere", SKY, 860, 110, 40, 0.3),
        ]
        self.shapes = []
        for i, (kind, col, x, y, sz, dep) in enumerate(spec):
            if kind == "sphere":
                spr = sphere_sprite(sz, col)
            elif kind == "ring":
                spr = ring_sprite(sz, col)
            elif kind == "pill":
                spr = pill_sprite(sz, sz * 0.36, col)
            else:
                spr = sparkle_sprite(sz, col)
            self.shapes.append(dict(kind=kind, spr=spr, x=x, y=y, sz=sz, dep=dep, t=0.9 + i * 0.12,
                                    amp=10 + r.random() * 14, per=3.0 + r.random() * 2.5, ph=r.random() * math.tau,
                                    rot0=r.uniform(-30, 30), vr=r.uniform(-12, 12),
                                    sh=shadow_sprite(sz * 0.8, sz * 0.18) if kind == "sphere" else None))
        # 유리 카드 일정 (t_in, t_out, cx, cy, w, h)
        self.cards = [
            (4.8, 12.2, 960, 478, 860, 460),
            (12.6, 19.95, 960, 470, 1040, 250),
            (20.2, 27.45, 960, 510, 1140, 300),
            (45.0, 52.1, 960, 348, 1080, 210),
        ]
        self._card_shadow = {}
        # 스타버스트 불꽃
        fwcol = [GOLD, CORAL, MINT, PINK, SKY, LAV]
        for i, f in enumerate(self.fw):
            f["c1"] = fwcol[i % len(fwcol)]
            f["c2"] = fwcol[(i + 2) % len(fwcol)]
            f["L"] = 150 + (i * 37) % 80
            f["n"] = 14 + (i % 3) * 2
            f["a0"] = (i * 0.37) % 1.0
        print("모던 장면 준비 완료", flush=True)

    # ---------------- 배경: 메시 그라디언트
    def mesh(self, t):
        dawn = sstep(T_DAWN - 0.5, T_DAWN + 5.5, t)
        top = np.array(rc.lerpc((238, 242, 255), (255, 236, 228), dawn), np.float32)
        bot = np.array(rc.lerpc((255, 248, 242), (255, 226, 210), dawn), np.float32)
        yy = (self.qy / self.qh)[..., None]
        img = top * (1 - yy) + bot * yy
        for b in self.blobs:
            cx = (b["x"] + b["fx"] * math.sin(t * 0.35 + b["ph"])) * self.qw
            cy = (b["y"] + b["fy"] * math.cos(t * 0.3 + b["ph"] * 1.3)) * self.qh
            rr = b["r"] * self.qw
            g = np.exp(-(((self.qx - cx) ** 2 + (self.qy - cy) ** 2) / (rr * rr)))[..., None] * 0.75
            col = np.array(rc.lerpc(b["c"], b["cd"], dawn), np.float32)
            img = img * (1 - g) + col * g
        q = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").resize((W, H), Image.BICUBIC)
        out = np.asarray(q, np.float32) / 255
        return out

    def glass(self, canvas, cx, cy, w, h, a, radius=None, s=1.0):
        w, h = int(w * s), int(h * s)
        radius = int((radius if radius is not None else 36) * s)
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        key = (w // 8, h // 8, radius // 4)
        sh = self._card_shadow.get(key)
        if sh is None:
            pad = 40
            m = Image.new("L", (w + pad * 2, h + pad * 2), 0)
            ImageDraw.Draw(m).rounded_rectangle([pad, pad, pad + w, pad + h], radius=radius, fill=60)
            m = m.filter(ImageFilter.GaussianBlur(22))
            sh = Image.new("RGBA", m.size, NAVY + (0,))
            sh.putalpha(m)
            self._card_shadow[key] = sh
        box = (max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h))
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw <= 2 or bh <= 2:
            return
        region = canvas.crop(box).filter(ImageFilter.GaussianBlur(20)).convert("RGBA")
        canvas.alpha_composite(fade(sh, a), (x0 - 40, y0 - 40 + 16))
        tint = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        dt = ImageDraw.Draw(tint)
        dt.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=radius, fill=(255, 255, 255, 120))
        dt.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=radius, outline=(255, 255, 255, 235), width=2)
        region.alpha_composite(tint)
        mask = Image.new("L", (bw, bh), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=radius, fill=int(255 * a))
        canvas.paste(region, box[:2], mask)

    def starbursts(self, t):
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        any_ = False
        for f in self.fw:
            dt = t - f["t0"]
            if -0.6 < dt < 0:
                q = (dt + 0.6) / 0.6
                y = rc.lerp(H + 10, f["y"], ease_out_cubic(q))
                d.line([(f["x"], y), (f["x"], y + 70 * (1 - q) + 10)], fill=f["c1"] + (200,), width=4)
                d.ellipse([f["x"] - 5, y - 5, f["x"] + 5, y + 5], fill=f["c1"] + (255,))
                any_ = True
            if 0 <= dt < 1.6:
                any_ = True
                Lm = f["L"]
                r1 = Lm * ease_out_cubic(min(1, dt / 0.42))
                r0 = Lm * (min(1, max(0, (dt - 0.18) / 0.75)) ** 1.6)
                al = int(255 * (1 - sstep(1.0, 1.6, dt)))
                for k in range(f["n"]):
                    a = (k / f["n"] + f["a0"]) * math.tau
                    c = f["c1"] if k % 2 == 0 else f["c2"]
                    ll = 1.0 if k % 2 == 0 else 0.72
                    if r1 * ll - r0 * ll > 2:
                        x0, y0 = f["x"] + math.cos(a) * (r0 * ll + 8), f["y"] + math.sin(a) * (r0 * ll + 8)
                        x1, y1 = f["x"] + math.cos(a) * r1 * ll, f["y"] + math.sin(a) * r1 * ll
                        d.line([(x0, y0), (x1, y1)], fill=c + (al,), width=7)
                        d.ellipse([x1 - 3.5, y1 - 3.5, x1 + 3.5, y1 + 3.5], fill=c + (al,))
                    if dt > 0.35:
                        rr = Lm * ll * (1.12 + 0.15 * (dt - 0.35))
                        ds = max(0, 6 * (1 - (dt - 0.35) / 1.2))
                        xd, yd = f["x"] + math.cos(a + 0.2) * rr, f["y"] + math.sin(a + 0.2) * rr + 20 * (dt - 0.35) ** 2
                        if ds > 0.5:
                            d.ellipse([xd - ds, yd - ds, xd + ds, yd + ds], fill=c + (al,))
                # 충격파 링
                rw = Lm * 0.75 * ease_out_cubic(min(1, dt / 0.6))
                ra = int(200 * (1 - min(1, dt / 0.7)))
                if ra > 0:
                    d.ellipse([f["x"] - rw, f["y"] - rw, f["x"] + rw, f["y"] + rw], outline=f["c2"] + (ra,), width=4)
        return ov if any_ else None

    def frame(self, t):
        img = self.mesh(t)
        dawn = sstep(T_DAWN - 0.5, T_DAWN + 5.5, t)
        # 새해 해 (그라디언트 원)
        if dawn > 0.01:
            sy = int(rc.lerp(1150, 640, ease_out_cubic((t - T_DAWN) / 7.0)))
            R = 170
            y0, y1 = max(0, sy - R - 2), min(H, sy + R + 2)
            if y1 > y0:
                gx = self.gx_full[y0:y1, 960 - R - 2:960 + R + 2]
                gy = self.gy_full[y0:y1, 960 - R - 2:960 + R + 2]
                dd = np.hypot(gx - 960, gy - sy)
                a = smoothstep(R, R - 3, dd)[..., None] * dawn
                k = np.clip((gy - (sy - R)) / (2 * R), 0, 1)[..., None]
                col = np.array([1.0, 0.86, 0.55], np.float32) * (1 - k) + np.array([1.0, 0.58, 0.58], np.float32) * k
                sub = img[y0:y1, 960 - R - 2:960 + R + 2]
                sub[:] = sub * (1 - a) + col * a
        # 유리 스카이라인
        p = pan(t)
        alphas = [0.42, 0.58, 0.78]
        for li, L in enumerate(self.city):
            off = int(p * L["depth"])
            dy = rise_off(t, li)
            m = L["mask"][:, off:off + W]
            sh = self.city_shadow[li][:, off:off + W]
            if dy > 0:
                m = np.vstack([np.zeros((dy, W), np.float32), m[:H - dy]])
                sh = np.vstack([np.zeros((dy, W), np.float32), sh[:H - dy]])
            img *= (1 - 0.10 * sh)[..., None]
            wcol = np.array(rc.lerpc((255, 255, 255), (255, 244, 240), dawn), np.float32) / 255
            a = (m * alphas[li])[..., None]
            img = img * (1 - a) + wcol * a
        img = img + self.grain[int(t * FPS) % 3]
        canvas = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB").convert("RGBA")
        # 떠다니는 오브젝트 (탄성 등장)
        for s in self.shapes:
            k = t - s["t"]
            if k <= 0:
                continue
            sc = ease_out_back(min(1, k / 0.5), 2.2) * wobble(k, 0.1, 4, 14)
            fy = s["amp"] * math.sin(math.tau * t / s["per"] + s["ph"])
            x = s["x"] - p * 0.25 * s["dep"]
            y = s["y"] + fy
            spr = s["spr"]
            if s["kind"] in ("ring", "pill"):
                spr = spr.rotate(s["rot0"] + s["vr"] * t + 6 * math.sin(t * 1.3 + s["ph"]), Image.BICUBIC, expand=True)
            elif s["kind"] == "sparkle":
                sc *= 0.85 + 0.15 * math.sin(t * 3 + s["ph"])
                spr = spr.rotate(t * 30 + s["ph"] * 50, Image.BICUBIC, expand=True)
            if abs(sc - 1) > 0.01:
                spr = spr.resize((max(1, int(spr.size[0] * sc)), max(1, int(spr.size[1] * sc))), Image.BILINEAR)
            if s["sh"] is not None:
                shs = s["sh"]
                canvas.alpha_composite(fade(shs, min(1, k * 2) * (0.7 - 0.3 * fy / max(1, s["amp"]))),
                                       (int(x - shs.size[0] / 2), int(s["y"] + s["sz"] * 0.62 - shs.size[1] / 2)))
            canvas.alpha_composite(fade(spr, min(1, k * 3)), (int(x - spr.size[0] / 2), int(y - spr.size[1] / 2)))
        # 선물상자 (탄성)
        gf = 1 - sstep(T_COUNT - 0.8, T_COUNT, t)
        for g in self.gifts:
            k = (t - g["t"]) / 0.55
            if k <= 0 or gf <= 0:
                continue
            s_ = (ease_out_back(k, 2.4) if k < 1 else 1.0) * wobble(t - g["t"] - 0.3, 0.06, 4, 18)
            im = g["img"]
            nh = max(1, int(im.size[1] * s_))
            im2 = im.resize((max(1, int(im.size[0] * (0.7 + 0.3 * s_))), nh), Image.BILINEAR)
            yb = 1062 + 140 * (1 - gf)
            canvas.alpha_composite(fade(im2, gf), (int(g["x"] - im2.size[0] / 2), int(yb - nh)))
        canvas = self.camera(canvas, t)
        # 눈 (반짝이는 흰 점)
        sf = sstep(3.0, 5.0, t) * (1 - sstep(T_DAWN - 1, T_DAWN + 2, t))
        if sf > 0.01:
            for f in self.snow[::2]:
                y = (f["y"] + f["v"] * t) % (H + 40) - 20
                x = (f["x"] + f["amp"] * math.sin(f["fr"] * math.tau * t + f["ph"])) % W
                canvas.alpha_composite(fade(self.snow_spr[f["dep"]], sf * 0.95), (int(x), int(y)))
        # 컨페티
        for c in self.confetti:
            dt = t - c["t0"]
            if dt < 0:
                continue
            if c["burst"]:
                x = c["x"] + c["vx"] * (1 - math.exp(-2.2 * dt)) / 2.2 + 25 * math.sin(dt * 3 + c["ph"])
                y = c["y"] + c["vy"] * (1 - math.exp(-2.2 * dt)) / 2.2 + 60 * dt * dt + 60 * dt
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
        # 스타버스트
        ov = self.starbursts(t)
        if ov is not None:
            canvas.alpha_composite(ov)
        # 새 (일출)
        if t > T_DAWN + 1.5:
            d = ImageDraw.Draw(canvas)
            for b in self.birds:
                tt = t - T_DAWN - 1.5
                x = b["x"] + tt * 170
                y = b["y"] - tt * 12 + 6 * math.sin(tt * 1.3 + b["ph"])
                f = math.sin(tt * 9 + b["ph"]) * b["s"] * 0.55
                s = b["s"]
                d.line([(x - s, y - f), (x, y), (x + s, y - f)], fill=(70, 74, 130, 255), width=3)
        # 유리 카드 (탄성 등장)
        for (ti, to, cx, cy, w, h) in self.cards:
            if ti <= t <= to + 0.45:
                x = t - ti
                grow = ease_out_cubic(min(1, x / 0.35))
                s_ = (0.86 + 0.14 * grow) * wobble(x, 0.06, 5, 16)
                a = min(1, x / 0.25)
                if t > to:
                    q = (t - to) / 0.45
                    a *= 1 - q
                    s_ *= 1 - 0.06 * q
                self.glass(canvas, cx, cy, w, h, a, s=s_)
        # 카운트다운 (유리 원 + 숫자 + 진행 링)
        for k, (t0, im) in enumerate(self.count):
            dt = t - t0
            dur = 2 * BEAT
            if 0 <= dt < dur:
                if k == 0 or dt < 0.02:
                    pass
                a = min(1, dt / 0.08) * (1 - sstep(dur - 0.12, dur, dt))
                cs = wobble(dt, 0.08, 6, 18)
                self.glass(canvas, 960, 530, 440, 440, 1.0 if k > 0 or dt > 0.1 else min(1, dt / 0.1), radius=220, s=cs)
                s = 1.45 - 0.45 * ease_out_back(min(1, dt / 0.28), 2.2)
                if dt > dur - 0.12:
                    s *= 1 - 0.35 * (dt - (dur - 0.12)) / 0.12
                im2 = im.resize((max(1, int(im.size[0] * s * 0.8)), max(1, int(im.size[1] * s * 0.8))), Image.BILINEAR)
                canvas.alpha_composite(fade(im2, a), (int(960 - im2.size[0] / 2), int(520 - im2.size[1] / 2)))
                ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                dd = ImageDraw.Draw(ov)
                rr = 200
                dd.arc([960 - rr, 530 - rr, 960 + rr, 530 + rr], -90, -90 + 360 * min(1, dt / dur),
                       fill=[CORAL, MINT, GOLD, PINK, SKY][k] + (255,), width=10)
                canvas.alpha_composite(ov)
        # 엔딩 유리 패널
        if t > T_END:
            x = t - T_END
            s_ = (0.86 + 0.14 * ease_out_cubic(min(1, x / 0.4))) * wobble(x, 0.05, 4, 14)
            self.glass(canvas, 960, 530, 1180, 400, min(1, x / 0.4), radius=44, s=s_)
        # 문구
        for tx in self.texts:
            tx.draw(canvas, t)
        out = np.asarray(canvas.convert("RGB"), np.float32) / 255
        if 0 <= t - T_DROP < 0.4:
            out = out + (1 - out) * 0.5 * (1 - (t - T_DROP) / 0.4)
        f_in = sstep(0, 0.8, t)
        f_out = 1 - sstep(DUR - 1.2, DUR - 0.05, t)
        white = np.array([1.0, 0.99, 0.98], np.float32)
        out = out * f_in * f_out + white * (1 - f_in * f_out)
        return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGB")


SCENE = None


def render_chunk(args):
    idx, f0, f1, outdir = args
    path = os.path.join(outdir, f"modern_seg_{idx:02d}.mp4")
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
    SCENE = ModernScene(a.fonts)
    if a.preview:
        for s in a.preview.split(","):
            SCENE.frame(float(s)).save(os.path.join(work, f"modern_{float(s):05.1f}.png"))
        return
    wav = os.path.join(work, "modern_music.wav")
    make_music(wav, SCENE.fw_times)
    print("음악 생성 완료", flush=True)
    nj = max(1, a.jobs)
    edges = [round(i * NFRAMES / nj) for i in range(nj + 1)]
    with Pool(nj) as pool:
        segs = pool.map(render_chunk, [(i, edges[i], edges[i + 1], work) for i in range(nj)])
    lst = os.path.join(work, "modern_segs.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out])
    print("완료:", a.out)


if __name__ == "__main__":
    main()
