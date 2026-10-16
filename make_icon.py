#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 snore.ico —— 深色圆角底 + 白色声波柱(其中一根是鼾声尖峰)。
纯 numpy/zlib 手写, 不依赖 Pillow。同时输出 preview.png 方便肉眼检查。
"""
import os
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ICO = os.path.join(HERE, "snore.ico")
OUT_PNG = os.path.join(HERE, "snore_preview.png")

# 配色
C_TOP = (0x1D, 0x4E, 0xD8)     # 蓝
C_BOT = (0x0E, 0xA5, 0xE9)     # 青
C_BAR = (0xFF, 0xFF, 0xFF)
BARS = [0.20, 0.44, 0.90, 0.56, 0.30]   # 中间那根 = 鼾声尖峰


def _rr_coverage(n, radius, x0, y0, x1, y1):
    """圆角矩形覆盖率(0~1), 已抗锯齿。"""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hx, hy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    dx = np.maximum(np.abs(x - cx) - (hx - radius), 0.0)
    dy = np.maximum(np.abs(y - cy) - (hy - radius), 0.0)
    d = np.sqrt(dx * dx + dy * dy) - radius
    return np.clip(0.5 - d, 0.0, 1.0)


def render(n, ss=4):
    N = n * ss
    f = float(N)

    # 圆角底 + 渐变
    bg_cov = _rr_coverage(N, radius=0.19 * f, x0=0.0, y0=0.0, x1=f, y1=f)
    t = np.linspace(0.0, 1.0, N)[:, None, None]
    grad = (np.array(C_TOP, float) * (1 - t) + np.array(C_BOT, float) * t)

    rgb = np.zeros((N, N, 3), float)
    rgb += grad

    # 声波柱
    nb = len(BARS)
    span = 0.68 * f                      # 五根柱占的总宽
    bw = span / (nb + (nb - 1) * 0.5)    # 柱宽(间隙 = 半柱宽)
    gap = bw * 0.5
    x = (f - (nb * bw + (nb - 1) * gap)) / 2.0
    for h in BARS:
        bh = h * 0.62 * f
        y0 = (f - bh) / 2.0
        cov = _rr_coverage(N, radius=bw / 2.0, x0=x, y0=y0, x1=x + bw, y1=y0 + bh)
        rgb = rgb * (1 - cov[:, :, None]) + np.array(C_BAR, float) * cov[:, :, None]
        x += bw + gap

    # 底在外的地方 alpha=0
    alpha = bg_cov

    # 降采样抗锯齿
    img = np.dstack([rgb, alpha * 255.0]).reshape(n, ss, n, ss, 4).mean(axis=(1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def png_bytes(rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def ico_bytes(images):
    n = len(images)
    out = struct.pack("<HHH", 0, 1, n)
    entries, blob, off = [], b"", 6 + 16 * n
    for size, img in images:
        hdr = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, 0, 0, 0, 0, 0)
        xor = img[:, :, [2, 1, 0, 3]][::-1].tobytes()          # BGRA, 自下而上
        mask = b"\x00" * (((size + 31) // 32) * 4 * size)      # AND 掩码全 0
        body = hdr + xor + mask
        w = 0 if size >= 256 else size
        entries.append(struct.pack("<BBBBHHII", w, w, 0, 0, 1, 32, len(body), off))
        blob += body
        off += len(body)
    return out + b"".join(entries) + blob


if __name__ == "__main__":
    sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    imgs = [(s, render(s)) for s in sizes]
    with open(OUT_ICO, "wb") as f:
        f.write(ico_bytes(imgs))
    with open(OUT_PNG, "wb") as f:
        f.write(png_bytes(render(256)))
    print(f"已生成 {OUT_ICO}  {os.path.getsize(OUT_ICO)} 字节, {len(sizes)} 个尺寸")
    print(f"预览   {OUT_PNG}")
