#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打鼾检测器 · 第一阶段(电脑端)

链路:  麦克风 -> 分块 -> 特征(音量 / 低频占比) -> 事件状态机
        -> 存 WAV 片段(可人耳复核) + events CSV + 每秒音量日志 + 夜间汇总

用法:
    python snore_detect.py                     # 今晚采一整晚, Ctrl+C 结束出报告
    python snore_detect.py --calibrate 30      # 先标定房间噪声底(30 秒)
    python snore_detect.py --replay x.wav      # 离线跑一个音频文件, 不出片段
    python snore_detect.py --help
"""
import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import webbrowser
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------- 参数 ----------------
SR = 16000            # 采样率; snore 基频 20~300Hz, 16k 足够
BLOCK_S = 0.1         # 分块 100ms
BLOCK_N = int(SR * BLOCK_S)
PREROLL_S = 5.0       # 片段前留 5 秒上下文
POSTROLL_S = 0.5      # 片段后留 0.5 秒
GAP_S = 0.6           # 事件内部容许的静音间隔, 短于此算同一次
MIN_EVENT_S = 0.35    # 短于此不算一次打鼾
MAX_EVENT_S = 60.0    # 单次事件上限, 防止噪声卡住时内存爆掉

DEFAULT_CFG = {
    "abs_floor_dbfs": -45.0,   # 绝对音量门: 低于此一律不算(挡底噪)
    "rel_margin_db": 12.0,     # 相对门: 高于噪声底多少 dB 才算"响"
    "low_ratio": 0.55,         # 低频能量占比门(鼾声低频集中)
    "low_band": [40.0, 400.0], # 低频带宽 Hz
    "noise_window_s": 120.0,   # 噪声底估计窗口
    "noise_pct": 15.0,         # 噪声底取分位数
    "warmup_s": 20.0,          # 开场只学习噪声底, 不判定
    "cluster_gap_s": 5.0,      # 事件间隔短于此归为同一"鼾声串"
    "cluster_min": 3,          # 一"串"里至少几个才算真打鼾
}


def load_cfg(path):
    cfg = dict(DEFAULT_CFG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


# ---------------- 特征 ----------------
def block_features(block, low_band):
    """返回 (dbfs, low_ratio, centroid_hz)"""
    x = block.astype(np.float64)
    x = x - x.mean()
    n = len(x)
    if n == 0:
        return -120.0, 0.0, 0.0
    rms = float(np.sqrt(np.mean(x * x)))
    dbfs = 20.0 * np.log10(rms + 1e-12)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    tot = float(spec.sum()) + 1e-12
    band = (freqs >= low_band[0]) & (freqs <= low_band[1])
    low = float(spec[band].sum()) / tot
    centroid = float((freqs * spec).sum() / tot)
    return dbfs, low, centroid


class NoiseFloor:
    """滚动估计房间噪声底(取分位数), 鼾声只是偶尔响, 分位数不会被拖高。"""

    def __init__(self, window_s, pct):
        self.buf = deque(maxlen=int(window_s / BLOCK_S))
        self.pct = pct
        self.fixed = None

    def update(self, dbfs):
        self.buf.append(dbfs)

    def value(self):
        if self.fixed is not None:
            return self.fixed
        if len(self.buf) < 20:
            return -90.0
        return float(np.percentile(self.buf, self.pct))


# ---------------- 事件状态机 ----------------
class Detector:
    def __init__(self, cfg, on_event=None):
        self.cfg = cfg
        self.low_band = cfg["low_band"]
        self.warmup_s = cfg["warmup_s"]
        self.floor = NoiseFloor(cfg["noise_window_s"], cfg["noise_pct"])
        self.ring = deque(maxlen=int(PREROLL_S / BLOCK_S))
        self.ev_blocks = None      # 事件进行中累积的块
        self.ev_start_t = None
        self.last_snore_t = None
        self.n_events = 0
        self.total_snore_s = 0.0
        self.peak_dbfs = -120.0
        self.on_event = on_event   # callback(event_dict)

    def is_snore(self, dbfs, low):
        cfg = self.cfg
        if dbfs < cfg["abs_floor_dbfs"]:
            return False
        if dbfs < self.floor.value() + cfg["rel_margin_db"]:
            return False
        return low >= cfg["low_ratio"]

    def feed(self, block, t):
        """喂一个块, t = 该块结束时刻(秒, 相对开始)。"""
        dbfs, low, cent = block_features(block, self.low_band)
        self.floor.update(dbfs)
        self.peak_dbfs = max(self.peak_dbfs, dbfs)
        self.ring.append(block)
        # 开场学习期: 只喂噪声底, 不判定(避免冷启动把一切当鼾声)
        if t < self.warmup_s:
            return None
        snore = self.is_snore(dbfs, low)
        closed = None

        if self.ev_blocks is None:
            if snore:
                # 事件开始: 把环形缓冲里的历史当 pre-roll
                self.ev_blocks = list(self.ring)[:-1]
                self.ev_start_t = t
                self.last_snore_t = t
                self.ev_blocks.append(block)
                self.ev_meta = {"peak": dbfs, "dbfs": [dbfs], "low": [low],
                                "centroid": [cent]}
        else:
            self.ev_blocks.append(block)
            self.ev_meta["peak"] = max(self.ev_meta["peak"], dbfs)
            self.ev_meta["dbfs"].append(dbfs)
            self.ev_meta["low"].append(low)
            self.ev_meta["centroid"].append(cent)
            if snore:
                self.last_snore_t = t
            gap = t - self.last_snore_t
            too_long = (t - self.ev_start_t) > MAX_EVENT_S
            if gap > GAP_S or too_long:
                closed = self._close(t, gap)

        if closed and self.on_event:
            self.on_event(closed)
        return closed

    def _close(self, t, gap):
        blocks = self.ev_blocks
        dur = self.last_snore_t - self.ev_start_t
        # 裁掉超出 post-roll 的尾巴
        keep_s = (self.last_snore_t - self.ev_start_t) + PREROLL_S + POSTROLL_S
        keep = min(len(blocks), int(keep_s / BLOCK_S))
        audio = np.concatenate(blocks[:keep]) if keep > 0 else np.zeros(0, np.float32)
        meta = self.ev_meta
        ev = {
            "start_t": self.ev_start_t,
            "duration_s": max(0.0, dur),
            "peak_dbfs": meta["peak"],
            "mean_dbfs": float(np.mean(meta["dbfs"])),
            "mean_low_ratio": float(np.mean(meta["low"])),
            "mean_centroid": float(np.mean(meta["centroid"])),
            "audio": audio,
        }
        self.ev_blocks = None
        self.ev_start_t = None
        self.last_snore_t = None
        self.ev_meta = None
        if ev["duration_s"] >= MIN_EVENT_S:
            self.n_events += 1
            self.total_snore_s += ev["duration_s"]
            return ev
        return None

    def flush(self):
        """结束时的收尾: 事件还没关就关掉。"""
        if self.ev_blocks is not None:
            ev = self._close(self.last_snore_t + GAP_S + 1e-6, 0)
            if ev and self.on_event:
                self.on_event(ev)
            return ev
        return None


# ---------------- 输出 ----------------
def write_wav(path, float_audio):
    a = np.clip(float_audio, -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def fmt_hms(sec):
    return str(timedelta(seconds=int(sec)))


# ---------------- 图文报告 (自包含 HTML, 内联 SVG, 零依赖) ----------------
REPORT_CSS = """
:root{--ink:#1d1d1f;--ink2:#6e6e73;--ink3:#8e8e93;--line:rgba(0,0,0,.07);
 --blue:#0071e3;--orange:#ff9500;--red:#ff3b30;--green:#34c759}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);letter-spacing:-.01em;
 background-color:#f5f5f7;background-repeat:no-repeat;background-attachment:fixed;
 background-image:
  radial-gradient(40% 32% at 12% 4%,rgba(0,113,227,.24),transparent 72%),
  radial-gradient(36% 30% at 90% 0%,rgba(255,149,0,.22),transparent 72%),
  radial-gradient(46% 38% at 74% 96%,rgba(52,199,89,.17),transparent 72%),
  radial-gradient(32% 28% at 2% 90%,rgba(175,82,222,.15),transparent 72%);
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
 -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{max-width:1000px;margin:0 auto;padding:44px 22px 72px}
h1{font-size:34px;font-weight:700;letter-spacing:-.035em;margin:0 0 7px}
.sub{color:var(--ink2);font-size:14px;margin-bottom:28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(164px,1fr));gap:16px;margin-bottom:24px}
.card,section{background:rgba(255,255,255,.72);
 -webkit-backdrop-filter:blur(20px) saturate(180%);backdrop-filter:blur(20px) saturate(180%);
 border:1px solid rgba(255,255,255,.7);box-shadow:0 8px 30px rgba(0,0,0,.06)}
.card{border-radius:22px;padding:18px 20px}
.card .k{font-size:12px;color:var(--ink2);margin-bottom:9px;font-weight:500}
.card .v{font-size:30px;font-weight:600;line-height:1.08;letter-spacing:-.025em}
.card .u{font-size:14px;font-weight:400;color:var(--ink3);margin-left:4px}
.card .h{font-size:12px;color:var(--ink3);margin-top:7px}
.hi{color:var(--red)}.mid{color:var(--orange)}.ok{color:var(--green)}
section{border-radius:22px;padding:22px 24px 24px;margin-bottom:20px}
section h2{font-size:17px;margin:0 0 16px;font-weight:600;letter-spacing:-.015em}
.legend{font-size:12px;color:var(--ink2);margin-top:12px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:4px;margin:0 6px 0 16px;vertical-align:-1px}
.legend i:first-child{margin-left:0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600;font-size:12px}
tr.likely{background:rgba(0,113,227,.055)}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:600}
.pill.y{background:rgba(0,113,227,.15);color:#0a5bbd}
.pill.n{background:rgba(120,120,128,.14);color:#48484a}
audio{height:30px;vertical-align:middle}
.clipcell{color:var(--ink3);font-size:11px;margin-left:6px}
.empty{color:var(--ink3);font-size:13px;padding:16px 0}
.note{background:rgba(0,113,227,.09);border:1px solid rgba(0,113,227,.20);
 padding:14px 18px;border-radius:16px;font-size:13px;color:#0b4a8f;margin-bottom:24px;line-height:1.7}
.foot{color:var(--ink3);font-size:12px;text-align:center;margin-top:36px;line-height:1.9}
@media print{body{background:#fff;background-image:none}.wrap{padding:0}
 .card,section{background:#fff;-webkit-backdrop-filter:none;backdrop-filter:none;
  box-shadow:none;border:1px solid #e5e5ea;break-inside:avoid}
 audio{display:none}.clipcell{font-size:11px}}
"""


def _esc(s):
    return html.escape(str(s), quote=True)


def _hms(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _svgtxt(x, y, s, size=11, fill="#8e8e93", anchor="middle", weight="400"):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>')


def _glass_defs(sfx):
    """凝胶柱用的渐变定义(每张图各一份, 免得 id 撞车)。"""
    b, g, s = f"glassBlue{sfx}", f"glassGray{sfx}", f"glassGloss{sfx}"
    return (
        "<defs>"
        f'<linearGradient id="{b}" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#BFE0FF" stop-opacity="0.98"/>'
        '<stop offset="38%" stop-color="#6FB6FF" stop-opacity="0.88"/>'
        '<stop offset="100%" stop-color="#2E84E6" stop-opacity="0.72"/>'
        "</linearGradient>"
        f'<linearGradient id="{g}" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#EEEEF2" stop-opacity="0.96"/>'
        '<stop offset="100%" stop-color="#C4C4CB" stop-opacity="0.86"/>'
        "</linearGradient>"
        f'<linearGradient id="{s}" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.66"/>'
        '<stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>'
        "</linearGradient>"
        "</defs>"
    )


def _svg_timeline(rows, total_s, t0, w=920, h=66):
    """整夜时间条: 每个鼾声事件一格, 横轴是时间。"""
    pl, pr, pt, ph = 8, 8, 14, 26
    inner = w - pl - pr
    x = lambda t: pl + (t / total_s) * inner if total_s else pl
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="display:block">', _glass_defs("T")]
    p.append(f'<rect x="{pl}" y="{pt}" width="{inner}" height="{ph}" rx="8" fill="#e8e8ed"/>')
    # 小时刻度
    step = 3600.0
    if total_s > 12 * 3600:
        step = 7200.0
    t = 0.0
    while t <= total_s:
        xx = x(t)
        anc = "start" if xx < 26 else ("end" if xx > w - pr - 26 else "middle")
        p.append(f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt+ph}" stroke="#d1d1d6" stroke-width="1"/>')
        p.append(_svgtxt(xx, pt - 4, (t0 + timedelta(seconds=t)).strftime("%H:%M"),
                         anchor=anc))
        t += step
    for ev in rows:
        x0, x1 = x(ev["start_t"]), x(ev["start_t"] + max(ev["duration_s"], 1.0))
        col = "url(#glassBlueT)" if ev["likely_snore"] else "url(#glassGrayT)"
        bw3 = max(3.5, x1 - x0)
        p.append(f'<rect x="{x0:.1f}" y="{pt}" width="{bw3:.1f}" height="{ph}" '
                 f'fill="{col}" rx="3.5" stroke="#ffffff" stroke-opacity="0.55" stroke-width="0.9"/>')
        if bw3 > 6:                                        # 窄条不画高光, 不然糊成一团
            p.append(f'<rect x="{x0+1:.1f}" y="{pt+1.4:.1f}" width="{bw3-2:.1f}" '
                     f'height="{ph*0.42:.1f}" rx="2.6" fill="url(#glassGlossT)"/>')
    p.append("</svg>")
    return "".join(p)


def _svg_hourly(rows, total_s, t0, w=920, h=170):
    """每小时打了多少秒鼾。"""
    nh = max(1, int(np.ceil(total_s / 3600.0)))
    per = np.zeros(nh)
    for ev in rows:
        if ev["likely_snore"]:
            per[min(nh - 1, int(ev["start_t"] // 3600))] += ev["duration_s"]
    pl, pr, pt, pb = 46, 10, 12, 30
    ih = h - pt - pb
    bw = (w - pl - pr) / nh
    mx = max(1.0, per.max())
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="display:block">', _glass_defs("H")]
    for gv in (0, mx / 2, mx):
        yy = pt + ih - (gv / mx) * ih
        p.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="#f0f0f2"/>')
        p.append(_svgtxt(pl - 8, yy + 4, f"{int(gv)}s", 10, "#8e8e93", "end"))
    for i, v in enumerate(per):
        bh = (v / mx) * ih
        xx = pl + i * bw + bw * 0.16
        bwid = bw * 0.68
        if v > 0:
            p.append(f'<rect x="{xx:.1f}" y="{pt+ih-bh:.1f}" width="{bwid:.1f}" '
                     f'height="{max(bh,1.5):.1f}" rx="6" fill="url(#glassBlueH)" '
                     f'stroke="#ffffff" stroke-opacity="0.6" stroke-width="1"/>')
            gh = min(bh * 0.40, 11.0)                      # 顶部高光 = 凝胶质感
            if bh > 5 and bwid > 6:
                p.append(f'<rect x="{xx+1.6:.1f}" y="{pt+ih-bh+1.4:.1f}" '
                         f'width="{bwid-3.2:.1f}" height="{gh:.1f}" rx="{min(5.0, gh/2):.1f}" '
                         f'fill="url(#glassGlossH)"/>')
        else:
            p.append(f'<rect x="{xx:.1f}" y="{pt+ih-1.5:.1f}" width="{bwid:.1f}" '
                     f'height="1.5" rx="0.75" fill="#e8e8ed"/>')
        lbl = (t0 + timedelta(hours=i)).strftime("%H:%M")
        if bw > 34 or i % 2 == 0:
            p.append(_svgtxt(xx + bwid / 2, h - 10, lbl, 10))
    p.append("</svg>")
    return "".join(p)


def _svg_levels(segs, rows, total_s, thresh, w=920, h=210):
    """整夜音量: 每秒 dBFS 的上下包络 + 鼾声事件阴影 + 判定门。"""
    pl, pr, pt, pb = 46, 10, 12, 30
    iw, ih = w - pl - pr, h - pt - pb
    DB0, DB1 = 0.0, -100.0
    x = lambda t: pl + (t / total_s) * iw if total_s else pl
    y = lambda db: pt + (DB0 - max(DB1, min(DB0, db))) / (DB0 - DB1) * ih
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="display:block">', _glass_defs("L")]
    for gd in (0, -20, -40, -60, -80, -100):
        yy = y(gd)
        p.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" stroke="#f0f0f2"/>')
        p.append(_svgtxt(pl - 8, yy + 4, f"{gd}", 10, "#8e8e93", "end"))
    # 事件阴影
    for ev in rows:
        if ev["likely_snore"]:
            x0, x1 = x(ev["start_t"]), x(ev["start_t"] + ev["duration_s"])
            p.append(f'<rect x="{x0:.1f}" y="{pt}" width="{max(1.5,x1-x0):.1f}" '
                     f'height="{ih}" fill="#7FBFFF" opacity="0.18" rx="2"/>')
    # 音量包络
    if segs and total_s:
        nb = min(iw, 900)
        idx = np.clip((np.array([s[0] for s in segs]) / total_s * nb).astype(int), 0, nb - 1)
        dv = np.array([s[1] for s in segs])
        top, bot = np.full(nb, -100.0), np.full(nb, 0.0)
        for i, d in zip(idx, dv):
            top[i] = max(top[i], d)
            bot[i] = min(bot[i], d)
        xs = [pl + (i + 0.5) / nb * iw for i in range(nb)]
        up = " ".join(f"{xs[i]:.1f},{y(top[i]):.1f}" for i in range(nb))
        dn = " ".join(f"{xs[i]:.1f},{y(bot[i]):.1f}" for i in range(nb - 1, -1, -1))
        p.append(f'<polygon points="{up} {dn}" fill="#0071e3" opacity="0.40"/>')
        p.append(f'<polyline points="{up}" fill="none" stroke="#0071e3" stroke-width="1.2"/>')
    # 判定门
    yt = y(thresh)
    p.append(f'<line x1="{pl}" y1="{yt:.1f}" x2="{w-pr}" y2="{yt:.1f}" stroke="#ff3b30" '
             f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    p.append(_svgtxt(w - pr - 4, yt - 6, f"判定门 {thresh:.0f} dBFS", 10, "#ff3b30", "end"))
    # 时间轴
    step = 3600.0 if total_s <= 12 * 3600 else 7200.0
    t = 0.0
    while t <= total_s:
        xx = x(t)
        anc = "start" if xx < 26 else ("end" if xx > w - pr - 26 else "middle")
        p.append(_svgtxt(xx, h - 10, _hms(t), 10, anchor=anc))
        t += step
    p.append("</svg>")
    return "".join(p)


def write_report(outdir, tag, summary, rows, segs, cfg, note=None):
    total = summary["total_recorded_s"] or 1.0
    t0 = datetime.strptime(summary["start"], "%Y-%m-%d %H:%M:%S")
    likely = [e for e in rows if e["likely_snore"]]
    n_lik, sec_lik = len(likely), summary["total_likely_snore_s"]
    frac = sec_lik / total * 100 if total else 0.0

    hrs = total / 3600.0
    eph = n_lik / hrs if hrs > 0 else 0.0          # 成串鼾声「次/小时」
    if summary["n_snore_events"] == 0:
        verdict, vcls = "整夜未检出鼾声", "ok"
        advice = ("没有检出任何鼾声事件。如果 <b>levels 曲线整夜贴在 -90 dBFS 上下</b>，"
                  "那多半是麦克风没录到（而不是真的没打鼾），请改用手机录音那条路重测。")
    elif frac >= 15 or eph >= 45:
        verdict, vcls = f"鼾声较重 · {n_lik} 次 / 占整夜 {frac:.1f}%", "hi"
        advice = "次数和时长都偏多。建议重点听几个片段确认判定是否准确；若确实如此，值得去看睡眠门诊。"
    elif frac >= 4 or eph >= 12:
        verdict, vcls = f"有打鼾 · {n_lik} 次 / 占整夜 {frac:.1f}%", "mid"
        advice = "存在成串的鼾声，属常见范围。先听片段确认准确性，再观察几晚趋势。"
    else:
        verdict, vcls = f"轻微 · {n_lik} 次 / 占整夜 {frac:.1f}%", "ok"
        advice = "只有零星鼾声。注意区分这些是不是翻身、关门之类的噪声（听片段即可判断）。"

    if rows:
        first = (t0 + timedelta(seconds=rows[0]["start_t"])).strftime("%H:%M")
        last = (t0 + timedelta(seconds=rows[-1]["start_t"])).strftime("%H:%M")
        span = f"{first} – {last}"
    else:
        span = "—"

    cards = [
        ("成串鼾声", f"{n_lik}", "次", f"共 {summary['n_clusters']} 串", "hi" if n_lik else "ok"),
        ("鼾声总时长", _hms(sec_lik), "", f"占录音 {frac:.1f}%", "mid" if sec_lik else "ok"),
        ("夜间最响", f"{summary['peak_dbfs']:.0f}", "dBFS", f"环境底噪 {summary['noise_floor_dbfs']:.0f}", ""),
        ("鼾声时段", span, "", f"录音 {_hms(total)}", ""),
        ("疑似事件", f"{summary['n_snore_events']}", "次", "含未成串的孤立响声", ""),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div>'
        f'<div class="v {c}">{v}<span class="u">{u}</span></div>'
        f'<div class="h">{hh}</div></div>' for k, v, u, hh, c in cards)

    if rows:
        MAX_ROWS = 80
        trs = []
        for i, ev in enumerate(rows, 1):
            if i > MAX_ROWS:
                break
            ts = (t0 + timedelta(seconds=ev["start_t"])).strftime("%H:%M:%S")
            clip = ev.get("clip", "")
            if clip:
                media = f'<audio controls preload="none" src="clips/{_esc(clip)}"></audio>'
                media += f'<span class="clipcell">{_esc(clip)}</span>'
            else:
                media = '<span class="clipcell">(未存)</span>'
            pill = ('<span class="pill y">成串</span>' if ev["likely_snore"]
                    else '<span class="pill n">孤立</span>')
            trs.append(
                f'<tr class="{"likely" if ev["likely_snore"] else "iso"}">'
                f'<td>{i}</td><td>{ts}</td><td>{ev["duration_s"]:.1f}s</td>'
                f'<td>{ev["peak_dbfs"]:.1f}</td><td>{ev["mean_low_ratio"]:.2f}</td>'
                f'<td>{pill}</td><td>{media}</td></tr>')
        table = ("<table><thead><tr><th>#</th><th>时间</th><th>时长</th><th>峰值 dBFS</th>"
                 "<th>低频占比</th><th>判定</th><th>片段（可点播复核）</th></tr></thead><tbody>"
                 + "".join(trs) + "</tbody></table>")
        if len(rows) > MAX_ROWS:
            table += (f'<div class="empty">仅显示前 {MAX_ROWS} 条；'
                      f'其余 {len(rows) - MAX_ROWS} 条见同目录 events CSV。</div>')
    else:
        table = '<div class="empty">整夜没有检出任何鼾声事件。</div>'

    note_html = f'<div class="note">{note}</div>' if note else ""
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>打鼾报告 {_esc(summary['start'])}</title><style>{REPORT_CSS}</style></head><body>
<div class="wrap">
<h1>打鼾监测报告</h1>
<div class="sub">{_esc(summary['start'])} 起 · 录音 {_hms(total)} · 麦克风采样 16 kHz</div>
{note_html}
<div class="cards">{card_html}</div>
<section><h2>整夜时间轴</h2>
{_svg_timeline(rows, total, t0)}
<div class="legend"><i style="background:linear-gradient(180deg,#BFE0FF,#2E84E6)"></i>成串鼾声<i style="background:linear-gradient(180deg,#EEEEF2,#C4C4CB)"></i>孤立响声</div>
</section>
<section><h2>每小时鼾声时长</h2>
{_svg_hourly(rows, total, t0)}
</section>
<section><h2>整夜音量曲线</h2>
{_svg_levels(segs, rows, total, summary['cfg']['abs_floor_dbfs'])}
<div class="legend">深蓝色线为每秒音量（上下包络）；浅蓝色阴影为判定出的鼾声时段；红色虚线为绝对判定门。</div>
</section>
<section><h2>逐次事件明细</h2>{table}</section>
<section><h2>结论</h2>
<p style="font-size:15px;font-weight:600;margin:0 0 8px" class="{vcls}">{verdict}</p>
<p style="font-size:13px;color:#475569;line-height:1.8;margin:0">{advice}</p>
</section>
<div class="foot">由 snore_detect.py 生成 · 判定门 {summary['cfg']['abs_floor_dbfs']} dBFS
 / 噪声底+{summary['cfg']['rel_margin_db']} dB / 低频占比&gt;{summary['cfg']['low_ratio']}<br>
 想存成 PDF: 用浏览器打开本页 → Ctrl+P → 目标选「另存为 PDF」</div>
</div></body></html>"""

    path = os.path.join(outdir, f"report_{tag}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


# ---------------- 标定 ----------------
def calibrate(device, seconds, cfg_path):
    cfg = load_cfg(cfg_path)
    print(f"标定中… 请保持房间日常安静(该有的底噪都有), {seconds} 秒")
    lvl = []
    with sd.InputStream(device=device, samplerate=SR, channels=1,
                        blocksize=BLOCK_N, dtype="float32") as s:
        n = int(seconds / BLOCK_S)
        for i in range(n):
            blk, _ = s.read(BLOCK_N)
            d, low, _c = block_features(blk[:, 0], cfg["low_band"])
            lvl.append(d)
            if i % 10 == 0:
                print(f"  {i*BLOCK_S:5.1f}s  {d:7.1f} dBFS")
    lvl = np.array(lvl)
    noise = float(np.percentile(lvl, cfg["noise_pct"]))
    med = float(np.median(lvl))
    # 注意: 阵列麦克风常把"无声"闸到 -90 以下, 据此算门会过于宽松 -> 下限锁死 -50
    raw = noise + cfg["rel_margin_db"]
    cfg["abs_floor_dbfs"] = round(max(-50.0, min(-25.0, raw)), 1)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"\n房间噪声底({cfg['noise_pct']:.0f}%分位) = {noise:.1f} dBFS "
          f"(中位 {med:.1f}, 最响 {lvl.max():.1f})")
    if med < -85.0:
        print("  ⚠ 读数低到 -85 dBFS 以下, 这台麦克风很可能自带静音闸/降噪 ——")
        print("    说明它录到的'安静'是假的。标定结果仅供参考, 真正判定靠滚动噪声底自适应。")
    print(f"绝对门 abs_floor_dbfs = {cfg['abs_floor_dbfs']} dBFS (原始算得 {raw:.1f})")
    print(f"已写入 {cfg_path}")


# ---------------- 主流程 ----------------
def process_session(blocks_iter, cfg, outdir, save_clips=True, tag="run",
                    verbose_every=60.0, session_start=None, note=None):
    """blocks_iter: 产出 (block_float32_mono, t_seconds)"""
    os.makedirs(outdir, exist_ok=True)
    clips_dir = os.path.join(outdir, "clips")
    if save_clips:
        os.makedirs(clips_dir, exist_ok=True)
    ev_csv = os.path.join(outdir, f"events_{tag}.csv")
    lv_csv = os.path.join(outdir, f"levels_{tag}.csv")
    t0 = session_start or datetime.now()

    ef = open(ev_csv, "w", encoding="utf-8")
    lf = open(lv_csv, "w", encoding="utf-8")
    ef.write("idx,start_time,duration_s,peak_dbfs,mean_dbfs,low_ratio,centroid_hz,clip\n")
    lf.write("time,dbfs,low_ratio\n")
    ef.flush(); lf.flush()

    rows = []

    def on_event(ev):
        ts = t0 + timedelta(seconds=ev["start_t"])
        fn = ""
        if save_clips:
            fn = f"snore_{len(rows)+1:04d}_{ts.strftime('%H%M%S')}_{ev['duration_s']:.1f}s.wav"
            write_wav(os.path.join(clips_dir, fn), ev["audio"])
        ev["clip"] = fn
        rows.append(ev)
        ef.write("{},{},{:.2f},{:.1f},{:.1f},{:.2f},{:.0f},{}\n".format(
            len(rows), ts.strftime("%Y-%m-%d %H:%M:%S"), ev["duration_s"],
            ev["peak_dbfs"], ev["mean_dbfs"], ev["mean_low_ratio"],
            ev["mean_centroid"], fn))
        ef.flush()
        print(f"  [鼾] #{len(rows)}  {ts.strftime('%H:%M:%S')}  "
              f"时长 {ev['duration_s']:.1f}s  峰值 {ev['peak_dbfs']:.1f} dBFS"
              + (f"  -> {fn}" if fn else ""))

    det = Detector(cfg, on_event=on_event)

    # 每秒一行音量日志
    sec_acc = []
    segs = []          # (t, dbfs, low_ratio) 供出图文报告用
    sec_i = -1
    next_verbose = 0.0
    n_blocks = 0

    for block, t in blocks_iter:
        det.feed(block, t)
        m = block_features(block, cfg["low_band"])
        si = int(t)
        if si != sec_i:
            if sec_acc:
                a = np.array(sec_acc)
                md, ml = float(a[:, 0].mean()), float(a[:, 1].mean())
                segs.append((float(sec_i), md, ml))
                lf.write("{},{:.1f},{:.2f}\n".format(
                    (t0 + timedelta(seconds=sec_i)).strftime("%Y-%m-%d %H:%M:%S"),
                    md, ml))
                lf.flush()
            sec_acc = []
            sec_i = si
        sec_acc.append((m[0], m[1]))

        n_blocks += 1
        if t >= next_verbose:
            next_verbose = t + verbose_every
            print(f"  t={fmt_hms(t)}  当前 {m[0]:6.1f} dBFS | 噪声底 {det.floor.value():6.1f} "
                  f"| 已记 {det.n_events} 次")
    det.flush()
    if sec_acc:
        a = np.array(sec_acc)
        md, ml = float(a[:, 0].mean()), float(a[:, 1].mean())
        segs.append((float(sec_i), md, ml))
        lf.write("{},{:.1f},{:.2f}\n".format(
            (t0 + timedelta(seconds=sec_i)).strftime("%Y-%m-%d %H:%M:%S"), md, ml))
    ef.close(); lf.close()

    # 成串判定: 单次孤立巨响多半是噪声(关门/翻身/键盘), 连成串的才像打鼾
    gap_th = cfg["cluster_gap_s"]
    cl_id, cl_start = 0, 0
    for i, ev in enumerate(rows):
        if i == 0 or (ev["start_t"] - (rows[i-1]["start_t"] + rows[i-1]["duration_s"])) > gap_th:
            cl_id += 1
            cl_start = i
        ev["cluster"] = cl_id
    cl_sizes = {}
    for ev in rows:
        cl_sizes[ev["cluster"]] = cl_sizes.get(ev["cluster"], 0) + 1
    for ev in rows:
        ev["cluster_size"] = cl_sizes[ev["cluster"]]
        ev["likely_snore"] = ev["cluster_size"] >= cfg["cluster_min"]
    # 重写事件表, 补上成串信息
    with open(ev_csv, "w", encoding="utf-8") as f:
        f.write("idx,start_time,duration_s,peak_dbfs,mean_dbfs,low_ratio,centroid_hz,"
                "cluster,cluster_size,likely_snore,clip\n")
        for i, ev in enumerate(rows, 1):
            f.write("{},{},{:.2f},{:.1f},{:.1f},{:.2f},{:.0f},{},{},{},{}\n".format(
                i, (t0 + timedelta(seconds=ev["start_t"])).strftime("%Y-%m-%d %H:%M:%S"),
                ev["duration_s"], ev["peak_dbfs"], ev["mean_dbfs"], ev["mean_low_ratio"],
                ev["mean_centroid"], ev["cluster"], ev["cluster_size"],
                int(ev["likely_snore"]), ev.get("clip", "")))

    total_s = t if n_blocks else 0.0
    likely = [e for e in rows if e["likely_snore"]]
    summary = {
        "start": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "total_recorded_s": round(total_s, 1),
        "n_snore_events": det.n_events,
        "n_likely_snore": len(likely),
        "total_snore_s": round(det.total_snore_s, 1),
        "total_likely_snore_s": round(sum(e["duration_s"] for e in likely), 1),
        "n_clusters": cl_id,
        "snore_fraction": round(det.total_snore_s / total_s, 4) if total_s > 0 else 0.0,
        "peak_dbfs": round(det.peak_dbfs, 1),
        "noise_floor_dbfs": round(det.floor.value(), 1),
        "first_event": rows[0]["start_t"] if rows else None,
        "last_event": rows[-1]["start_t"] if rows else None,
        "cfg": cfg,
    }
    sp = os.path.join(outdir, f"summary_{tag}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    try:
        rp = write_report(outdir, tag, summary, rows, segs, cfg, note=note)
    except Exception as e:
        print(f"  (图文报告生成失败: {e})")
        rp = None
    return summary, rows, ev_csv, lv_csv, rp


def live_blocks(device, max_seconds=None):
    def gen():
        t = 0.0
        with sd.InputStream(device=device, samplerate=SR, channels=1,
                            blocksize=BLOCK_N, dtype="float32") as s:
            while True:
                blk, overflowed = s.read(BLOCK_N)
                t += BLOCK_S
                yield blk[:, 0].copy(), t
                if max_seconds and t >= max_seconds:
                    break
    return gen()


def _find_ffmpeg():
    """优先用 imageio-ffmpeg 自带的现代版; 系统 PATH 上那个可能是 2013 年的老古董。"""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    return shutil.which("ffmpeg")


def ensure_16k_mono_wav(path, tmpdir):
    """把任意音频(手机 m4a/44.1k 立体声/…)弄成一个可流式读的 16k 单声道 WAV。

    返回 (路径, 是否转换过)。用 ffmpeg 转, 内存占用恒定, 8 小时录音也不爆。
    """
    try:
        with wave.open(path, "rb") as w:
            if (w.getframerate() == SR and w.getnchannels() == 1
                    and w.getsampwidth() == 2):
                return path, False
    except Exception:
        pass  # 不是 WAV, 或读不动 -> 交给 ffmpeg

    ff = _find_ffmpeg()
    if not ff:
        raise SystemExit(
            "这个文件不是 16kHz 单声道 WAV, 而系统里找不到 ffmpeg。\n"
            "两个办法: ① 装 ffmpeg; ② 先手动转: ffmpeg -i 输入 -ac 1 -ar 16000 输出.wav")
    out = os.path.join(tmpdir, "converted_16k.wav")
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
           "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", out]
    print(f"  ffmpeg 转码中: {os.path.basename(path)} -> 16kHz 单声道 …")
    subprocess.run(cmd, check=True)
    return out, True


def replay_blocks(path):
    def gen():
        t = 0.0
        with wave.open(path, "rb") as w:
            while True:
                raw = w.readframes(BLOCK_N)
                if not raw:
                    break
                a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if len(a) < BLOCK_N:
                    a = np.pad(a, (0, BLOCK_N - len(a)))
                t += BLOCK_S
                yield a, t
    return gen()


def main():
    ap = argparse.ArgumentParser(description="打鼾检测器 · 电脑端第一阶段")
    ap.add_argument("--device", type=int, default=None, help="输入设备编号(默认系统默认麦克风)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "records"))
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--calibrate", type=int, default=0, metavar="SECONDS", help="标定房间噪声底")
    ap.add_argument("--replay", metavar="WAV", help="离线跑一个 16kHz 单声道 WAV")
    ap.add_argument("--duration", type=float, default=0, metavar="SECONDS",
                    help="只采这么多秒就停(默认 0 = 一直采到 Ctrl+C)")
    ap.add_argument("--warmup", type=float, metavar="SECONDS",
                    help="开场只学噪声底、不判定的秒数(默认配置 20)")
    ap.add_argument("--no-clips", action="store_true", help="不存音频片段")
    ap.add_argument("--no-open", action="store_true", help="跑完不自动打开图文报告")
    ap.add_argument("--low-ratio", type=float)
    ap.add_argument("--rel-margin", type=float)
    ap.add_argument("--abs-floor", type=float)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.low_ratio is not None:
        cfg["low_ratio"] = args.low_ratio
    if args.rel_margin is not None:
        cfg["rel_margin_db"] = args.rel_margin
    if args.abs_floor is not None:
        cfg["abs_floor_dbfs"] = args.abs_floor
    if args.warmup is not None:
        cfg["warmup_s"] = args.warmup

    if args.calibrate:
        calibrate(args.device, args.calibrate, args.config)
        return

    # 设备信息 / 素材准备
    tmpdir = None
    if args.replay:
        print(f"回放素材: {args.replay}")
        tmpdir = tempfile.mkdtemp(prefix="snore_replay_")
        src_path, _ = ensure_16k_mono_wav(args.replay, tmpdir)
    else:
        idx = args.device if args.device is not None else sd.default.device[0]
        dinfo = sd.query_devices(idx)
        print(f"麦克风: [{idx}] {dinfo['name']}")
        print(f"采样率 {SR} Hz / 分块 {int(BLOCK_S*1000)} ms")
    print(f"判定门: 绝对 >{cfg['abs_floor_dbfs']} dBFS, 相对 >噪声底+{cfg['rel_margin_db']} dB, "
          f"低频占比 >{cfg['low_ratio']}")
    print("开始采集。Ctrl+C 结束并出报告。\n" if not args.replay else "开始分析。\n")

    t0 = datetime.now()
    tag = t0.strftime("%Y-%m-%d_%H%M")
    outdir = os.path.join(args.out, tag)

    src = replay_blocks(src_path) if args.replay else live_blocks(
        args.device, max_seconds=args.duration or None)
    try:
        summary, rows, ev_csv, lv_csv, rp = process_session(
            src, cfg, outdir, save_clips=not args.no_clips, tag=tag, session_start=t0)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C, 整理报告…")
        return
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 56)
    print(f"当晚汇总  ({summary['start']} 起)")
    print(f"  录音时长      {fmt_hms(summary['total_recorded_s'])}")
    print(f"  疑似鼾声事件  {summary['n_snore_events']} 次 "
          f"(其中成串的 {summary['n_likely_snore']} 次, 共 {summary['n_clusters']} 串)")
    print(f"  成串鼾声时长  {fmt_hms(summary['total_likely_snore_s'])} "
          f"(占 {summary['total_likely_snore_s']/summary['total_recorded_s']*100:.1f}%)"
          if summary["total_recorded_s"] else "")
    print(f"  最响峰值      {summary['peak_dbfs']} dBFS")
    if rows:
        print(f"  首次/末次     第 {fmt_hms(rows[0]['start_t'])} / 第 {fmt_hms(rows[-1]['start_t'])}")
    print(f"\n  片段目录      {os.path.join(outdir, 'clips') if not args.no_clips else '(未存)'}")
    print(f"  事件表        {ev_csv}")
    print(f"  音量日志      {lv_csv}")
    if rp:
        print(f"  图文报告      {rp}")
    print("=" * 56)
    if rp and not args.no_open:
        try:
            webbrowser.open("file:///" + rp.replace("\\", "/"))
            print("\n已用默认浏览器打开图文报告(想存 PDF: Ctrl+P → 另存为 PDF)。")
        except Exception:
            pass


if __name__ == "__main__":
    main()
