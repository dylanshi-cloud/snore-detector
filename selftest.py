#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自测: 用合成音频验证检测逻辑, 不碰麦克风。

  场景 A  静音+低频鼾声+同响度高频噪声 -> 抓得到鼾声, 高频不误报
  场景 B  三次鼾声相隔 3s            -> 成一串, likely_snore=True
  场景 C  静音里一次孤立低频巨响      -> 检出但不算"真打鼾"(cluster_size=1)
"""
import os
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snore_detect as sd

SR = sd.SR
RESULT = []


def set_rms(x, dbfs):
    cur = np.sqrt(np.mean(x ** 2)) + 1e-12
    return x * (10 ** (dbfs / 20.0) / cur)


def quiet(dur, dbfs=-70.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return set_rms(rng.standard_normal(int(dur * SR)) * 0.3, dbfs)


def snore(dur, dbfs=-30.0):
    """低频基频+谐波, 带 25Hz flutter 调幅, 模拟鼾声。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 110 * t)
    for k, a in [(2, 0.6), (3, 0.4), (4, 0.25), (5, 0.15), (6, 0.1)]:
        sig += a * np.sin(2 * np.pi * 110 * k * t)
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 25 * t)
    return set_rms(sig, dbfs)


def hifreq(dur, f0=2000.0, dbfs=-30.0):
    """同响度的高频纯音 —— 必须不被当成鼾声。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    return set_rms(np.sin(2 * np.pi * f0 * t), dbfs)


def write_wav(path, audio):
    pcm = np.clip(np.concatenate(audio), -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def run_case(name, parts):
    tmp = tempfile.mkdtemp(prefix=f"snore_{name}_")
    wav = os.path.join(tmp, f"{name}.wav")
    write_wav(wav, parts)
    cfg = dict(sd.DEFAULT_CFG)
    cfg["warmup_s"] = 0.0          # 合成音频短, 关掉学习期
    cfg["cluster_gap_s"] = 5.0
    summary, rows, _, _, _ = sd.process_session(
        sd.replay_blocks(wav), cfg, tmp, save_clips=True, tag=name,
        verbose_every=1e9)
    return summary, rows, tmp


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    RESULT.append(bool(cond))


def case_a():
    print("\n[场景 A] 静音 + 低频鼾声 + 同响度高频噪声")
    parts = [quiet(3.0), snore(1.0), quiet(1.0), snore(1.0),
             quiet(3.0), hifreq(1.5), quiet(3.0)]
    s, rows, tmp = run_case("A", parts)
    for i, r in enumerate(rows, 1):
        print(f"    #{i} @ {r['start_t']:5.1f}s  {r['duration_s']:.1f}s  "
              f"{r['peak_dbfs']:.1f}dBFS  低频{r['mean_low_ratio']:.2f}")
    check(s["n_snore_events"] == 2, f"检出 2 次鼾声 (实际 {s['n_snore_events']})")
    if rows:
        check(abs(rows[0]["start_t"] - 3.0) < 1.0, f"第 1 次约 3.0s (实际 {rows[0]['start_t']:.1f}s)")
    if len(rows) > 1:
        check(abs(rows[1]["start_t"] - 5.0) < 1.0, f"第 2 次约 5.0s (实际 {rows[1]['start_t']:.1f}s)")
    bad = [r for r in rows if 7.0 <= r["start_t"] <= 10.5]
    check(not bad, f"2kHz 强音段不应误报 (实际 {len(bad)} 条)")
    check(len(os.listdir(os.path.join(tmp, "clips"))) == 2, "存了 2 个片段")


def case_b():
    print("\n[场景 B] 三声鼾相隔 3s —— 应成串")
    parts = [quiet(3.0)]
    for _ in range(3):
        parts += [snore(1.0), quiet(3.0)]
    s, rows, tmp = run_case("B", parts)
    for r in rows:
        print(f"    @{r['start_t']:5.1f}s  串#{r['cluster']} 串长{r['cluster_size']} "
              f"likely={r['likely_snore']}")
    check(s["n_snore_events"] == 3, f"检出 3 次 (实际 {s['n_snore_events']})")
    check(s["n_clusters"] == 1, f"归为 1 串 (实际 {s['n_clusters']})")
    check(s["n_likely_snore"] == 3, f"3 次都标为真打鼾 (实际 {s['n_likely_snore']})")


def case_c():
    print("\n[场景 C] 静音里一次孤立低频巨响 —— 不应算真打鼾")
    parts = [quiet(5.0), snore(0.6, dbfs=-28.0), quiet(6.0)]
    s, rows, tmp = run_case("C", parts)
    for r in rows:
        print(f"    @{r['start_t']:5.1f}s  串长{r['cluster_size']} likely={r['likely_snore']}")
    check(s["n_snore_events"] == 1, f"检出 1 次 (实际 {s['n_snore_events']})")
    check(s["n_likely_snore"] == 0, f"不算真打鼾 (实际 {s['n_likely_snore']})")


if __name__ == "__main__":
    case_a(); case_b(); case_c()
    print("\n" + ("全部通过 ✅" if all(RESULT) else f"有失败 ❌ ({RESULT.count(False)} 项)"))
    sys.exit(0 if all(RESULT) else 1)
