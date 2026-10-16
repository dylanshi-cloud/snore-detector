# 打鼾检测器 · snore-detector

> 用一台普通电脑（或一部手机录音）判断你夜里**有没有打鼾、什么时候打、打得凶不凶**。
> 纯 Python，判定逻辑完全可解释 —— 每一次判定都存下音频片段，供你第二天人耳复核。

![图标](snore_preview.png)

## 它能干什么

- **整夜无人值守录音**，或**离线分析手机录的整晚录音**（`--replay`）
- 判定「疑似鼾声事件」，并把间隔很近的**连成一串** —— 只有成串的才算真打鼾（一次孤立的巨响多半是翻身 / 关门）
- 每次事件**存一个音频片段**，早上点开就能听，自己判断它报得准不准
- 跑完生成**一份自包含的图文 HTML 报告**：整夜时间轴 / 每小时鼾声时长 / 整夜音量曲线 / 逐次明细（可当场点播）/ 自动结论。想存 PDF 按 `Ctrl+P`

## 判定原理（可解释，非黑盒）

每 0.1 秒取一块音频，算三个量，**全部满足**才算「疑似鼾声」：

| 条件 | 含义 | 默认 |
|---|---|---|
| 绝对音量 | 比 `abs_floor_dbfs` 响 | > -45 dBFS |
| 相对音量 | 比**滚动噪声底**高 `rel_margin_db` | > 噪声底 + 12 dB |
| 低频占比 | 40–400 Hz 能量占比 | > 0.55 |

再加一层**归串**：间隔 ≤ `cluster_gap_s` 的事件算同一串，一串里 ≥ `cluster_min` 次才标记为「真打鼾」（`likely_snore=1`）。
开场有 `warmup_s` 秒**只学噪声底、不判定**，避免冷启动误报。

## 安装

需要 Python 3.9+。

```bash
pip install numpy sounddevice
# 仅当要用 --replay 分析 m4a/mp3 等压缩格式时：
pip install imageio-ffmpeg
```

## 使用

```bash
python snore_detect.py                     # 采一整夜（Ctrl+C 结束并出报告）
python snore_detect.py --duration 60       # 只采 60 秒试试
python snore_detect.py --calibrate 60      # 标定房间噪声底（睡前做）
python snore_detect.py --replay 昨晚.m4a    # 离线分析手机录音（自动转码）
python snore_detect.py --warmup 3          # 临时改学习期
python snore_detect.py --no-clips          # 不存音频片段
python snore_detect.py --no-open           # 跑完不自动打开报告
python snore_detect.py --help
```

Windows 用户：双击 `run.bat` 即开始采集；把录音文件拖到 `分析录音.bat` 上即开始分析。

自测（纯合成音频，不需要麦克风）：

```bash
python selftest.py
```

## 输出

```
records/<日期_时刻>/
    report_*.html    # 图文报告（浏览器打开，片段可直接播）
    clips/           # 每次事件的音频片段
    events_*.csv     # 每次一行：时间/时长/峰值/低频占比/是否成串
    levels_*.csv     # 每秒音量，用来看整晚趋势、排查「没录到」
    summary_*.json   # 汇总
```

## 参数（`config.json`）

| 参数 | 含义 | 默认 |
|---|---|---|
| `abs_floor_dbfs` | 绝对音量门，低于它一律不算 | -45 |
| `rel_margin_db` | 要比噪声底高多少 dB 才算响 | 12 |
| `low_ratio` | 低频(40–400Hz)能量占比门 | 0.55 |
| `noise_window_s` / `noise_pct` | 滚动噪声底的窗口 / 取第几百分位 | 120 / 15 |
| `warmup_s` | 开场只学噪声底、不判定的秒数 | 20 |
| `cluster_gap_s` | 间隔小于它算同一串 | 5 |
| `cluster_min` | 一串里至少几次才算真打鼾 | 3 |

## 已知坑（重要）

1. **内置阵列麦克风的「音频增强 / 降噪」会吃掉鼾声。**
   某些笔记本（尤其带 Intel 智音技术的阵列麦）的降噪会滤掉稳定低频的非人声信号 —— 结果就是**你说话它听得到，但鼾声频段被静默丢弃**，整夜读数贴在 -90 dBFS 上下。
   排查：设置 → 系统 → 声音 → 输入 → 麦克风 → **音频增强 = 关**；再看 `levels_*.csv` 是不是还是一条死线。
   实在收不到，就改走**手机录音 + `--replay`**，这条最稳。
2. **`--replay` 转码依赖新版 ffmpeg。** 代码优先用 `imageio-ffmpeg` 自带的 7.x；若系统 PATH 上是很老的版本会报错。
3. **电脑别睡眠。** 设置 → 系统 → 电源 → 接通电源时「永不睡眠」；屏幕可以关，睡眠不行。

## 免责声明

个人自用的小工具，**不是医疗器械**，不能用于诊断或替代就医。判定结果仅供参考，请结合音频片段人工复核。

## License

[MIT](LICENSE)
