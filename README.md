# 磨底选股 · 60日线第三次磨底 daily A-share screen

每天新加坡时间早上 6 点，GitHub Actions 下载最新的社区 A 股 Qlib 数据
（[chenditc/investment_data](https://github.com/chenditc/investment_data)），用 `ma60.py` 的规则检查每只沪深股票，
并通过 GitHub Pages 发布结果页面：https://kychanbp.github.io/modi-xuangu/ 。每天的结果保存在 `history/<date>.csv`。
另见 [波浪选股](https://kychanbp.github.io/bolang-xuangu/)。

Every day at 06:00 Singapore time a GitHub Actions job downloads the latest community A-share Qlib bundle, checks every
Shanghai/Shenzhen stock against the "60-day MA pullback, 3rd bottom" rules in `ma60.py` (monthly bar size, first weekly
wave, a volume breakout above MA60, a low-volume fall back below it, shrinking down waves, a hill between bottoms, and
4+ low-volume down candles then an up candle at the 3rd bottom below MA60), and publishes the result page via GitHub
Pages. It also lists "watch" stocks that need only the up candle. Daily results are kept in `history/<date>.csv`.

A second tier, 宽松版 (relaxed, `RULES_R`), keeps the same structure with looser rules: no weekly rule and no D2;
monthly bodies 1.2x; the 2nd or 3rd bottom; 3+ low-volume down candles; a hill of 4+ up candles or +15%; breakout volume
1.5x. The page lists strict signals first, then relaxed signals, then the watch list for both tiers. The `tier` column in
the history CSV says which tier a row is from, and a stock in the strict list is not repeated in the relaxed list.

Run locally: `python ma60.py --qlib <qlib_data dir> --out site` (`--asof YYYY-MM-DD` screens a past day).

Research only, not investment advice. Backtest 2008–2026 (about 350 signals): with the sell-at-+10% / add-at-−5% rule,
each trade made about 0.9 pp more than random A-shares bought the same day (t ≈ 2.2 over 20 random draws). The result is
not stable: with fixed holding periods or small rule changes the edge goes away.

Relaxed tier backtest (2009 to 2026-10, about 12,900 signals, about 690 a year, 20 random draws): with the +10% rule each
trade made +0.1 pp more than random (t ≈ 0.4, no edge). 5-day and 20-day holds made +0.4 and +0.7 pp (t ≈ 2.8 and 2.9).
The 10-slot book made 6.5% a year against a random median of 6.6% and 5.3% for CSI 1000.
