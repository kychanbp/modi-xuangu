#!/usr/bin/env python3
"""Daily 60日线第三次磨底 screen for China A-shares (v2 rules of the ma60-pullback research).

Reads the community Qlib A-share bundle (github.com/chenditc/investment_data), checks every
Shanghai/Shenzhen stock on the latest trading day, and writes <out>/index.html (+ charts)
and <history>/<date>.csv.

Rules (each stock on its own trading days; defaults in RULES):
 Stretch: the days since the last close above MA60; the up candle (today) closes at or below MA60
    and some close in the stretch is below MA60.
 D1: within 250 days before today and before the stretch, the latest close crossing from <= MA60 to
    > MA60 on volume >= 2x the mean of the 20 days before.
 D2: from the highest close after D1 to the first day of the stretch, the mean volume is lower than
    from D1 to that high.
 Bottoms: swing lows (lowest low of 5 bars each side) in the stretch; the signal bottom (lowest low of the
    down run and today) is lower than the 5 lows before it. Today is the 3rd bottom of the stretch.
 D3: each down wave (after the highest high before a bottom, through the bottom) has a lower mean volume
    than the wave before; 1 exception allowed.
 Hill: between some two consecutive bottoms, >= 4 up candles in a row or a rise >= 20% from the earlier
    bottom's low.
 Candle: >= 4 down candles in a row, each with volume below the mean volume of the hill before the run
    (days after the previous bottom up to the highest high before the run); today an up candle.
 M2 (monthly, completed months, last 36): in the up leg (lowest low before the 36-month high, to the high),
    mean body of up months >= 1.5x mean body of down months.
 W1 (weekly, completed weeks): first wave from the up-leg low week (8-20 weeks, to its highest high):
    up-week bodies >= 2x down-week bodies, and >= 70% of down weeks inside the prior up week's range.
 Only the first signal of a stretch counts. "Watch": today is the 4th+ down candle of a run that meets
    every rule; it becomes a signal if the next candle is an up candle that closes at or below MA60.
Tier E (例子版, RULES_E), rebuilt from the author's examples: close >= 30% below the 250-day high; >= 30 of the last
    60 closes below MA60; mean volume of the 5 days before today <= 20% of the highest 20-day mean volume of the last
    250 days and in the lowest 15% of its last 120 values; >= 2 down candles in a row before today that made the
    20-day low; today an up candle closing at or below MA60 (not limit-up). Only the first signal in 20 days counts.
    A stock in the strict list is not repeated in tier E.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RULES = dict(d1_lookback=250, d1_vol=2.0, pivot=5, run_min=4, bottom=3, d3_exc=1, hill_uprun=4, hill_rise=0.20,
             m_window=36, m_min_months=12, m_body=1.5, w_min=8, w_max=20, w_body=2.0, w_inside=0.70, min_listed=250,
             bottoms=(3,), need_d2=True, need_w1=True)
# 例子版 (example tier, 2026-10-31): rebuilt from the author's 10 examples; the key is very low volume before the buy
RULES_E = dict(dd=-0.30, below_win=60, below_min=30, vb_days=5, peak_win=250, peak_avg=20, q_peak=0.20,
               q_win=120, q_pct=0.15, run_min=2, low_win=20, gap=20, min_listed=250)
WINDOW = 1000         # trading days loaded per stock (36 months of monthly bars + MA60 + D1 look-back)
FIELDS = ["open", "high", "low", "close", "volume", "factor"]


def load(qdir: Path, asof=None):
    cal = pd.to_datetime(open(qdir / "calendars" / "day.txt").read().split())
    inst = pd.read_csv(qdir / "instruments" / "all.txt", sep="\t", header=None, names=["code", "a", "b"])
    inst = inst[inst.code.str[:2].isin(["SH", "SZ"]) & ~inst.code.str.match(r"SH000|SZ399")]
    last = pd.Timestamp(asof) if asof else pd.Timestamp(inst.b.max())
    T1 = int(cal.get_loc(last)) + 1
    T0 = max(T1 - WINDOW, 0)
    dates = cal[T0:T1]
    W = T1 - T0
    codes, arr, first = [], {f: [] for f in FIELDS}, []
    for c in inst.code:
        d = qdir / "features" / c.lower()
        if not (d / "close.day.bin").exists():
            continue
        cols = {}
        for f in FIELDS:
            a = np.fromfile(d / f"{f}.day.bin", dtype="<f4")
            s = int(a[0])
            out = np.full(W, np.nan, dtype=np.float32)
            lo, hi = max(s, T0), min(s + len(a) - 1, T1)
            if hi > lo:
                out[lo - T0:hi - T0] = a[1 + lo - s:1 + hi - s]
            cols[f] = out
            if f == "close":
                first.append(s - T0)
        codes.append(c)
        for f in FIELDS:
            arr[f].append(cols[f])
    p = {f: np.stack(v, axis=1) for f, v in arr.items()}
    trade = np.isfinite(p["close"]) & (p["volume"] > 0)
    for f in ("open", "high", "low", "close", "volume"):
        p[f] = np.where(trade, p[f], np.nan)
    p["dates"], p["codes"], p["first"] = dates, np.array(codes), np.array(first)
    return p


# ---------------------------------------------------------------- monthly / weekly
def bars(idx, o, h, l, c, v, key):
    ids, start = np.unique(key, return_index=True)
    end = np.r_[start[1:], len(key)]
    out = dict(id=ids, o=o[start], c=c[end - 1], h=np.maximum.reduceat(h, start), l=np.minimum.reduceat(l, start))
    out["lowday"] = np.array([idx[a + int(np.argmin(l[a:b]))] for a, b in zip(start, end)])
    out["day0"] = idx[start]
    return out


def m2_w1(mb, wb, mid, wid, q=RULES):
    """M2 and W1 for a signal in month id `mid` / week id `wid` (completed bars only)."""
    e = int(np.searchsorted(mb["id"], mid))
    b = max(int(np.searchsorted(mb["id"], mid - q["m_window"])), 0)
    if e - b < q["m_min_months"]:
        return False, False, {}
    hm = b + int(np.argmax(mb["h"][b:e]))
    lm = b + int(np.argmin(mb["l"][b:hm + 1]))
    oo, cc = mb["o"][lm:hm + 1], mb["c"][lm:hm + 1]
    up, dn = cc > oo, cc < oo
    body = np.abs(cc - oo) / oo
    m2 = bool(not dn.any() or (up.any() and body[up].mean() >= q["m_body"] * body[dn].mean()))
    info = dict(m2_ratio=float(body[up].mean() / body[dn].mean()) if up.any() and dn.any() else float("inf"),
                m_up=int(up.sum()), m_dn=int(dn.sum()), m_from=int(mb["id"][lm]), m_to=int(mb["id"][hm]))
    ws = int(np.searchsorted(wb["day0"], int(mb["lowday"][lm]), side="right")) - 1
    last = min(ws + q["w_max"] - 1, int(np.searchsorted(wb["id"], wid)) - 1)
    if ws < 0 or last < ws:
        return m2, False, info
    we = max(ws + int(np.argmax(wb["h"][ws:last + 1])), min(ws + q["w_min"] - 1, last))
    o, c, h, l = (wb[x][ws:we + 1] for x in ("o", "c", "h", "l"))
    up, dn = c > o, c < o
    body = np.abs(c - o)
    ins, nd, lastup = 0, 0, -1
    for i in range(len(c)):
        if dn[i]:
            nd += 1
            if lastup >= 0 and max(o[i], c[i]) <= h[lastup] and min(o[i], c[i]) >= l[lastup]:
                ins += 1
        elif up[i]:
            lastup = i
    w1 = bool(body[up].sum() >= q["w_body"] * body[dn].sum() and (nd == 0 or ins / nd >= q["w_inside"]))
    info.update(w1_ratio=float(body[up].sum() / body[dn].sum()) if dn.any() else float("inf"),
                w1_inside=ins / nd if nd else 1.0, w_weeks=int(we - ws + 1))
    return m2, w1, info


# ---------------------------------------------------------------- daily rules
def daily(o, h, lw, c, v, ma, isd1, piv, down, s, t, q=RULES, watch=False):
    """Daily facts for a candle day t in the stretch starting at s (indices on the stock's trading days).
    watch=True: t is the last down candle of the run and the up candle has not come yet (virtual t + 1)."""
    if watch:
        L = 0
        while t - L >= 0 and down[t - L]:
            L += 1
        r0, tt, lo_end = t - L + 1, t + 1, t        # run r0..t, virtual up candle at t + 1
    else:
        L = 0
        while t - 1 - L >= 0 and down[t - 1 - L]:
            L += 1
        r0, tt, lo_end = t - L, t, t
    if L < q["run_min"]:
        return None
    pv = q["pivot"]
    bs = r0
    for x in range(r0, lo_end + 1):
        if lw[x] < lw[bs]:
            bs = x
    if bs < pv or bs < s or not (lw[bs - pv:bs] > lw[bs]).all():
        return None
    d1 = -1
    for x in range(s - 1, max(tt - q["d1_lookback"], 0) - 1, -1):
        if isd1[x]:
            d1 = x
            break
    d2, d2_after, d2_before = False, np.nan, np.nan
    if d1 >= 0:
        pk = d1
        for x in range(d1, s):
            if c[x] > c[pk]:
                pk = x
        d2_after, d2_before = v[pk + 1:s + 1].mean(), v[d1:pk + 1].mean()
        d2 = d2_after < d2_before
    bl = [x for x in range(s, bs) if piv[x] and x + pv <= (t if watch else tt)] + [bs]
    k = len(bl)
    viol, prev, st, waves = 0, -1.0, s, []
    for i, b in enumerate(bl):
        pk = st
        for x in range(st, b):
            if h[x] > h[pk]:
                pk = x
        mv = v[pk + 1:b + 1].mean() if b > pk else v[b]
        if i > 0 and not (mv < prev):
            viol += 1
        waves.append(float(mv))
        prev, st = mv, b + 1
    b_prev = bl[-2] if k > 1 else s
    pk = b_prev
    for x in range(b_prev, r0):
        if h[x] > h[pk]:
            pk = x
    hv = v[b_prev + 1:pk + 1].mean() if pk > b_prev else v[r0 - 1]
    vhill = bool((v[r0:r0 + L] < hv).all())
    nu, gaps = 0, []
    for i in range(1, k):
        b0, b1 = bl[i - 1], bl[i]
        gh = max(h[b0], h[b0 + 1:b1 + 1].max())
        urun = best = 0
        for x in range(b0 + 1, b1 + 1):
            urun = urun + 1 if c[x] > o[x] else 0
            best = max(best, urun)
        ok = best >= q["hill_uprun"] or gh / lw[b0] - 1 >= q["hill_rise"]
        if ok:
            nu += 1
        gaps.append(dict(b0=b0, b1=b1, uprun=int(best), rise=float(gh / lw[b0] - 1), ok=bool(ok)))
    return dict(L=L, bs=bs, d1=d1, d2=bool(d2), k=k, bottoms=bl, d3viol=viol, vhill=vhill, hill=nu >= 1,
                r0=r0, d2_ratio=float(d2_after / d2_before) if d1 >= 0 else np.nan, waves=waves, gaps=gaps,
                run_vol_ratio=float(v[r0:r0 + L].max() / hv), hill_b=b_prev, hill_pk=pk)


def daily_ok(r, q=RULES):
    return (r is not None and r["d1"] >= 0 and (r["d2"] or not q["need_d2"]) and r["k"] in q["bottoms"] and r["vhill"]
            and r["hill"] and r["d3viol"] <= q["d3_exc"])


def check(j, p, mkey, wkey, q=RULES):
    """Signal or watch row for stock j on the last panel day, or None."""
    T = len(p["dates"]) - 1
    cl = p["close"][:, j]
    idx = np.flatnonzero(np.isfinite(cl))
    m = len(idx)
    if m < 130 or idx[-1] != T or T - p["first"][j] < q["min_listed"]:
        return None
    o, h, lw, c, v = (p[f][idx, j].astype(float) for f in ("open", "high", "low", "close", "volume"))
    ma = pd.Series(c).rolling(60).mean().values
    va = pd.Series(v).rolling(20).mean().shift(1).values
    isd1 = np.zeros(m, bool)
    isd1[60:] = (c[60:] > ma[60:]) & (c[59:-1] <= ma[59:-1]) & (v[60:] >= q["d1_vol"] * va[60:])
    pv = q["pivot"]
    piv = np.zeros(m, bool)
    for i in range(pv, m - pv):
        piv[i] = (lw[i - pv:i] > lw[i]).all() and (lw[i + 1:i + pv + 1] >= lw[i]).all()
    down = c < o
    t = m - 1
    if not c[t] <= ma[t]:
        return None
    s = t
    while s - 1 >= 60 and c[s - 1] <= ma[s - 1]:
        s -= 1
    below = np.cumsum(c[s:] < ma[s:]) > 0          # some close < MA60 in [s, x]
    mw = {}

    def flags(x):
        if x not in mw:
            m2, w1, info = m2_w1(mb, wb, mkey[idx[min(x, m - 1)]], wkey[idx[min(x, m - 1)]], q)
            mw[x] = (m2 and (w1 or not q["need_w1"]), dict(info, m2=m2, w1=w1))
        return mw[x][0]

    mb = bars(idx, o, h, lw, c, v, mkey[idx])
    wb = bars(idx, o, h, lw, c, v, wkey[idx])
    # every earlier candle day of this stretch: only the first signal of a stretch counts
    for x in range(s, t + 1):
        if not below[x - s] or idx[x] - p["first"][j] < q["min_listed"] or not c[x] > o[x]:
            continue
        r = daily(o, h, lw, c, v, ma, isd1, piv, down, s, x, q)
        if daily_ok(r, q) and flags(x):
            if x < t:
                return None
            kind = "signal"
            break
    else:
        if not down[t] or not below[-1]:
            return None
        r = daily(o, h, lw, c, v, ma, isd1, piv, down, s, t, q, watch=True)
        if not (daily_ok(r, q) and flags(t)):
            return None
        kind = "watch"
    D = p["dates"]
    f = float(p["factor"][T, j])
    r = dict(r, d1_vol_ratio=float(v[r["d1"]] / va[r["d1"]]), mw=mw[t][1])
    return dict(code=str(p["codes"][j]), kind=kind, j=j, idx=idx, ma=ma, s=s, t=t, r=r, q=q,
                close_px=c[t] / f, ma60_px=ma[t] / f, below_ma60=c[t] / ma[t] - 1,
                stretch_start=str(D[idx[s]].date()), d1_date=str(D[idx[r["d1"]]].date()),
                bottoms=";".join(str(D[idx[b]].date()) for b in r["bottoms"]), down_candles=r["L"], d3viol=r["d3viol"])


def ex_feats(o, h, lw, c, v, q=RULES_E):
    """Example-tier facts on every day of one stock's trading days (value for buying at that day's close)."""
    C, V, L = pd.Series(c), pd.Series(v), pd.Series(lw)
    ma = C.rolling(60).mean()
    f = pd.DataFrame(dict(ma=ma))
    f["dd"] = C / pd.Series(h).rolling(q["peak_win"], min_periods=200).max() - 1
    f["below"] = (C < ma).astype(float).rolling(q["below_win"]).sum().shift(1)
    vb = V.rolling(q["vb_days"]).mean().shift(1)                    # the 5 days before today
    peak = V.rolling(q["peak_avg"]).mean().rolling(q["peak_win"], min_periods=200).max()
    f["vb"], f["peak"] = vb, peak
    f["q_peak"] = vb / peak
    f["vq"] = vb.rolling(q["q_win"], min_periods=100).quantile(q["q_pct"])
    f["q_rank"] = vb.rolling(q["q_win"], min_periods=100).rank(pct=True)
    down = (C < pd.Series(o)).values
    run = np.zeros(len(c), int)
    for i in range(1, len(c)):
        run[i] = run[i - 1] + 1 if down[i - 1] else 0                # down candles in a row before today
    f["run"] = run
    f["newlow"] = L.rolling(q["vb_days"]).min().shift(1) <= L.rolling(q["low_win"]).min().shift(1) * 1.0001
    f["up"] = c > o
    return f


def ex_ok(f, x, q=RULES_E, watch=False):
    r = f.iloc[x]
    base = (r.dd <= q["dd"] and r.below >= q["below_min"] and r.q_peak <= q["q_peak"] and r.vb <= r.vq
            and r.run >= q["run_min"] and bool(r.newlow))
    return bool(base if watch else base and r.up and c_le_ma(f, x))


def c_le_ma(f, x):
    return bool(f["c"].iloc[x] <= f["ma"].iloc[x])


def check_ex(j, p, q=RULES_E):
    """例子版 signal or watch row for stock j on the last panel day, or None."""
    T = len(p["dates"]) - 1
    cl = p["close"][:, j]
    idx = np.flatnonzero(np.isfinite(cl))
    m = len(idx)
    if m < 300 or idx[-1] != T or T - p["first"][j] < q["min_listed"]:
        return None
    o, h, lw, c, v = (p[k][idx, j].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = ex_feats(o, h, lw, c, v, q)
    f["c"] = c
    t = m - 1
    chg = c[t] / c[t - 1] - 1
    lim = 0.195 if str(p["codes"][j]).startswith(("SZ300", "SZ301", "SH688", "SH689")) else 0.095
    if ex_ok(f, t, q) and not (chg >= lim and c[t] >= h[t] * 0.999):
        if any(ex_ok(f, x, q) for x in range(max(t - q["gap"], 0), t)):
            return None
        kind = "signal"
    else:
        # watch: if tomorrow is an up candle at or below MA60, the rules hold (features measured through today)
        if not (c[t] < o[t] and c[t] <= f["ma"].iloc[t]):
            return None
        g = ex_feats(np.r_[o, c[t]], np.r_[h, c[t]], np.r_[lw, c[t]], np.r_[c, c[t]], np.r_[v, v[t]], q)
        g["c"] = np.r_[c, c[t]]
        if not ex_ok(g, t + 1, q, watch=True) or any(ex_ok(f, x, q) for x in range(max(t - q["gap"], 0), t + 1)):
            return None
        f = g
        kind = "watch"
    D, fac = p["dates"], float(p["factor"][T, j])
    k = t if kind == "signal" else t + 1
    rr = f.iloc[k]
    run = int(rr.run)
    return dict(code=str(p["codes"][j]), kind=kind, tier="example", j=j, idx=idx, ma=f["ma"].values[:m], t=t, q=q,
                close_px=c[t] / fac, ma60_px=f["ma"].iloc[t] / fac, below_ma60=c[t] / f["ma"].iloc[t] - 1,
                dd=float(rr.dd), below=int(rr.below), q_peak=float(rr.q_peak), q_rank=float(rr.q_rank), run=run,
                vb=float(rr.vb), peak=float(rr.peak), run0=k - run, stretch_start="", d1_date="", bottoms="",
                down_candles=run, d3viol="")


# ---------------------------------------------------------------- output
def chart(r, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9})
    j, idx, ma, rr = r["j"], r["idx"], r["ma"], r["r"]
    o, h, l, c, v = (p[k][idx, j] for k in ("open", "high", "low", "close", "volume"))
    # window: the stretch below MA60 plus 60 days before it (and the D1 day when it is near)
    a, b = max(min(rr["d1"] - 10, r["s"] - 60) if rr["d1"] >= r["s"] - 90 else r["s"] - 60, 0), r["t"]
    x = np.arange(b - a + 1)
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i in range(a, b + 1):
        col = "#d63031" if c[i] >= o[i] else "#20a162"
        ax.vlines(i - a, l[i], h[i], color=col, lw=0.6)
        ax.add_patch(plt.Rectangle((i - a - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), (h[i] - l[i]) * 0.02 + 1e-9), color=col))
        axv.bar(i - a, v[i], color=col, width=0.7)
    ax.plot(x, ma[a:b + 1], color="#0984e3", lw=1.1, label="MA60")
    ymin, ymax = np.nanmin(l[a:b + 1]), np.nanmax(h[a:b + 1])
    if rr["d1"] >= a:
        ax.axvline(rr["d1"] - a, color="#e67e22", lw=0.7, ls="--")
        ax.text(rr["d1"] - a + 0.5, ymax * 1.06, "放量站上60日线", color="#e67e22", fontsize=8, va="top")
    else:
        ax.text(0, ymax * 1.06, f"← 放量站上60日线 {r['d1_date']}（图外）", color="#e67e22", fontsize=8, va="top")
    ax.axvspan(r["s"] - a - 0.5, b - a + 0.5, color="#0984e3", alpha=0.05)
    for n, bt in enumerate(rr["bottoms"], 1):
        ax.annotate(f"底{n}", (bt - a, l[bt]), (bt - a, l[bt] - (ymax - ymin) * 0.06), ha="center", fontsize=8,
                    color="#6c5ce7", arrowprops=dict(arrowstyle="-", color="#6c5ce7", lw=0.6))
    for g in rr["gaps"]:
        if g["ok"]:
            ax.axvspan(g["b0"] - a - 0.5, g["b1"] - a + 0.5, color="#e17055", alpha=0.08)
            ax.text((g["b0"] + g["b1"]) / 2 - a, ymax * 1.0, f"小山丘\n{g['uprun']}连阳 / +{g['rise']*100:.0f}%",
                    ha="center", va="top", fontsize=7.5, color="#e17055")
    r0, L = rr["r0"], rr["L"]
    ax.axvspan(r0 - a - 0.5, r0 + L - 1 - a + 0.5, color="#20a162", alpha=0.12)
    axv.axvspan(r0 - a - 0.5, r0 + L - 1 - a + 0.5, color="#20a162", alpha=0.12)
    if rr["hill_pk"] > rr["hill_b"]:
        axv.hlines(v[rr["hill_b"] + 1:rr["hill_pk"] + 1].mean(), rr["hill_b"] + 1 - a, r0 + L - 1 - a, color="#e17055",
                   lw=0.8, ls="--")
    axv.text(r0 - a - 1, axv.get_ylim()[1] * 0.95, f"{L}根缩量阴线", fontsize=7.5, color="#20a162", ha="right", va="top")
    if r["kind"] == "signal":
        ax.annotate("买点", (b - a, l[b]), (b - a, l[b] - (ymax - ymin) * 0.09), ha="center", fontsize=8,
                    color="#d63031", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#d63031", lw=0.8))
    ax.set_ylim(ymin - (ymax - ymin) * 0.14, ymax * 1.08)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ticks = list(range(0, b - a + 1, max((b - a) // 6, 1)))
    axv.set_xticks(ticks, [p["dates"][idx[a + t]].strftime("%Y-%m-%d") for t in ticks])
    axv.set_yticks([])
    for s in (ax, axv):
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def criteria(r):
    """(class, mark, text) for each rule: how the stock met it, with the rule's threshold."""
    rr, q, mw = r["r"], r["q"], r["r"]["mw"]
    ok = lambda b: ("ok", "✓") if b else ("no", "✗")
    skip = ("skip", "–")
    x = lambda v: "∞" if not np.isfinite(v) else f"{v:.1f}"
    out = [(*ok(mw.get("m2", False)), f"月线：上涨段阳线月平均实体 = 阴线月的 {x(mw.get('m2_ratio', np.nan))} 倍"
            f"（要求 ≥ {q['m_body']}）")]
    w1 = (f"周线：第一波 {mw.get('w_weeks', '?')} 周，阳线周实体合计 = 阴线周的 {x(mw.get('w1_ratio', np.nan))} 倍"
          f"（要求 ≥ {q['w_body']:.0f}）；{mw.get('w1_inside', 0)*100:.0f}% 阴线周在前一根阳线周范围内（要求 ≥ {q['w_inside']*100:.0f}%）")
    out.append((*ok(mw.get("w1", False)), w1) if q["need_w1"] else (*skip, w1 + "；宽松版不检查"))
    out.append((*ok(True), f"放量站上 60 日线：{r['d1_date']}，成交量 = 前 20 日均量的 {rr['d1_vol_ratio']:.1f} 倍"
                f"（要求 ≥ {q['d1_vol']}）"))
    d2 = f"缩量回落：回落段日均量 = 站上段的 {rr['d2_ratio']*100:.0f}%（要求 < 100%）"
    out.append((*ok(rr["d2"]), d2) if q["need_d2"] else (*skip, d2 + "；宽松版不检查"))
    out.append((*ok(rr["d3viol"] <= q["d3_exc"]), f"一浪比一浪缩量：{len(rr['waves'])} 波下跌，"
                f"{rr['d3viol']} 次没有缩量（允许 ≤ {q['d3_exc']} 次）"))
    out.append((*ok(True), f"底：今天是第 {rr['k']} 个底（要求第 {'或'.join(map(str, q['bottoms']))} 个）"))
    g = [g for g in rr["gaps"] if g["ok"]]
    if g:
        g = g[-1]
        out.append((*ok(True), f"小山丘：底{rr['gaps'].index(g)+1}→底{rr['gaps'].index(g)+2} 之间最多 {g['uprun']} 根连阳，"
                    f"反弹 +{g['rise']*100:.0f}%（要求 ≥ {q['hill_uprun']} 根连阳 或 ≥ {q['hill_rise']*100:.0f}%）"))
    out.append((*ok(True), f"缩量阴线：连续 {rr['L']} 根阴线，最大成交量 = 小山丘日均量的 {rr['run_vol_ratio']*100:.0f}%"
                f"（要求 ≥ {q['run_min']} 根，都 < 100%）"))
    if r["kind"] == "signal":
        out.append((*ok(True), f"买点：今天收阳线，收盘低于 60 日线 {-r['below_ma60']*100:.1f}%"))
    else:
        out.append(("wait", "…", "买点：还差一根阳线，收盘必须在 60 日线下方"))
    return out


def chart_ex(r, p, path):
    """例子版 chart: last ~180 days, MA60, the quiet 5 days and the down candles; volume panel with the 20%-of-peak line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9})
    j, idx, ma, t = r["j"], r["idx"], r["ma"], r["t"]
    fac = p["factor"][idx, j]
    o, h, l, c = (p[k][idx, j] / fac for k in ("open", "high", "low", "close"))   # unadjusted prices on the chart
    v = p["volume"][idx, j]
    ma = ma / fac
    a, b = max(t - 180, 0), t
    x = np.arange(b - a + 1)
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for i in range(a, b + 1):
        col = "#d63031" if c[i] >= o[i] else "#20a162"
        ax.vlines(i - a, l[i], h[i], color=col, lw=0.6)
        ax.add_patch(plt.Rectangle((i - a - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), (h[i] - l[i]) * 0.02 + 1e-9), color=col))
        axv.bar(i - a, v[i], color=col, width=0.7)
    ax.plot(x, ma[a:b + 1], color="#0984e3", lw=1.1, label="MA60")
    ymin, ymax = np.nanmin(l[a:b + 1]), np.nanmax(h[a:b + 1])
    sig = r["kind"] == "signal"
    q0 = (t if sig else t + 1) - 5          # the 5 days whose mean volume is measured
    e0 = t - 1 if sig else t
    for axx in (ax, axv):
        axx.axvspan(q0 - a - 0.5, e0 - a + 0.5, color="#6c5ce7", alpha=0.10)
    axv.axhline(r["peak"] * r["q"]["q_peak"], color="#6c5ce7", lw=0.8, ls="--")
    axv.text(1, axv.get_ylim()[1] * 0.92, f"虚线 = 一年高峰 20 日均量的 {r['q']['q_peak']*100:.0f}%", fontsize=7.5,
             color="#6c5ce7", va="top")
    axv.hlines(r["vb"], q0 - a - 0.5, e0 - a + 0.5, color="#6c5ce7", lw=1.6)
    axv.text(q0 - a - 1, axv.get_ylim()[1] * 0.92, f"买点前 5 天均量 = 高峰的 {r['q_peak']*100:.0f}%", fontsize=7.5,
             color="#6c5ce7", ha="right", va="top")
    r0 = r["run0"]
    ax.text(r0 - a - 1, l[r0] - (ymax - ymin) * 0.04, f"{r['run']}根阴线 →", fontsize=7.5, color="#20a162",
            ha="right", va="top")
    if sig:
        ax.annotate("买点", (b - a, l[b]), (b - a, l[b] - (ymax - ymin) * 0.09), ha="center", fontsize=8,
                    color="#d63031", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#d63031", lw=0.8))
    ax.set_ylim(ymin - (ymax - ymin) * 0.14, ymax * 1.04)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ticks = list(range(0, b - a + 1, max((b - a) // 6, 1)))
    axv.set_xticks(ticks, [p["dates"][idx[a + k]].strftime("%Y-%m-%d") for k in ticks])
    axv.set_yticks([])
    for s_ in (ax, axv):
        s_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def criteria_ex(r):
    q = r["q"]
    ok = lambda b: ("ok", "✓") if b else ("no", "✗")
    out = [(*ok(True), f"深跌：比一年最高价低 {-r['dd']*100:.0f}%（要求 ≥ {-q['dd']*100:.0f}%）"),
           (*ok(True), f"在 60 日线下方：最近 60 天有 {r['below']} 天收盘在线下（要求 ≥ {q['below_min']} 天）"),
           (*ok(True), f"极度缩量：买点前 5 天均量 = 一年最高 20 日均量的 {r['q_peak']*100:.0f}%（要求 ≤ {q['q_peak']*100:.0f}%）；"
                       f"在最近 120 天中排最低 {r['q_rank']*100:.0f}%（要求 ≤ {q['q_pct']*100:.0f}%）"),
           (*ok(True), f"阴线：连续 {r['run']} 根阴线，创 20 天新低（要求 ≥ {q['run_min']} 根）")]
    if r["kind"] == "signal":
        out.append((*ok(True), f"买点：今天收阳线，收盘低于 60 日线 {-r['below_ma60']*100:.1f}%"))
    else:
        out.append(("wait", "…", "买点：还差一根阳线，收盘必须在 60 日线下方"))
    return out


def page(day, n_checked, rows, out, max_cards=20):
    def card(r):
        sig, k = r["kind"] == "signal", r["r"]["k"]
        why = (f"今天是第 {k} 个底，前面 {r['down_candles']} 根缩量阴线，今天收阳线（收盘在 60 日线下方）" if sig else
               f"第 {k} 个底已有 {r['down_candles']} 根缩量阴线，还差一根阳线（收盘须在 60 日线下方）")
        tag = "严格版" + ("买点" if sig else "观察")
        crit = "".join(f'<li class="{c}"><span>{m}</span>{t}</li>' for c, m, t in criteria(r))
        return f'''<article class="card"><header><span class="badge {"buy" if sig else "watch"}">{tag}</span>
<h3>{r["code"]}</h3></header><p class="why">{why}</p>
<img loading="lazy" src="charts/{r["code"]}.png" alt="{r["code"]} 日K线">
<ul class="crit">{crit}</ul>
<dl><div><dt>收盘价</dt><dd>{r["close_px"]:.2f}</dd></div><div><dt>60 日线</dt><dd>{r["ma60_px"]:.2f}</dd></div>
<div><dt>低于 60 日线</dt><dd>{r["below_ma60"]*100:.1f}%</dd></div><div><dt>放量站上 60 日线</dt><dd>{r["d1_date"]}</dd></div>
<div><dt>跌回 60 日线下方</dt><dd>{r["stretch_start"]}</dd></div><div><dt>{k} 个底</dt><dd>{r["bottoms"].replace(";", "<br>")}</dd></div></dl></article>'''

    def card_ex(r):
        sig = r["kind"] == "signal"
        why = (f"买点前 5 天均量只有一年高峰的 {r['q_peak']*100:.0f}%，{r['run']} 根阴线后今天收阳线（收盘在 60 日线下方）" if sig else
               f"5 天均量只有一年高峰的 {r['q_peak']*100:.0f}%，已有 {r['run']} 根阴线，还差一根阳线（收盘须在 60 日线下方）")
        tag = "例子版" + ("买点" if sig else "观察")
        crit = "".join(f'<li class="{c}"><span>{m}</span>{t}</li>' for c, m, t in criteria_ex(r))
        return f'''<article class="card"><header><span class="badge {"buy" if sig else "watch"}">{tag}</span>
<h3>{r["code"]}</h3></header><p class="why">{why}</p>
<img loading="lazy" src="charts/{r["code"]}.png" alt="{r["code"]} 日K线">
<ul class="crit">{crit}</ul>
<dl><div><dt>收盘价</dt><dd>{r["close_px"]:.2f}</dd></div><div><dt>60 日线</dt><dd>{r["ma60_px"]:.2f}</dd></div>
<div><dt>低于 60 日线</dt><dd>{r["below_ma60"]*100:.1f}%</dd></div><div><dt>比一年高点</dt><dd>{r["dd"]*100:.0f}%</dd></div></dl></article>'''

    def table_ex(rs):
        if not rs:
            return ""
        tr = "".join(f"<tr><td>{r['code']}</td><td>{r['close_px']:.2f}</td><td>{r['q_peak']*100:.0f}%</td><td>{r['run']}</td>"
                     f"<td>{r['dd']*100:.0f}%</td><td>{r['below_ma60']*100:.1f}%</td></tr>" for r in rs)
        return (f'<p class="small">另外 {len(rs)} 只（缩量程度较弱，不画图）：</p><div class="tbl"><table><tr><th>代码</th><th>收盘价</th>'
                f'<th>5 天均量占高峰</th><th>阴线</th><th>比一年高点</th><th>低于 60 日线</th></tr>{tr}</table></div>')

    def cards(rs, empty):
        if not rs:
            return f'<p class="empty">{empty}</p>'
        f = lambda r: card_ex(r) if r["tier"] == "example" else card(r)
        return "".join(map(f, rs[:max_cards])) + table_ex(rs[max_cards:])

    sig = [r for r in rows if r["kind"] == "signal" and r["tier"] == "strict"]
    esig = [r for r in rows if r["kind"] == "signal" and r["tier"] == "example"]
    wat = [r for r in rows if r["kind"] == "watch" and r["tier"] == "strict"]
    ewat = [r for r in rows if r["kind"] == "watch" and r["tier"] == "example"]
    tpl = (Path(__file__).parent / "template.html").read_text()
    html = (tpl.replace("{{DAY}}", day).replace("{{N}}", f"{n_checked:,}").replace("{{NB}}", str(len(sig)))
            .replace("{{NE}}", str(len(esig))).replace("{{NW}}", str(len(wat) + len(ewat)))
            .replace("{{BUYS}}", "".join(map(card, sig)) or '<p class="empty">今天没有股票符合全部规则。</p>')
            .replace("{{EBUYS}}", cards(esig, "今天没有股票符合例子版规则。"))
            .replace("{{WATCH}}", "".join(map(card, wat)) + cards(ewat, "") if wat or ewat else '<p class="empty">暂无。</p>'))
    (out / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib", default="qlib_data")
    ap.add_argument("--out", default="site")
    ap.add_argument("--history", default="history")
    ap.add_argument("--asof", default=None, help="screen as of this trading day (test)")
    ap.add_argument("--no-page", action="store_true")
    a = ap.parse_args()
    p = load(Path(a.qlib), a.asof)
    T = len(p["dates"]) - 1
    day = str(p["dates"][T].date())
    D = p["dates"]
    mkey = np.asarray(D.year * 12 + D.month)
    wkey = np.asarray((D - pd.Timestamp("2000-01-03")).days // 7)
    checked = np.isfinite(p["close"][T])
    rows = [{**r, "tier": "strict"} for j in np.flatnonzero(checked) if (r := check(int(j), p, mkey, wkey, RULES))]
    seen = {r["code"] for r in rows}
    rows += [r for j in np.flatnonzero(checked) if (r := check_ex(int(j), p)) and r["code"] not in seen]
    rows.sort(key=lambda r: (r["kind"] != "signal", r["tier"] != "strict", r.get("q_peak", 0), r["code"]))
    keep = ["code", "tier", "kind", "close_px", "ma60_px", "below_ma60", "stretch_start", "d1_date", "bottoms", "down_candles",
            "d3viol", "q_peak", "dd"]
    Path(a.history).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: r.get(k, "") for k in keep} for r in rows], columns=keep).to_csv(Path(a.history) / f"{day}.csv", index=False)
    if not a.no_page:
        out = Path(a.out)
        (out / "charts").mkdir(parents=True, exist_ok=True)
        n = {}
        for r in rows:
            key = (r["tier"], r["kind"])
            n[key] = n.get(key, 0) + 1
            if r["tier"] == "strict":
                chart(r, p, out / "charts" / f"{r['code']}.png")
            elif n[key] <= 20:
                chart_ex(r, p, out / "charts" / f"{r['code']}.png")
        page(day, int(checked.sum()), rows, out)
    print(f"{day}: checked {checked.sum()}, " + ", ".join(
        f"{tier} signal {sum(r['kind'] == 'signal' and r['tier'] == tier for r in rows)} "
        f"watch {sum(r['kind'] == 'watch' and r['tier'] == tier for r in rows)}" for tier in ("strict", "example")))


if __name__ == "__main__":
    main()
