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
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RULES = dict(d1_lookback=250, d1_vol=2.0, pivot=5, run_min=4, bottom=3, d3_exc=1, hill_uprun=4, hill_rise=0.20,
             m_window=36, m_min_months=12, m_body=1.5, w_min=8, w_max=20, w_body=2.0, w_inside=0.70, min_listed=250)
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
        return False, False
    hm = b + int(np.argmax(mb["h"][b:e]))
    lm = b + int(np.argmin(mb["l"][b:hm + 1]))
    oo, cc = mb["o"][lm:hm + 1], mb["c"][lm:hm + 1]
    up, dn = cc > oo, cc < oo
    body = np.abs(cc - oo) / oo
    m2 = bool(not dn.any() or (up.any() and body[up].mean() >= q["m_body"] * body[dn].mean()))
    ws = int(np.searchsorted(wb["day0"], int(mb["lowday"][lm]), side="right")) - 1
    last = min(ws + q["w_max"] - 1, int(np.searchsorted(wb["id"], wid)) - 1)
    if ws < 0 or last < ws:
        return m2, False
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
    return m2, w1


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
    d2 = False
    if d1 >= 0:
        pk = d1
        for x in range(d1, s):
            if c[x] > c[pk]:
                pk = x
        d2 = v[pk + 1:s + 1].mean() < v[d1:pk + 1].mean()
    bl = [x for x in range(s, bs) if piv[x] and x + pv <= (t if watch else tt)] + [bs]
    k = len(bl)
    viol, prev, st = 0, -1.0, s
    for i, b in enumerate(bl):
        pk = st
        for x in range(st, b):
            if h[x] > h[pk]:
                pk = x
        mv = v[pk + 1:b + 1].mean() if b > pk else v[b]
        if i > 0 and not (mv < prev):
            viol += 1
        prev, st = mv, b + 1
    b_prev = bl[-2] if k > 1 else s
    pk = b_prev
    for x in range(b_prev, r0):
        if h[x] > h[pk]:
            pk = x
    hv = v[b_prev + 1:pk + 1].mean() if pk > b_prev else v[r0 - 1]
    vhill = bool((v[r0:r0 + L] < hv).all())
    nu = 0
    for i in range(1, k):
        b0, b1 = bl[i - 1], bl[i]
        gh = max(h[b0], h[b0 + 1:b1 + 1].max())
        urun = best = 0
        for x in range(b0 + 1, b1 + 1):
            urun = urun + 1 if c[x] > o[x] else 0
            best = max(best, urun)
        if best >= q["hill_uprun"] or gh / lw[b0] - 1 >= q["hill_rise"]:
            nu += 1
    return dict(L=L, bs=bs, d1=d1, d2=bool(d2), k=k, bottoms=bl, d3viol=viol, vhill=vhill, hill=nu >= 1)


def daily_ok(r, q=RULES):
    return (r is not None and r["d1"] >= 0 and r["d2"] and r["k"] == q["bottom"] and r["vhill"] and r["hill"]
            and r["d3viol"] <= q["d3_exc"])


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
            mw[x] = m2_w1(mb, wb, mkey[idx[min(x, m - 1)]], wkey[idx[min(x, m - 1)]], q)
        return mw[x]

    mb = bars(idx, o, h, lw, c, v, mkey[idx])
    wb = bars(idx, o, h, lw, c, v, wkey[idx])
    # every earlier candle day of this stretch: only the first signal of a stretch counts
    for x in range(s, t + 1):
        if not below[x - s] or idx[x] - p["first"][j] < q["min_listed"] or not c[x] > o[x]:
            continue
        r = daily(o, h, lw, c, v, ma, isd1, piv, down, s, x, q)
        if daily_ok(r) and all(flags(x)):
            if x < t:
                return None
            kind = "signal"
            break
    else:
        if not down[t] or not below[-1]:
            return None
        r = daily(o, h, lw, c, v, ma, isd1, piv, down, s, t, q, watch=True)
        if not (daily_ok(r) and all(flags(t))):
            return None
        kind = "watch"
    D = p["dates"]
    f = float(p["factor"][T, j])
    return dict(code=str(p["codes"][j]), kind=kind, j=j, idx=idx, ma=ma, s=s, t=t, r=r,
                close_px=c[t] / f, ma60_px=ma[t] / f, below_ma60=c[t] / ma[t] - 1,
                stretch_start=str(D[idx[s]].date()), d1_date=str(D[idx[r["d1"]]].date()),
                bottoms=";".join(str(D[idx[b]].date()) for b in r["bottoms"]), down_candles=r["L"], d3viol=r["d3viol"])


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
    a, b = max(rr["d1"] - 40, 0), r["t"]
    x = np.arange(b - a + 1)
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i in range(a, b + 1):
        col = "#d63031" if c[i] >= o[i] else "#20a162"
        ax.vlines(i - a, l[i], h[i], color=col, lw=0.6)
        ax.add_patch(plt.Rectangle((i - a - 0.35, min(o[i], c[i])), 0.7, max(abs(c[i] - o[i]), (h[i] - l[i]) * 0.02 + 1e-9), color=col))
        axv.bar(i - a, v[i], color=col, width=0.7)
    ax.plot(x, ma[a:b + 1], color="#0984e3", lw=1.1, label="MA60")
    ymin, ymax = np.nanmin(l[a:b + 1]), np.nanmax(h[a:b + 1])
    ax.axvline(rr["d1"] - a, color="#e67e22", lw=0.7, ls="--")
    ax.text(rr["d1"] - a + 0.5, ymax * 1.06, "放量站上60日线", color="#e67e22", fontsize=8, va="top")
    ax.axvspan(r["s"] - a - 0.5, b - a + 0.5, color="#0984e3", alpha=0.05)
    for n, bt in enumerate(rr["bottoms"], 1):
        ax.annotate(f"底{n}", (bt - a, l[bt]), (bt - a, l[bt] - (ymax - ymin) * 0.06), ha="center", fontsize=8,
                    color="#6c5ce7", arrowprops=dict(arrowstyle="-", color="#6c5ce7", lw=0.6))
    ax.set_ylim(ymin - (ymax - ymin) * 0.12, ymax * 1.08)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ticks = list(range(0, b - a + 1, max((b - a) // 6, 1)))
    axv.set_xticks(ticks, [p["dates"][idx[a + t]].strftime("%Y-%m-%d") for t in ticks])
    axv.set_yticks([])
    for s in (ax, axv):
        s.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def page(day, n_checked, rows, out):
    def card(r):
        sig = r["kind"] == "signal"
        why = (f"今天是第 3 个底，前面 {r['down_candles']} 根缩量阴线，今天收阳线（收盘在 60 日线下方）" if sig else
               f"第 3 个底已有 {r['down_candles']} 根缩量阴线，还差一根阳线（收盘须在 60 日线下方）")
        return f'''<article class="card"><header><span class="badge {"buy" if sig else "watch"}">{"买点" if sig else "观察"}</span>
<h3>{r["code"]}</h3></header><p class="why">{why}</p>
<img loading="lazy" src="charts/{r["code"]}.png" alt="{r["code"]} 日K线">
<dl><div><dt>收盘价</dt><dd>{r["close_px"]:.2f}</dd></div><div><dt>60 日线</dt><dd>{r["ma60_px"]:.2f}</dd></div>
<div><dt>低于 60 日线</dt><dd>{r["below_ma60"]*100:.1f}%</dd></div><div><dt>放量站上 60 日线</dt><dd>{r["d1_date"]}</dd></div>
<div><dt>跌回 60 日线下方</dt><dd>{r["stretch_start"]}</dd></div><div><dt>三个底</dt><dd>{r["bottoms"].replace(";", "<br>")}</dd></div></dl></article>'''
    sig = [r for r in rows if r["kind"] == "signal"]
    wat = [r for r in rows if r["kind"] == "watch"]
    tpl = (Path(__file__).parent / "template.html").read_text()
    html = (tpl.replace("{{DAY}}", day).replace("{{N}}", f"{n_checked:,}").replace("{{NB}}", str(len(sig)))
            .replace("{{NW}}", str(len(wat)))
            .replace("{{BUYS}}", "".join(map(card, sig)) or '<p class="empty">今天没有股票符合全部规则。</p>')
            .replace("{{WATCH}}", "".join(map(card, wat)) or '<p class="empty">暂无。</p>'))
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
    rows = [r for j in np.flatnonzero(checked) if (r := check(int(j), p, mkey, wkey))]
    rows.sort(key=lambda r: (r["kind"] != "signal", r["code"]))
    keep = ["code", "kind", "close_px", "ma60_px", "below_ma60", "stretch_start", "d1_date", "bottoms", "down_candles", "d3viol"]
    Path(a.history).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: r[k] for k in keep} for r in rows], columns=keep).to_csv(Path(a.history) / f"{day}.csv", index=False)
    if not a.no_page:
        out = Path(a.out)
        (out / "charts").mkdir(parents=True, exist_ok=True)
        for r in rows:
            chart(r, p, out / "charts" / f"{r['code']}.png")
        page(day, int(checked.sum()), rows, out)
    print(f"{day}: checked {checked.sum()}, signal {sum(r['kind'] == 'signal' for r in rows)}, "
          f"watch {sum(r['kind'] == 'watch' for r in rows)}")


if __name__ == "__main__":
    main()
