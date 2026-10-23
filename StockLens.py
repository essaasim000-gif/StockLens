"""
StockLens (desktop app)
=======================
Opens a welcome window; press "Open StockLens" to launch the analysis window.
All inputs (ticker, benchmark, history, risk-free rate) are typed into the
window itself. Nothing is asked in the terminal.

Install once:   python -m pip install numpy pandas scipy matplotlib yfinance
Run:            python stock_dashboard_gui.py

Uses tkinter, which comes with Python on Windows and Mac. On Linux you may
need:  sudo apt install python3-tk
Needs an internet connection (data comes from Yahoo Finance).
"""

import os
import queue
import sys
import threading

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    sys.exit("tkinter is missing. On Windows/Mac reinstall Python from python.org (tick 'tcl/tk'). "
             "On Linux run: sudo apt install python3-tk")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from scipy import stats
except ImportError as err:
    sys.exit(f"A required package is missing ({err.name}). Install everything with:\n"
             "    python -m pip install -r requirements.txt")

TRADING_DAYS = 252
HORIZON = 252        # Monte Carlo forecast length (trading days)
N_PATHS = 2000       # Monte Carlo simulations

# Colours: two palettes. set_theme() switches the module-level colour names below,
# and the chart functions read them each time a chart is drawn.
NAVY = "#0f172a"        # welcome page background (always dark)
W_BLUE = "#2563eb"      # welcome page button
W_GREEN = "#4ade80"     # welcome page logo

PALETTES = {
    "light": dict(BG="#f4f6f8", CARD="#ffffff", TEXT="#1f2933", MUTED="#6b7785", BORDER="#d7dde4",
                  BLUE="#2563eb", ORANGE="#f59e0b", RED="#dc2626", GREEN="#16a34a",
                  BTN="#2563eb", BTN_ACTIVE="#1d4ed8", TAB="#e3e8ee"),
    "dark": dict(BG="#0b1220", CARD="#162032", TEXT="#e5e7eb", MUTED="#9aa7b8", BORDER="#2a3a52",
                 BLUE="#60a5fa", ORANGE="#fbbf24", RED="#f87171", GREEN="#4ade80",
                 BTN="#2563eb", BTN_ACTIVE="#1d4ed8", TAB="#1d2a40"),
}
BG = CARD = TEXT = MUTED = BORDER = BLUE = ORANGE = RED = GREEN = BTN = BTN_ACTIVE = TAB = ""

PREF_FILE = os.path.join(os.path.expanduser("~"), ".stocklens_theme")


def set_theme(name):
    """Switch every colour name (and the matplotlib defaults) to the 'light' or 'dark' palette."""
    global BG, CARD, TEXT, MUTED, BORDER, BLUE, ORANGE, RED, GREEN, BTN, BTN_ACTIVE, TAB
    p = PALETTES[name]
    BG, CARD, TEXT, MUTED, BORDER = p["BG"], p["CARD"], p["TEXT"], p["MUTED"], p["BORDER"]
    BLUE, ORANGE, RED, GREEN = p["BLUE"], p["ORANGE"], p["RED"], p["GREEN"]
    BTN, BTN_ACTIVE, TAB = p["BTN"], p["BTN_ACTIVE"], p["TAB"]
    matplotlib.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.color": MUTED, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "font.size": 9,
        "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "legend.labelcolor": TEXT,
    })


def load_saved_theme():
    """Remember the last choice between runs (a one-word file in your home folder)."""
    try:
        with open(PREF_FILE) as f:
            return f.read().strip() == "dark"
    except OSError:
        return False


def save_theme(dark):
    try:
        with open(PREF_FILE, "w") as f:
            f.write("dark" if dark else "light")
    except OSError:
        pass


set_theme("light")


# --------------------------------------------------------------------------
# Data and calculations
# --------------------------------------------------------------------------
def load_prices(ticker, period):
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist:
        return pd.Series(dtype=float)
    prices = hist["Close"].dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.astype(float)


def load_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def drawdown_series(prices):
    return prices / prices.cummax() - 1.0


def compute_metrics(prices, bench, rf):
    rets = prices.pct_change().dropna()
    n_years = (prices.index[-1] - prices.index[0]).days / 365.25
    rf_d = rf / TRADING_DAYS
    excess = rets - rf_d

    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    sortino = excess.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan
    max_dd = drawdown_series(prices).min()
    q5, q1 = np.percentile(rets, 5), np.percentile(rets, 1)
    _, jb_p = stats.jarque_bera(rets)

    m = {
        "total_return": prices.iloc[-1] / prices.iloc[0] - 1, "cagr": cagr,
        "ann_vol": rets.std(ddof=1) * np.sqrt(TRADING_DAYS),
        "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "var95": -q5, "var99": -q1,
        "cvar95": -rets[rets <= q5].mean(), "cvar99": -rets[rets <= q1].mean(),
        "skew": stats.skew(rets), "kurt": stats.kurtosis(rets), "jb_p": jb_p,
        "best_day": rets.max(), "worst_day": rets.min(), "up_days": (rets > 0).mean(),
        "n_years": n_years,
        "beta": np.nan, "alpha": np.nan, "corr": np.nan, "r2": np.nan,
        "tracking_error": np.nan, "info_ratio": np.nan,
    }

    if bench is not None and len(bench) > 30:
        both = pd.concat([rets, bench.pct_change().dropna()], axis=1, join="inner").dropna()
        both.columns = ["s", "b"]
        if len(both) > 30 and both["b"].var() > 0:
            beta = both["s"].cov(both["b"]) / both["b"].var()
            alpha = ((both["s"] - rf_d).mean() - beta * (both["b"] - rf_d).mean()) * TRADING_DAYS
            corr = both["s"].corr(both["b"])
            active = both["s"] - both["b"]
            te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
            m.update(beta=beta, alpha=alpha, corr=corr, r2=corr ** 2, tracking_error=te,
                     info_ratio=active.mean() * TRADING_DAYS / te if te > 0 else np.nan)
    return m


def monte_carlo_gbm(prices, horizon, n_paths, seed=1):
    log_rets = np.log(prices / prices.shift(1)).dropna()
    mu, sigma = log_rets.mean(), log_rets.std(ddof=1)
    rng = np.random.default_rng(seed)
    steps = mu + sigma * rng.standard_normal((n_paths, horizon))
    paths = prices.iloc[-1] * np.exp(np.cumsum(steps, axis=1))
    return np.hstack([np.full((n_paths, 1), prices.iloc[-1]), paths])


def pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def sharpe_verdict(s):
    if not np.isfinite(s):
        return ""
    if s < 0:
        return "Poor"
    if s < 0.5:
        return "Weak"
    if s < 1:
        return "Acceptable"
    if s < 2:
        return "Good"
    return "Excellent"


# --------------------------------------------------------------------------
# Chart builders (each returns a Figure)
# --------------------------------------------------------------------------
def fig_overview(ticker, bench_ticker, prices, bench):
    fig = Figure(figsize=(10, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(prices.index, prices, color=BLUE, lw=1.6, label="Price")
    for w, style in [(50, "--"), (200, ":")]:
        if len(prices) > w:
            ax.plot(prices.index, prices.rolling(w).mean(), style, lw=1.2, color=MUTED if w == 200 else ORANGE,
                    label=f"{w}-day average")
    ax.set_title(f"{ticker} price")
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(prices.index, prices / prices.iloc[0], color=BLUE, lw=1.5, label=ticker)
    if bench is not None:
        b = bench[bench.index >= prices.index[0]]
        if len(b) > 1:
            ax.plot(b.index, b / b.iloc[0], color=ORANGE, lw=1.5, label=bench_ticker)
    ax.set_title("Growth of 1 unit invested")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    dd = drawdown_series(prices) * 100
    ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.45)
    ax.set_title("Drawdown from peak (%)")
    return fig


def fig_risk(ticker, bench_ticker, prices, bench, m):
    rets = prices.pct_change().dropna()
    fig = Figure(figsize=(10, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(rets * 100, bins=70, density=True, color=BLUE, alpha=0.65, label="Daily returns")
    xs = np.linspace(rets.min(), rets.max(), 400)
    ax.plot(xs * 100, stats.norm.pdf(xs, rets.mean(), rets.std()) / 100, color=RED, label="Normal curve")
    ax.axvline(-m["var95"] * 100, color=ORANGE, ls="--", label="VaR 95%")
    ax.set_title("Daily return distribution (%)")
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(rets, dist="norm")
    ax.plot(osm, np.asarray(osr) * 100, ".", ms=3, color=BLUE)
    ax.plot(osm, (slope * np.asarray(osm) + intercept) * 100, color=RED)
    ax.set_title("Q-Q plot (curving away = fat tails)")
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Actual quantiles (%)")

    ax = fig.add_subplot(gs[1, 0])
    if bench is not None and np.isfinite(m["beta"]):
        both = pd.concat([rets, bench.pct_change()], axis=1, join="inner").dropna()
        both.columns = ["s", "b"]
        ax.plot(both["b"] * 100, both["s"] * 100, ".", ms=3, alpha=0.4, color=BLUE)
        xs = np.linspace(both["b"].min(), both["b"].max(), 50)
        ax.plot(xs * 100, ((both["s"].mean() - m["beta"] * both["b"].mean()) + m["beta"] * xs) * 100,
                color=RED, label=f"Beta = {m['beta']:.2f}")
        ax.set_xlabel(f"{bench_ticker} daily return (%)")
        ax.set_ylabel(f"{ticker} daily return (%)")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No benchmark data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Sensitivity to the market (beta)")

    ax = fig.add_subplot(gs[1, 1])
    labels = ["VaR 95%", "ES 95%", "VaR 99%", "ES 99%"]
    vals = [m["var95"], m["cvar95"], m["var99"], m["cvar99"]]
    bars = ax.bar(labels, [v * 100 for v in vals], color=[ORANGE, RED, ORANGE, RED])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v * 100:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 100 * 1.15)
    ax.set_title("One-day loss risk (%)")
    ax.grid(False)
    return fig


def fig_forecast(prices, rf):
    rets = prices.pct_change().dropna()
    fig = Figure(figsize=(10, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    window = 63

    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    if len(rets) > window + 5:
        roll_vol = rets.rolling(window).std() * np.sqrt(TRADING_DAYS) * 100
        ex = rets - rf / TRADING_DAYS
        roll_sh = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(TRADING_DAYS)
        ax1.plot(roll_vol.index, roll_vol, color=BLUE)
        ax2.plot(roll_sh.index, roll_sh, color=GREEN)
        ax2.axhline(0, color=MUTED, ls=":")
    ax1.set_title("Rolling 3-month volatility (%)")
    ax2.set_title("Rolling 3-month Sharpe ratio")

    ax = fig.add_subplot(gs[:, 1])
    paths = monte_carlo_gbm(prices, HORIZON, N_PATHS)
    days = np.arange(HORIZON + 1)
    for i in range(40):
        ax.plot(days, paths[i], lw=0.5, color=BLUE, alpha=0.2)
    ax.plot(days, np.percentile(paths, 95, axis=0), color=GREEN, lw=2, label="Optimistic (95th)")
    ax.plot(days, np.percentile(paths, 50, axis=0), color=TEXT, lw=2, label="Median")
    ax.plot(days, np.percentile(paths, 5, axis=0), color=RED, lw=2, label="Pessimistic (5th)")
    p_loss = np.mean(paths[:, -1] < prices.iloc[-1])
    ax.set_title(f"One-year simulation  |  chance of a loss: {p_loss * 100:.0f}%")
    ax.set_xlabel("Trading days ahead")
    ax.legend(frameon=False, loc="upper left")
    return fig


def details_text(ticker, bench_ticker, prices, m, info, rf):
    name = info.get("longName") or info.get("shortName") or ticker
    rows = [
        f"{name} ({ticker})",
        f"{prices.index[0].date()} to {prices.index[-1].date()}  |  {len(prices)} trading days  |  "
        f"{m['n_years']:.1f} years  |  risk-free rate {rf * 100:.2f}%",
        "",
        "RETURNS",
        f"  Total return                         {pct(m['total_return'])}",
        f"  Annualised return (CAGR)             {pct(m['cagr'])}",
        f"  Best day / worst day                 {pct(m['best_day'])} / {pct(m['worst_day'])}",
        f"  Days up                              {pct(m['up_days'], 1)}",
        "",
        "RISK-ADJUSTED PERFORMANCE",
        f"  Annualised volatility                {pct(m['ann_vol'])}",
        f"  Sharpe ratio                         {num(m['sharpe'])}",
        f"  Sortino ratio                        {num(m['sortino'])}",
        f"  Calmar ratio                         {num(m['calmar'])}",
        f"  Maximum drawdown                     {pct(m['max_dd'])}",
        "",
        "TAIL RISK (one day, historical)",
        f"  Value at Risk 95% / 99%              {pct(m['var95'])} / {pct(m['var99'])}",
        f"  Expected Shortfall 95% / 99%         {pct(m['cvar95'])} / {pct(m['cvar99'])}",
        "",
        f"VERSUS {bench_ticker}",
        f"  Beta                                 {num(m['beta'])}",
        f"  Alpha (annual, Jensen)               {pct(m['alpha'])}",
        f"  Correlation / R-squared              {num(m['corr'])} / {num(m['r2'])}",
        f"  Tracking error / information ratio  {pct(m['tracking_error'])} / {num(m['info_ratio'])}",
        "",
        "DISTRIBUTION",
        f"  Skewness / excess kurtosis           {num(m['skew'])} / {num(m['kurt'])}",
        f"  Jarque-Bera p-value                  {num(m['jb_p'], 4)}  "
        f"({'returns are not normal' if m['jb_p'] < 0.05 else 'consistent with normal'})",
    ]
    if info:
        cap = info.get("marketCap")
        rows += ["", "FUNDAMENTALS (Yahoo Finance)",
                 f"  Market cap                           {f'${cap / 1e9:,.1f}B' if isinstance(cap, (int, float)) else 'n/a'}",
                 f"  Trailing P/E                         {num(info.get('trailingPE'))}",
                 f"  52-week low / high                 {num(info.get('fiftyTwoWeekLow'))} / "
                 f"{num(info.get('fiftyTwoWeekHigh'))}"]
    rows += ["", "For education only, not financial advice."]
    return "\n".join(rows)


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, dark=None):
        super().__init__()
        self.title("StockLens")
        self.geometry("1150x800")
        self.minsize(900, 650)
        self.results = queue.Queue()
        self.canvases = {}
        self.last = None              # last analysis, so charts can be redrawn when the theme changes
        self.status_kind = "muted"
        self.message_kind = "muted"
        self.dark_var = tk.BooleanVar(value=load_saved_theme() if dark is None else dark)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.build_inputs()
        self.build_cards()
        self.build_tabs()
        self.apply_theme(redraw=False)
        self.show_message("Enter a ticker above and press Analyse.")
        self.after(100, self.poll)

    # ---- colours (light / dark)
    def toggle_theme(self):
        save_theme(self.dark_var.get())
        self.apply_theme(redraw=True)

    def apply_theme(self, redraw=True):
        set_theme("dark" if self.dark_var.get() else "light")
        st = self.style
        self.configure(bg=BG)

        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=TEXT)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("TCheckbutton", background=BG, foreground=TEXT, focuscolor=BG)
        st.map("TCheckbutton", background=[("active", BG)], foreground=[("active", TEXT)],
               indicatorcolor=[("selected", BTN), ("!selected", CARD)])
        st.configure("TEntry", fieldbackground=CARD, foreground=TEXT, insertcolor=TEXT,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        st.map("TEntry", bordercolor=[("focus", BTN)], lightcolor=[("focus", BTN)], darkcolor=[("focus", BTN)])
        st.configure("TCombobox", fieldbackground=CARD, background=TAB, foreground=TEXT, arrowcolor=TEXT,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                     selectbackground=CARD, selectforeground=TEXT)
        st.map("TCombobox", fieldbackground=[("readonly", CARD)], foreground=[("readonly", TEXT)],
               selectbackground=[("readonly", CARD)], selectforeground=[("readonly", TEXT)])
        st.configure("Go.TButton", font=("Helvetica", 10, "bold"), padding=6, foreground="white",
                     background=BTN, borderwidth=0, focusthickness=0, focuscolor=BTN)
        st.map("Go.TButton", background=[("disabled", MUTED), ("pressed", BTN_ACTIVE), ("active", BTN_ACTIVE)],
               foreground=[("disabled", "#e5e7eb")])
        st.configure("TNotebook", background=BG, bordercolor=BORDER, lightcolor=BG, darkcolor=BG)
        st.configure("TNotebook.Tab", padding=(16, 6), background=TAB, foreground=MUTED,
                     bordercolor=BORDER, lightcolor=TAB, darkcolor=TAB)
        st.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", TEXT)],
               lightcolor=[("selected", CARD)], darkcolor=[("selected", CARD)])
        st.configure("Vertical.TScrollbar", background=TAB, troughcolor=CARD, bordercolor=BORDER,
                     arrowcolor=TEXT, lightcolor=TAB, darkcolor=TAB)

        # The pop-up list of the History drop-down is a plain tk widget
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", BTN)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        try:
            pop = self.tk.call("ttk::combobox::PopdownWindow", str(self.period_combo))
            self.tk.call(f"{pop}.f.l", "configure", "-background", CARD, "-foreground", TEXT,
                         "-selectbackground", BTN, "-selectforeground", "white")
        except tk.TclError:
            pass

        self.status.config(bg=BG, fg=self.status_color(self.status_kind))
        for frame, title, value, sub in self.cards:
            frame.config(bg=CARD, highlightbackground=BORDER, highlightcolor=BORDER)
            title.config(bg=CARD, fg=MUTED)
            value.config(bg=CARD, fg=TEXT)
            sub.config(bg=CARD, fg=MUTED)
        for frame in self.tab_frames.values():
            frame.config(bg=CARD)
        self.details.config(bg=CARD, fg=TEXT, insertbackground=TEXT)
        self.message.config(bg=CARD, fg=self.status_color(self.message_kind))

        if redraw and self.last is not None:
            self.render(**self.last)      # redraw the charts and cards in the new colours

    @staticmethod
    def status_color(kind):
        return {"muted": MUTED, "red": RED, "orange": ORANGE}.get(kind, MUTED)

    # ---- inputs
    def build_inputs(self):
        bar = ttk.Frame(self, padding=(14, 12, 14, 6))
        bar.pack(fill="x")

        self.ticker_var = tk.StringVar(value="AAPL")
        self.bench_var = tk.StringVar(value="SPY")
        self.period_var = tk.StringVar(value="5y")
        self.rf_var = tk.StringVar(value="4.0")

        def field(label, widget_factory, col):
            ttk.Label(bar, text=label, style="Muted.TLabel").grid(row=0, column=col, sticky="w", padx=(0, 4))
            w = widget_factory()
            w.grid(row=1, column=col, sticky="w", padx=(0, 14), pady=(2, 0))
            return w

        self.ticker_entry = field("Stock ticker", lambda: ttk.Entry(bar, textvariable=self.ticker_var, width=12), 0)
        field("Benchmark", lambda: ttk.Entry(bar, textvariable=self.bench_var, width=10), 1)
        self.period_combo = field("History", lambda: ttk.Combobox(
            bar, textvariable=self.period_var, width=6, state="readonly",
            values=["1y", "2y", "3y", "5y", "10y", "max"]), 2)
        field("Risk-free rate (%)", lambda: ttk.Entry(bar, textvariable=self.rf_var, width=8), 3)

        self.go_btn = ttk.Button(bar, text="Analyse", style="Go.TButton", command=self.start_analysis)
        self.go_btn.grid(row=1, column=4, padx=(6, 14), pady=(2, 0))

        self.status = tk.Label(bar, text="", bg=BG, fg=MUTED, anchor="w")
        self.status.grid(row=1, column=5, sticky="w")
        bar.columnconfigure(5, weight=1)

        self.theme_check = ttk.Checkbutton(bar, text="Dark mode", variable=self.dark_var,
                                           command=self.toggle_theme, cursor="hand2")
        self.theme_check.grid(row=0, column=6, rowspan=2, sticky="e", padx=(10, 0))

        self.bind("<Return>", lambda e: self.start_analysis())
        self.ticker_entry.focus_set()

    # ---- headline cards
    def build_cards(self):
        row = ttk.Frame(self, padding=(14, 6, 14, 6))
        row.pack(fill="x")
        self.card_vars = {}
        self.cards = []
        specs = [("sharpe", "Sharpe ratio"), ("sortino", "Sortino ratio"), ("cagr", "Annual return"),
                 ("vol", "Volatility"), ("dd", "Max drawdown"), ("beta", "Beta")]
        for i, (key, title) in enumerate(specs):
            row.columnconfigure(i, weight=1, uniform="cards")
            card = tk.Frame(row, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            title_lbl = tk.Label(card, text=title, bg=CARD, fg=MUTED, font=("Helvetica", 9))
            title_lbl.pack(anchor="w", padx=12, pady=(8, 0))
            value = tk.Label(card, text="-", bg=CARD, fg=TEXT, font=("Helvetica", 20, "bold"))
            value.pack(anchor="w", padx=12)
            sub = tk.Label(card, text=" ", bg=CARD, fg=MUTED, font=("Helvetica", 9))
            sub.pack(anchor="w", padx=12, pady=(0, 8))
            self.card_vars[key] = (value, sub)
            self.cards.append((card, title_lbl, value, sub))

    def set_card(self, key, value, sub=" ", color=None):
        value_lbl, sub_lbl = self.card_vars[key]
        value_lbl.config(text=value, fg=color or TEXT)
        sub_lbl.config(text=sub)

    # ---- tabs
    def build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        self.tab_frames = {}
        for key, title in [("overview", "Overview"), ("risk", "Risk"), ("forecast", "Trends and forecast"),
                           ("details", "All numbers")]:
            frame = tk.Frame(self.tabs, bg=CARD)
            self.tabs.add(frame, text=title)
            self.tab_frames[key] = frame

        scroll = ttk.Scrollbar(self.tab_frames["details"], orient="vertical")
        scroll.pack(side="right", fill="y")
        self.details = tk.Text(self.tab_frames["details"], font=("Courier", 11), bg=CARD, fg=TEXT,
                               relief="flat", padx=16, pady=12, wrap="none", highlightthickness=0,
                               yscrollcommand=scroll.set)
        scroll.config(command=self.details.yview)
        self.details.pack(side="left", fill="both", expand=True)
        self.details.config(state="disabled")

        self.message = tk.Label(self.tab_frames["overview"], text="", bg=CARD, fg=MUTED, font=("Helvetica", 13))

    def show_message(self, text, kind="muted"):
        self.clear_canvas("overview")
        self.message_kind = kind
        self.message.config(text=text, fg=self.status_color(kind))
        self.message.place(relx=0.5, rely=0.5, anchor="center")

    def clear_canvas(self, key):
        if key in self.canvases:
            self.canvases[key].get_tk_widget().destroy()
            del self.canvases[key]

    def draw(self, key, fig):
        self.clear_canvas(key)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_frames[key])
        widget = canvas.get_tk_widget()
        widget.config(width=200, height=200)   # small requested size, so it stretches to fit the window
        widget.pack(fill="both", expand=True)

        pending = {"id": None}

        def apply_fit(fig=fig, canvas=canvas, widget=widget):
            # Match the figure to its widget using the final dpi (avoids clipping on scaled displays)
            pending["id"] = None
            w, h = widget.winfo_width(), widget.winfo_height()
            if w > 50 and h > 50:
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
                canvas.draw_idle()

        def on_configure(event, widget=widget):
            if pending["id"] is not None:
                widget.after_cancel(pending["id"])
            pending["id"] = widget.after(120, apply_fit)   # wait until resizing has settled

        widget.bind("<Configure>", on_configure, add="+")
        pending["id"] = widget.after(300, apply_fit)
        canvas.draw()
        self.canvases[key] = canvas

    # ---- running an analysis
    def start_analysis(self):
        if self.go_btn["state"] == "disabled":
            return
        ticker = self.ticker_var.get().strip().upper()
        bench_ticker = self.bench_var.get().strip().upper()
        period = self.period_var.get()
        if not ticker:
            self.set_status("Please enter a ticker.", "red")
            return
        try:
            rf = float(self.rf_var.get().replace("%", "").strip()) / 100
        except ValueError:
            self.set_status("Risk-free rate must be a number, e.g. 4.0", "red")
            return

        self.go_btn.config(state="disabled")
        self.set_status(f"Downloading {ticker}...", "muted")
        threading.Thread(target=self.worker, args=(ticker, bench_ticker, period, rf), daemon=True).start()

    def worker(self, ticker, bench_ticker, period, rf):
        """Runs in a background thread so the window never freezes."""
        try:
            prices = load_prices(ticker, period)
            if len(prices) < 60:
                self.results.put(("error", f"No usable data for '{ticker}'. Check the symbol and your internet "
                                           "connection (e.g. AAPL, MSFT, HBL.KA, RELIANCE.NS)."))
                return
            bench, bench_note = None, ""
            if bench_ticker:
                try:
                    b = load_prices(bench_ticker, period)
                    if len(b) >= 60:
                        bench = b
                    else:
                        bench_note = f"Benchmark '{bench_ticker}' not found, so beta and alpha are skipped."
                except Exception:
                    bench_note = f"Benchmark '{bench_ticker}' failed to load, so beta and alpha are skipped."
            m = compute_metrics(prices, bench, rf)
            info = load_info(ticker)
            self.results.put(("ok", dict(ticker=ticker, bench_ticker=bench_ticker, prices=prices, bench=bench,
                                         m=m, info=info, rf=rf, note=bench_note)))
        except Exception as err:
            self.results.put(("error", f"Something went wrong: {err}"))

    def poll(self):
        try:
            kind, payload = self.results.get_nowait()
        except queue.Empty:
            self.after(100, self.poll)
            return
        self.go_btn.config(state="normal")
        if kind == "error":
            self.set_status(payload, "red")
            self.show_message(payload, "red")
        else:
            try:
                self.last = payload
                self.render(**payload)
            except Exception as err:
                self.set_status(f"Could not draw results: {err}", "red")
        self.after(100, self.poll)

    def set_status(self, text, kind="muted"):
        self.status_kind = kind
        self.status.config(text=text, fg=self.status_color(kind))

    def render(self, ticker, bench_ticker, prices, bench, m, info, rf, note):
        self.message.place_forget()
        verdict = sharpe_verdict(m["sharpe"])
        colour = GREEN if m["sharpe"] >= 1 else (RED if m["sharpe"] < 0 else TEXT)
        self.set_card("sharpe", num(m["sharpe"]), verdict, colour)
        self.set_card("sortino", num(m["sortino"]))
        self.set_card("cagr", pct(m["cagr"], 1), f"total {pct(m['total_return'], 0)}")
        self.set_card("vol", pct(m["ann_vol"], 1))
        self.set_card("dd", pct(m["max_dd"], 1), color=RED)
        self.set_card("beta", num(m["beta"]), f"vs {bench_ticker}" if bench is not None else "no benchmark")

        self.draw("overview", fig_overview(ticker, bench_ticker, prices, bench))
        self.draw("risk", fig_risk(ticker, bench_ticker, prices, bench, m))
        self.draw("forecast", fig_forecast(prices, rf))

        self.details.config(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", details_text(ticker, bench_ticker, prices, m, info, rf))
        self.details.config(state="disabled")

        name = info.get("longName") or info.get("shortName") or ticker
        self.title(f"{name} ({ticker}) - StockLens")
        self.set_status(note or f"{name}: {prices.index[0].date()} to {prices.index[-1].date()}",
                        "orange" if note else "muted")


class Welcome(tk.Tk):
    """Full-screen start page: the StockLens name and a button that opens the program.

    Esc or F11 switches between full screen and a normal window. The Quit link closes everything.
    """

    def __init__(self):
        super().__init__()
        self.title("StockLens")
        self.opened = False
        self.configure(bg=NAVY)

        self.screen_w = self.winfo_screenwidth()
        self.screen_h = self.winfo_screenheight()
        k = min(max(self.screen_h / 800, 1.0), 1.6)          # scale text and logo on big screens
        self.minsize(640, 500)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Launch.TButton", font=("Helvetica", int(14 * k), "bold"), foreground="white",
                        background=W_BLUE, padding=(int(36 * k), int(14 * k)), borderwidth=0,
                        focusthickness=0, focuscolor=W_BLUE)
        style.map("Launch.TButton", background=[("pressed", "#1e40af"), ("active", "#1d4ed8")])

        # Logo and name, just above the centre
        top = tk.Frame(self, bg=NAVY)
        top.place(relx=0.5, rely=0.52, anchor="s")
        size = int(110 * k)
        logo = tk.Canvas(top, width=size, height=size, bg=NAVY, highlightthickness=0)
        logo.pack()
        logo.create_oval(10 * k, 10 * k, 74 * k, 74 * k, outline="white", width=5 * k)                  # lens
        logo.create_line(68 * k, 68 * k, 98 * k, 98 * k, fill="white", width=9 * k, capstyle="round")   # handle
        logo.create_line(26 * k, 54 * k, 38 * k, 42 * k, 47 * k, 49 * k, 60 * k, 30 * k, fill=W_GREEN,
                         width=4 * k, joinstyle="round", capstyle="round")                              # chart
        tk.Label(top, text="StockLens", bg=NAVY, fg="white",
                 font=("Helvetica", int(44 * k), "bold")).pack(pady=(4, 0))
        tk.Label(top, text="Evaluate any stock in seconds", bg=NAVY, fg="#9fb3c8",
                 font=("Helvetica", int(14 * k))).pack(pady=(2, 0))

        # Button in the middle of the window
        self.launch_btn = ttk.Button(self, text="Open StockLens", style="Launch.TButton", command=self.open_app,
                                     cursor="hand2")
        self.launch_btn.place(relx=0.5, rely=0.64, anchor="center")

        tk.Label(self, text="For education only. Not financial advice.", bg=NAVY, fg="#64748b",
                 font=("Helvetica", 10)).place(relx=0.5, rely=0.95, anchor="center")
        self.hint = tk.Label(self, text="Esc: leave full screen", bg=NAVY, fg="#475569", font=("Helvetica", 9))
        self.hint.place(relx=0.02, rely=0.97, anchor="sw")
        quit_link = tk.Label(self, text="Quit", bg=NAVY, fg="#94a3b8", font=("Helvetica", 10, "underline"),
                             cursor="hand2")
        quit_link.place(relx=0.98, rely=0.97, anchor="se")
        quit_link.bind("<Button-1>", lambda e: self.destroy())

        self.bind("<Return>", lambda e: self.open_app())
        self.bind("<Escape>", lambda e: self.set_fullscreen(False))
        self.bind("<F11>", lambda e: self.set_fullscreen(not self.is_full))
        self.launch_btn.focus_set()

        self.is_full = False
        self.set_fullscreen(True)

    def set_fullscreen(self, on):
        self.is_full = on
        if on:
            self.geometry(f"{self.screen_w}x{self.screen_h}+0+0")   # fills the screen even without a window manager
            try:
                self.attributes("-fullscreen", True)
            except tk.TclError:
                pass
            self.hint.config(text="Esc: leave full screen")
        else:
            try:
                self.attributes("-fullscreen", False)
            except tk.TclError:
                pass
            w, h = 900, 620
            self.geometry(f"{w}x{h}+{(self.screen_w - w) // 2}+{max((self.screen_h - h) // 3, 0)}")
            self.hint.config(text="F11: full screen")

    def open_app(self):
        self.opened = True
        self.destroy()


def main():
    welcome = Welcome()
    welcome.mainloop()
    if welcome.opened:            # closing the welcome window with X quits without opening the program
        App().mainloop()


if __name__ == "__main__":
    main()