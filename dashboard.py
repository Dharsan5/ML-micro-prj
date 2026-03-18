"""
Amazon Product K-Means Clustering Dashboard
Minimal modern Apple-style UI — Tkinter + Matplotlib + PIL
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from PIL import Image, ImageDraw, ImageTk

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── path resolution ──────────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CSV = os.path.join(BASE_DIR, "Dataset", "amazon.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES       = ["discounted_price", "actual_price",
                  "discount_percentage", "rating", "rating_count"]
FEATURE_LABELS = ["Disc. Price", "Actual Price", "Discount %", "Rating", "Rating Count"]

# ── Apple design tokens ──────────────────────────────────────────────────────
C = {
    "bg":         "#F2F2F7",
    "surface":    "#FFFFFF",
    "surface2":   "#F9F9F9",
    "border":     "#E5E5EA",
    "text1":      "#1C1C1E",
    "text2":      "#6D6D72",
    "text3":      "#AEAEB2",
    "accent":     "#007AFF",
    "accent_d":   "#0051D5",
    "accent_fg":  "#FFFFFF",
    "nav_active": "#EBF3FF",
    "nav_hover":  "#F5F5F7",
    "success":    "#34C759",
    "warning":    "#FF9500",
    "danger":     "#FF3B30",
    "seg_border": "#D1D1D6",
}

APPLE_PALETTE = ["#007AFF", "#FF9500", "#34C759", "#FF3B30", "#AF52DE",
                 "#FF2D55", "#5AC8FA"]

FONT = "Segoe UI"

# ── Matplotlib Apple style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#E5E5EA",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        "#F2F2F7",
    "grid.linewidth":    0.8,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "font.family":       ["Segoe UI", "Helvetica Neue", "sans-serif"],
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.titlepad":     14,
    "axes.labelsize":    9,
    "axes.labelcolor":   "#6D6D72",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.color":       "#AEAEB2",
    "ytick.color":       "#AEAEB2",
    "legend.frameon":    False,
    "legend.fontsize":   8.5,
    "figure.dpi":        110,
})


# ════════════════════════════════════════════════════════════════════════════
# PIL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rounded_image(w, h, radius, fill_hex, outline_hex=None, outline_w=0):
    """Return a Tkinter PhotoImage of a solid rounded rectangle."""
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = _hex_to_rgb(fill_hex) + (255,)
    out  = (_hex_to_rgb(outline_hex) + (255,)) if outline_hex else None
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                           fill=fill, outline=out, width=outline_w)
    return ImageTk.PhotoImage(img)


# ════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ════════════════════════════════════════════════════════════════════════════

class PillButton(tk.Label):
    """True pill-shaped button via PIL rounded-rect image background."""

    def __init__(self, parent, text, command, width=190, height=38,
                 bg=C["accent"], hover=C["accent_d"], fg=C["accent_fg"],
                 font_size=10, bold=True, **kw):
        r = height // 2
        self._img_n = _rounded_image(width, height, r, bg)
        self._img_h = _rounded_image(width, height, r, hover)
        bg_p = parent.cget("bg")
        super().__init__(parent, image=self._img_n, text=text,
                         compound="center", fg=fg, cursor="hand2",
                         font=(FONT, font_size, "bold" if bold else "normal"),
                         bg=bg_p, borderwidth=0, highlightthickness=0, **kw)
        self.bind("<Enter>",           lambda e: self.config(image=self._img_h))
        self.bind("<Leave>",           lambda e: self.config(image=self._img_n))
        self.bind("<ButtonRelease-1>", lambda e: command())


class GhostButton(tk.Label):
    """Borderless text button."""

    def __init__(self, parent, text, command, **kw):
        super().__init__(parent, text=text, cursor="hand2",
                         font=(FONT, 9), fg=C["accent"],
                         bg=parent.cget("bg"), padx=8, pady=4, **kw)
        self.bind("<Enter>",          lambda e: self.config(fg=C["accent_d"]))
        self.bind("<Leave>",          lambda e: self.config(fg=C["accent"]))
        self.bind("<ButtonRelease-1>",lambda e: command())


class SegmentedControl(tk.Frame):
    """Apple-style segmented control with PIL rounded outline."""

    def __init__(self, parent, values, variable, on_change=None, **kw):
        bg_p = parent.cget("bg")
        super().__init__(parent, bg=bg_p, bd=0, **kw)
        self._var       = variable
        self._on_change = on_change
        self._btns      = {}

        # Outer pill border via PIL
        total_w = len(values) * 36 + 2
        self._border_img = _rounded_image(total_w, 30, 8,
                                          bg_p, C["seg_border"], 1)
        bg_lbl = tk.Label(self, image=self._border_img, bg=bg_p,
                          borderwidth=0, highlightthickness=0)
        bg_lbl.pack()

        inner = tk.Frame(bg_lbl, bg=bg_p)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        for v in values:
            b = tk.Label(inner, text=str(v), width=2, height=1,
                         font=(FONT, 9), cursor="hand2", bd=0, relief="flat")
            b.pack(side="left", ipadx=5, ipady=2)
            b.bind("<ButtonRelease-1>", lambda e, val=v: self._pick(val))
            self._btns[v] = b
        self._refresh()

    def _pick(self, val):
        self._var.set(val)
        self._refresh()
        if self._on_change:
            self._on_change(val)

    def _refresh(self):
        cur = int(self._var.get())
        for val, btn in self._btns.items():
            if val == cur:
                btn.config(bg=C["accent"], fg=C["accent_fg"])
            else:
                btn.config(bg=C["surface"], fg=C["text1"])


class NavItem(tk.Frame):
    """Sidebar navigation row with active pill highlight."""

    def __init__(self, parent, label, command, tooltip_text="", **kw):
        super().__init__(parent, bg=C["surface"], cursor="hand2", **kw)
        self._cmd    = command
        self._active = False
        self._dot = tk.Label(self, text="•", font=(FONT, 10),
                             bg=C["surface"], fg=C["text3"], width=2)
        self._dot.pack(side="left", padx=(10, 0))
        self._lbl = tk.Label(self, text=label, font=(FONT, 9),
                             bg=C["surface"], fg=C["text1"],
                             anchor="w", pady=7)
        self._lbl.pack(side="left", fill="x", expand=True, padx=(2, 10))
        for w in (self, self._dot, self._lbl):
            w.bind("<Enter>",          self._on_hover)
            w.bind("<Leave>",          self._on_leave)
            w.bind("<ButtonRelease-1>",self._on_click)
        if tooltip_text:
            Tooltip(self, tooltip_text)

    def _on_hover(self, _=None):
        if not self._active:
            for w in (self, self._dot, self._lbl):
                w.config(bg=C["nav_hover"])

    def _on_leave(self, _=None):
        if not self._active:
            for w in (self, self._dot, self._lbl):
                w.config(bg=C["surface"])

    def _on_click(self, _=None):
        self._cmd()

    def set_active(self, active: bool):
        self._active = active
        bg = C["nav_active"] if active else C["surface"]
        for w in (self, self._dot, self._lbl):
            w.config(bg=bg)
        if active:
            self._dot.config(fg=C["accent"])
            self._lbl.config(fg=C["accent"], font=(FONT, 9, "bold"))
        else:
            self._dot.config(fg=C["text3"])
            self._lbl.config(fg=C["text1"], font=(FONT, 9))


class Tooltip:
    """Lightweight floating tooltip shown on widget hover."""

    def __init__(self, widget, text):
        self._widget = widget
        self._text   = text
        self._win    = None
        widget.bind("<Enter>",       self._show, add="+")
        widget.bind("<Leave>",       self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _=None):
        if self._win:
            return
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 6
        y = self._widget.winfo_rooty() + self._widget.winfo_height() // 2 - 12
        self._win = tk.Toplevel(self._widget)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{x}+{y}")
        self._win.wm_attributes("-topmost", True)
        tk.Label(self._win, text=self._text,
                 bg=C["text1"], fg=C["surface"],
                 font=(FONT, 8), padx=10, pady=5).pack()

    def _hide(self, _=None):
        if self._win:
            self._win.destroy()
            self._win = None


class TabBar(tk.Frame):
    """Apple-style underline tab switcher."""

    def __init__(self, parent, tabs, on_switch, **kw):
        super().__init__(parent, bg=C["surface2"], **kw)
        self._tabs      = {}
        self._indicators= {}
        self._active    = None
        self._on_switch = on_switch

        for name in tabs:
            col = tk.Frame(self, bg=C["surface2"])
            col.pack(side="left")
            btn = tk.Label(col, text=name, font=(FONT, 9),
                           fg=C["text2"], bg=C["surface2"],
                           cursor="hand2", padx=16, pady=9)
            btn.pack()
            # Blue underline indicator
            ind = tk.Frame(col, height=2, bg=C["surface2"])
            ind.pack(fill="x")
            btn.bind("<ButtonRelease-1>", lambda e, n=name: self._pick(n))
            self._tabs[name]       = btn
            self._indicators[name] = ind

        self._pick(tabs[0])

    def _pick(self, name):
        if self._active:
            self._tabs[self._active].config(fg=C["text2"],
                                            font=(FONT, 9))
            self._indicators[self._active].config(bg=C["surface2"])
        self._active = name
        self._tabs[name].config(fg=C["accent"], font=(FONT, 9, "bold"))
        self._indicators[name].config(bg=C["accent"])
        self._on_switch(name)


class Divider(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, height=1, bg=C["border"], **kw)


class SectionLabel(tk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, text=text.upper(),
                         font=(FONT, 7, "bold"), fg=C["text3"],
                         bg=C["surface"], anchor="w", padx=14, pady=6, **kw)


class MetricCard(tk.Frame):
    """Stat card with subtle rounded-outline border."""

    def __init__(self, parent, label, value="—", **kw):
        super().__init__(parent, bg=parent.cget("bg"), **kw)
        inner = tk.Frame(self, bg=C["surface"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        tk.Label(inner, text=label, font=(FONT, 8), fg=C["text2"],
                 bg=C["surface"], anchor="w").pack(anchor="w", padx=10, pady=(8, 0))
        self._val = tk.Label(inner, text=value, font=(FONT, 15, "bold"),
                             fg=C["text1"], bg=C["surface"], anchor="w")
        self._val.pack(anchor="w", padx=10, pady=(0, 8))

    def update(self, value):
        self._val.config(text=str(value))


class DataTable(tk.Frame):
    """Sortable, filterable data table shown in the Data tab."""

    COLS = ("product_name", "primary_category", "discounted_price",
            "actual_price", "discount_percentage", "rating", "rating_count", "cluster")
    HDRS = ("Product", "Category", "Price (₹)", "Actual (₹)",
            "Discount %", "Rating", "Reviews", "Cluster")
    WIDS = (250, 160, 80, 80, 80, 65, 85, 75)

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["surface2"], **kw)
        self._df        = None
        self._sort_col  = None
        self._sort_asc  = True
        self._build()

    def _build(self):
        # ── Toolbar ──────────────────────────────────────────────
        bar = tk.Frame(self, bg=C["surface2"])
        bar.pack(fill="x", padx=16, pady=(12, 6))

        # Search entry with pill outline
        search_outer = tk.Frame(bar, bg=C["border"])
        search_outer.pack(side="left")
        search_inner = tk.Frame(search_outer, bg=C["surface"])
        search_inner.pack(padx=1, pady=1)
        tk.Label(search_inner, text="⌕", font=(FONT, 10), fg=C["text3"],
                 bg=C["surface"]).pack(side="left", padx=(8, 2))
        self._search_var = tk.StringVar()
        tk.Entry(search_inner, textvariable=self._search_var,
                 font=(FONT, 9), relief="flat", bg=C["surface"],
                 fg=C["text1"], insertbackground=C["accent"],
                 width=22).pack(side="left", ipady=5, padx=(0, 8))
        self._search_var.trace_add("write", lambda *_: self._apply_filter())

        # Cluster filter
        tk.Label(bar, text="Cluster", font=(FONT, 9), fg=C["text2"],
                 bg=C["surface2"]).pack(side="left", padx=(16, 6))
        self._cluster_var = tk.StringVar(value="All")
        self._cluster_cb  = ttk.Combobox(bar, textvariable=self._cluster_var,
                                          width=10, state="readonly",
                                          font=(FONT, 9))
        self._cluster_cb.pack(side="left")
        self._cluster_cb.bind("<<ComboboxSelected>>",
                              lambda _: self._apply_filter())

        self._row_lbl = tk.Label(bar, text="", font=(FONT, 8), fg=C["text2"],
                                  bg=C["surface2"])
        self._row_lbl.pack(side="right", padx=4)

        # ── Treeview style ────────────────────────────────────────
        style = ttk.Style()
        style.configure("Tbl.Treeview",
                        background=C["surface"],
                        foreground=C["text1"],
                        fieldbackground=C["surface"],
                        rowheight=30,
                        font=(FONT, 9),
                        borderwidth=0)
        style.configure("Tbl.Treeview.Heading",
                        background=C["surface2"],
                        foreground=C["text2"],
                        font=(FONT, 8, "bold"),
                        relief="flat", borderwidth=0)
        style.map("Tbl.Treeview",
                  background=[("selected", C["nav_active"])],
                  foreground=[("selected", C["accent"])])
        style.layout("Tbl.Treeview", [
            ("Treeview.treearea", {"sticky": "nswe"})])

        # ── Treeview ──────────────────────────────────────────────
        wrap = tk.Frame(self, bg=C["surface2"])
        wrap.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self._tree = ttk.Treeview(wrap, columns=self.COLS,
                                   show="headings", style="Tbl.Treeview",
                                   selectmode="extended")
        for col, hdr, w in zip(self.COLS, self.HDRS, self.WIDS):
            self._tree.heading(col, text=hdr,
                               command=lambda c=col: self._sort(c))
            self._tree.column(col, width=w, minwidth=50, anchor="w")
        self._tree.tag_configure("even", background="#FAFAFA")
        self._tree.tag_configure("odd",  background=C["surface"])

        vsb = ttk.Scrollbar(wrap, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        Tooltip(self._tree, "Click a column header to sort  ·  Shift-click to multi-select")

    def load(self, df):
        self._df = df.copy()
        options = ["All"] + [str(c) for c in sorted(df["cluster"].unique())]
        self._cluster_cb["values"] = options
        self._cluster_var.set("All")
        self._search_var.set("")
        self._populate(self._df)

    def _apply_filter(self):
        if self._df is None:
            return
        df = self._df.copy()
        q = self._search_var.get().lower().strip()
        if q:
            df = df[df["product_name"].str.lower().str.contains(q, na=False) |
                    df["primary_category"].str.lower().str.contains(q, na=False)]
        c = self._cluster_var.get()
        if c != "All":
            df = df[df["cluster"] == int(c)]
        self._populate(df)

    def _sort(self, col):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        if self._df is None:
            return
        df = self._df.copy()
        q = self._search_var.get().lower().strip()
        if q:
            df = df[df["product_name"].str.lower().str.contains(q, na=False)]
        c = self._cluster_var.get()
        if c != "All":
            df = df[df["cluster"] == int(c)]
        try:
            df = df.sort_values(col, ascending=self._sort_asc)
        except Exception:
            pass
        self._populate(df)
        # Update heading to show sort direction
        arrow = " ↑" if self._sort_asc else " ↓"
        for c2, hdr in zip(self.COLS, self.HDRS):
            self._tree.heading(c2, text=(hdr + arrow) if c2 == col else hdr)

    def _populate(self, df):
        self._tree.delete(*self._tree.get_children())
        for i, (_, row) in enumerate(df.iterrows()):
            tag = "even" if i % 2 == 0 else "odd"
            self._tree.insert("", "end", tags=(tag,), values=(
                str(row.get("product_name", ""))[:55],
                str(row.get("primary_category", ""))[:28],
                f"{row['discounted_price']:,.0f}",
                f"{row['actual_price']:,.0f}",
                f"{row['discount_percentage']:.0f}%",
                f"{row['rating']:.1f}",
                f"{int(row['rating_count']):,}",
                f"Cluster {int(row['cluster'])}",
            ))
        self._row_lbl.config(text=f"{len(df):,} rows")


# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

def clean_currency(s):
    return pd.to_numeric(
        s.astype(str).str.replace("₹", "", regex=False)
                     .str.replace(",", "", regex=False).str.strip(),
        errors="coerce")


def load_and_clean(path):
    df = pd.read_csv(path, encoding="utf-8")
    df["discounted_price"]    = clean_currency(df["discounted_price"])
    df["actual_price"]        = clean_currency(df["actual_price"])
    df["discount_percentage"] = pd.to_numeric(
        df["discount_percentage"].astype(str)
          .str.replace("%", "", regex=False).str.strip(), errors="coerce")
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(
        df["rating_count"].astype(str)
          .str.replace(",", "", regex=False).str.strip(), errors="coerce")
    df["primary_category"] = df["category"].astype(str).str.split("|").str[0]
    df.dropna(subset=FEATURES, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scale_features(df):
    X = df[FEATURES].copy()
    X["discounted_price"] = np.log1p(X["discounted_price"])
    X["actual_price"]     = np.log1p(X["actual_price"])
    X["rating_count"]     = np.log1p(X["rating_count"])
    return StandardScaler().fit_transform(X)


def run_kmeans(X_scaled, k):
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    return km, labels, sil


def elbow_data(X_scaled, k_range):
    inertia, sil = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, lbl))
    return inertia, sil


# ════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _pal(k):
    return [APPLE_PALETTE[i % len(APPLE_PALETTE)] for i in range(k)]


def fig_elbow(k_range, inertia, sil_scores, chosen_k):
    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    ax1.plot(ks, inertia, "o-", color=C["accent"], lw=2, ms=5)
    ax1.axvline(chosen_k, color=C["warning"], ls="--", lw=1.4, label=f"K = {chosen_k}")
    ax1.set(xlabel="Number of clusters (K)", ylabel="Inertia (WCSS)", title="Elbow Curve")
    ax1.legend()
    ax2.plot(ks, sil_scores, "o-", color=C["success"], lw=2, ms=5)
    ax2.axvline(chosen_k, color=C["warning"], ls="--", lw=1.4, label=f"K = {chosen_k}")
    ax2.set(xlabel="Number of clusters (K)", ylabel="Silhouette Score",
            title="Silhouette Score vs K")
    ax2.legend()
    fig.suptitle("Selecting Optimal K", fontsize=12, fontweight="bold",
                 color=C["text1"], y=1.01)
    fig.tight_layout()
    return fig


def fig_pca(X_scaled, labels, km, k):
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100
    cen = pca.transform(km.cluster_centers_)
    pal = _pal(k)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in range(k):
        m = labels == c
        ax.scatter(X2[m, 0], X2[m, 1], c=pal[c], label=f"Cluster {c}",
                   alpha=0.45, s=16, edgecolors="none")
    ax.scatter(cen[:, 0], cen[:, 1], s=200, marker="P",
               c=pal, edgecolors="#1C1C1E", lw=0.8, zorder=5, label="Centroids")
    ax.set(xlabel=f"PC1  ({var[0]:.1f}% variance)",
           ylabel=f"PC2  ({var[1]:.1f}% variance)",
           title="K-Means Clusters — PCA Projection")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_heatmap(km, k):
    cdf = pd.DataFrame(km.cluster_centers_, columns=FEATURE_LABELS,
                       index=[f"Cluster {i}" for i in range(k)])
    fig, ax = plt.subplots(figsize=(7.5, max(2.8, k * 0.8 + 1.4)))
    sns.heatmap(cdf, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.6, linecolor="#F2F2F7", ax=ax,
                cbar_kws={"label": "Standardised value", "shrink": 0.7})
    ax.set_title("Cluster Centroids Heatmap")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def fig_profiles(df, k):
    summary = df.groupby("cluster")[FEATURES].mean()
    norm    = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    x, w    = np.arange(len(FEATURES)), 0.75 / k
    pal     = _pal(k)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i in range(k):
        ax.bar(x + i * w, norm.iloc[i], w, label=f"Cluster {i}",
               color=pal[i], alpha=0.88, edgecolor="white", lw=0.5)
    ax.set_xticks(x + w * (k - 1) / 2)
    ax.set_xticklabels(FEATURE_LABELS)
    ax.set_ylim(0, 1.15)
    ax.set(ylabel="Normalised mean value", title="Cluster Profile Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_scatter(df, x_col, y_col, k):
    pal = _pal(k)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for c in range(k):
        m = df["cluster"] == c
        ax.scatter(df.loc[m, x_col], df.loc[m, y_col], c=pal[c],
                   label=f"Cluster {c}", alpha=0.45, s=18, edgecolors="none")
    ax.set(xlabel=x_col.replace("_", " ").title(),
           ylabel=y_col.replace("_", " ").title(),
           title=f"{y_col.replace('_',' ').title()} vs {x_col.replace('_',' ').title()}")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_correlation(df):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    corr = df[FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.6,
                linecolor="#F2F2F7", ax=ax, square=True,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_distribution(df, col):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    clipped = df[col].clip(upper=df[col].quantile(0.97))
    ax.hist(clipped, bins=40, color=C["accent"], alpha=0.8,
            edgecolor="white", lw=0.4)
    ax.set(xlabel=col.replace("_", " ").title(), ylabel="Count",
           title=f"Distribution — {col.replace('_', ' ').title()}")
    fig.tight_layout()
    return fig


def fig_top_categories(df):
    top = (df.groupby("primary_category")["rating"]
             .agg(["mean", "count"])
             .query("count >= 5")
             .sort_values("mean", ascending=False)
             .head(12).reset_index())
    pal = sns.color_palette([C["accent"], C["success"], C["warning"],
                             "#AF52DE", "#FF2D55"], n_colors=len(top))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(top["primary_category"], top["mean"],
                   color=pal, alpha=0.88, edgecolor="white", lw=0.4)
    ax.set_xlabel("Average rating")
    ax.set_title("Top Categories by Average Rating")
    ax.set_xlim(3.0, 5.4)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8, color=C["text2"])
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_segment_pie(df, k):
    counts = df["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        counts, labels=[f"Cluster {i}" for i in counts.index],
        autopct="%1.1f%%", colors=_pal(k), startangle=140,
        wedgeprops={"edgecolor": "white", "lw": 2},
        textprops={"color": C["text1"]})
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(C["text1"])
    ax.set_title("Segment Distribution")
    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Product Clustering")
        self.geometry("1280x800")
        self.minsize(1060, 680)
        self.configure(bg=C["bg"])

        self.df           = None
        self.X_scaled     = None
        self.km           = None
        self.k_var        = tk.IntVar(value=3)
        self._inertia     = None
        self._sil         = None
        self._k_range     = range(2, 10)
        self._current_fig = None
        self._nav_items   = {}
        self._active_nav  = None
        self._auto_refresh = tk.BooleanVar(value=False)
        self._cat_var      = tk.StringVar(value="All categories")
        self._data_table   = None
        self._active_tab   = "Chart"

        self._build_ui()
        self._auto_load()

    # ── build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_titlebar()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._build_sidebar(body)
        self._build_main(body)
        self._build_statusbar()

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=C["surface"], height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        dots = tk.Frame(bar, bg=C["surface"])
        dots.place(x=18, rely=0.5, anchor="w")
        for col in ("#FF5F57", "#FEBC2E", "#28C840"):
            tk.Label(dots, text="●", fg=col, bg=C["surface"],
                     font=(FONT, 10)).pack(side="left", padx=2)

        tk.Label(bar, text="Product Clustering",
                 font=(FONT, 13, "bold"), fg=C["text1"],
                 bg=C["surface"]).place(relx=0.5, rely=0.5, anchor="center")

        right = tk.Frame(bar, bg=C["surface"])
        right.place(relx=1.0, rely=0.5, anchor="e", x=-16)
        GhostButton(right, "Export CSV", self._export_csv).pack(side="left", padx=2)
        GhostButton(right, "Save Plot",  self._save_plot ).pack(side="left", padx=2)
        GhostButton(right, "Browse CSV", self._browse_csv).pack(side="left", padx=2)

        Divider(self).pack(fill="x")

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C["surface"], width=230)
        sb.pack(side="left", fill="y", padx=(0, 10), pady=12)
        sb.pack_propagate(False)

        # ── K clusters ──
        tk.Label(sb, text="Clusters", font=(FONT, 10, "bold"),
                 fg=C["text1"], bg=C["surface"],
                 anchor="w").pack(fill="x", padx=14, pady=(18, 2))
        tk.Label(sb, text="Number of clusters K",
                 font=(FONT, 8), fg=C["text2"], bg=C["surface"],
                 anchor="w").pack(fill="x", padx=14)

        seg_frame = tk.Frame(sb, bg=C["surface"])
        seg_frame.pack(padx=14, pady=(8, 4), anchor="w")
        self._seg = SegmentedControl(seg_frame, list(range(2, 10)),
                                     self.k_var, on_change=self._on_k_change)
        self._seg.pack()

        self._k_display = tk.Label(sb, text="K = 3",
                                   font=(FONT, 22, "bold"),
                                   fg=C["accent"], bg=C["surface"])
        self._k_display.pack(pady=(4, 4))

        # Auto-refresh toggle
        ar_frame = tk.Frame(sb, bg=C["surface"])
        ar_frame.pack(fill="x", padx=14, pady=(0, 8))
        cb = tk.Checkbutton(ar_frame, text="Auto-analyse on K change",
                            variable=self._auto_refresh,
                            font=(FONT, 8), fg=C["text2"], bg=C["surface"],
                            activebackground=C["surface"],
                            selectcolor=C["surface"],
                            cursor="hand2")
        cb.pack(side="left")
        Tooltip(cb, "Automatically re-run clustering when K changes")

        # Analyse button
        btn_frame = tk.Frame(sb, bg=C["surface"])
        btn_frame.pack(padx=17, pady=(0, 10), anchor="w")
        PillButton(btn_frame, "  Analyse  ", self._run_clustering,
                   width=196, height=38).pack()

        Divider(sb).pack(fill="x", padx=14, pady=(0, 4))

        # ── Category filter ──
        SectionLabel(sb, "Filter").pack(fill="x")
        cat_frame = tk.Frame(sb, bg=C["surface"])
        cat_frame.pack(fill="x", padx=10, pady=(0, 6))
        self._cat_cb = ttk.Combobox(cat_frame, textvariable=self._cat_var,
                                     state="readonly", font=(FONT, 9), width=22)
        self._cat_cb.pack(fill="x", ipady=3)
        self._cat_cb.bind("<<ComboboxSelected>>", lambda _: self._apply_category_filter())
        Tooltip(self._cat_cb, "Filter products by category before clustering")

        Divider(sb).pack(fill="x", padx=14, pady=(4, 2))

        # ── Nav ──
        groups = {
            "Analysis": [
                ("Elbow + Silhouette", self._show_elbow,
                 "WCSS and silhouette score vs K"),
                ("PCA Scatter",        self._show_pca,
                 "2-D PCA projection coloured by cluster"),
                ("Centroid Heatmap",   self._show_heatmap,
                 "Standardised feature values per cluster centre"),
                ("Cluster Profiles",   self._show_profiles,
                 "Normalised feature comparison across clusters"),
            ],
            "Explore": [
                ("Rating vs Discount",    lambda: self._show_scatter("discount_percentage", "rating"),
                 "How rating varies with discount"),
                ("Price vs Discount",     lambda: self._show_scatter("discount_percentage", "discounted_price"),
                 "Price vs discount percentage by cluster"),
                ("Correlation",           self._show_correlation,
                 "Pairwise Pearson correlation heatmap"),
                ("Price Distribution",    lambda: self._show_dist("discounted_price"),
                 "Histogram of discounted prices"),
                ("Discount Distribution", lambda: self._show_dist("discount_percentage"),
                 "Histogram of discount percentages"),
                ("Top Categories",        self._show_categories,
                 "Best-rated product categories"),
                ("Segment Pie",           self._show_pie,
                 "Share of products per cluster"),
            ],
        }
        for group, items in groups.items():
            SectionLabel(sb, group).pack(fill="x")
            for label, cmd, tip in items:
                def _cmd(c=cmd, lbl=label):
                    self._activate_nav(lbl)
                    c()
                nav = NavItem(sb, label, _cmd, tooltip_text=tip)
                nav.pack(fill="x", padx=6)
                self._nav_items[label] = nav

        Divider(sb).pack(fill="x", padx=14, pady=(6, 0))

        # ── Stats cards ──
        cards = tk.Frame(sb, bg=C["surface"])
        cards.pack(fill="x", padx=10, pady=8)
        self._card_products = MetricCard(cards, "Products loaded")
        self._card_products.pack(fill="x", pady=(0, 6))
        self._card_sil = MetricCard(cards, "Silhouette score")
        self._card_sil.pack(fill="x", pady=(0, 6))
        self._card_clusters = tk.Label(cards, text="",
                                        font=(FONT, 8), fg=C["text2"],
                                        bg=C["surface"], justify="left",
                                        anchor="w", wraplength=200)
        self._card_clusters.pack(fill="x", padx=12, pady=(0, 6))

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=C["surface"],
                        highlightbackground=C["border"],
                        highlightthickness=1)
        main.pack(side="left", fill="both", expand=True, pady=12)

        # Tab bar
        self._tabbar = TabBar(main, ["Chart", "Data Table"],
                              self._on_tab_switch)
        self._tabbar.pack(fill="x")
        Divider(main).pack(fill="x")

        # View title
        self._view_title = tk.Label(main, text="",
                                    font=(FONT, 11, "bold"), fg=C["text1"],
                                    bg=C["surface"], anchor="w",
                                    padx=18, pady=10)
        self._view_title.pack(fill="x")
        Divider(main).pack(fill="x")

        # Chart pane
        self._chart_pane = tk.Frame(main, bg=C["surface"])
        self._chart_pane.pack(fill="both", expand=True)

        self._canvas_frame = tk.Frame(self._chart_pane, bg=C["surface"])
        self._canvas_frame.pack(fill="both", expand=True)

        # Data table pane (hidden initially)
        self._data_table = DataTable(main)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["surface"], height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        Divider(self).pack(fill="x", side="bottom")
        self._status_dot = tk.Label(bar, text="●", fg=C["success"],
                                     bg=C["surface"], font=(FONT, 9))
        self._status_dot.pack(side="left", padx=(14, 4))
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var,
                 font=(FONT, 8), fg=C["text2"],
                 bg=C["surface"], anchor="w").pack(side="left")

    # ── helpers ───────────────────────────────────────────────────────────
    def _status(self, msg, dot_color=C["warning"]):
        self._status_var.set(msg)
        self._status_dot.config(fg=dot_color)
        self.update_idletasks()

    def _activate_nav(self, label):
        if self._active_nav and self._active_nav in self._nav_items:
            self._nav_items[self._active_nav].set_active(False)
        self._active_nav = label
        if label in self._nav_items:
            self._nav_items[label].set_active(True)
        self._view_title.config(text=label)

    def _on_k_change(self, val=None):
        k = int(self.k_var.get())
        self._k_display.config(text=f"K = {k}")
        if self._auto_refresh.get() and self.X_scaled is not None:
            self._run_clustering()

    def _on_tab_switch(self, tab):
        self._active_tab = tab
        # Guard: called by TabBar.__init__ before panes exist
        if self._data_table is None or self._chart_pane is None:
            return
        if tab == "Chart":
            self._data_table.pack_forget()
            self._chart_pane.pack(fill="both", expand=True)
            self._view_title.pack(fill="x")
        else:
            self._chart_pane.pack_forget()
            self._view_title.pack_forget()
            self._data_table.pack(fill="both", expand=True)
            if self.df is not None and "cluster" in self.df.columns:
                self._data_table.load(self.df)

    # ── data loading ─────────────────────────────────────────────────────
    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Open Amazon CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._load_csv(path)

    def _auto_load(self):
        if os.path.exists(DEFAULT_CSV):
            self._load_csv(DEFAULT_CSV)

    def _load_csv(self, path):
        self._status(f"Loading {os.path.basename(path)} …")
        try:
            self.df       = load_and_clean(path)
            self.X_scaled = scale_features(self.df)
            self._card_products.update(f"{len(self.df):,}")
            # Populate category filter
            cats = ["All categories"] + sorted(self.df["primary_category"].unique().tolist())
            self._cat_cb["values"] = cats
            self._cat_var.set("All categories")
            self._status("Computing elbow data …")
            self._inertia, self._sil = elbow_data(self.X_scaled, self._k_range)
            self._run_clustering(show_elbow=True)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._status("Error loading file.", C["danger"])

    def _apply_category_filter(self):
        if self.df is None:
            return
        cat = self._cat_var.get()
        if cat == "All categories":
            filtered = self.df
        else:
            filtered = self.df[self.df["primary_category"] == cat]
        if len(filtered) < 10:
            messagebox.showwarning("Too few rows",
                                   f"Only {len(filtered)} products in '{cat}'. Select a larger category.")
            self._cat_var.set("All categories")
            return
        self.X_scaled = scale_features(filtered)
        self._card_products.update(f"{len(filtered):,}")
        self._status(f"Filtered to '{cat}'  ({len(filtered):,} products)  — re-computing …")
        self._inertia, self._sil = elbow_data(self.X_scaled, self._k_range)
        # Keep cluster column in sync with filtered df
        self.df = filtered.copy().reset_index(drop=True)
        self._run_clustering()

    # ── clustering ────────────────────────────────────────────────────────
    def _run_clustering(self, show_elbow=False):
        if self.X_scaled is None:
            messagebox.showwarning("No data", "Please load a CSV first.")
            return
        k = int(self.k_var.get())
        self._status(f"Running K-Means  K = {k} …")
        self.km, labels, sil = run_kmeans(self.X_scaled, k)
        self.df["cluster"]   = labels
        self._card_sil.update(f"{sil:.4f}")
        summary = self.df.groupby("cluster")[FEATURES].mean()
        lines   = []
        for c in range(k):
            r = summary.iloc[c]
            lines.append(f"C{c}  ₹{r['discounted_price']:,.0f}  "
                         f"{r['discount_percentage']:.0f}%  ★{r['rating']:.1f}")
        self._card_clusters.config(text="\n".join(lines))
        self._status(f"Done  ·  K = {k}  ·  Silhouette = {sil:.4f}", C["success"])
        # Refresh data table if visible
        if self._active_tab == "Data Table":
            self._data_table.load(self.df)
        if show_elbow:
            self._show_elbow()
        else:
            self._show_pca()

    # ── export ────────────────────────────────────────────────────────────
    def _export_csv(self):
        if self.df is None or "cluster" not in self.df.columns:
            messagebox.showwarning("No data", "Run Analyse first to assign clusters.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=OUTPUT_DIR,
            initialfile="clustered_products.csv")
        if path:
            self.df.to_csv(path, index=False)
            self._status(f"Exported  →  {os.path.basename(path)}", C["success"])

    # ── chart display ─────────────────────────────────────────────────────
    def _display_fig(self, fig):
        for w in self._canvas_frame.winfo_children():
            w.destroy()
        if self._current_fig:
            plt.close(self._current_fig)
        self._current_fig = fig

        canvas = FigureCanvasTkAgg(fig, master=self._canvas_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self._canvas_frame, pack_toolbar=False)
        toolbar.config(background=C["surface"])
        for child in toolbar.winfo_children():
            try:
                child.config(background=C["surface"])
            except Exception:
                pass
        toolbar.update()
        toolbar.pack(side="bottom", fill="x", padx=6, pady=4)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=(4, 0))

    def _save_plot(self):
        if self._current_fig is None:
            messagebox.showinfo("Nothing to save", "Open a chart view first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")],
            initialdir=OUTPUT_DIR)
        if path:
            self._current_fig.savefig(path, bbox_inches="tight", dpi=150,
                                      facecolor="white")
            self._status(f"Saved  →  {os.path.basename(path)}", C["success"])

    def _guard(self):
        if self.df is None or self.km is None:
            messagebox.showwarning("No clustering", "Load data and run Analyse first.")
            return False
        return True

    # ── view methods ──────────────────────────────────────────────────────
    def _show_elbow(self):
        if self._inertia is None:
            return
        self._activate_nav("Elbow + Silhouette")
        self._display_fig(fig_elbow(self._k_range, self._inertia,
                                    self._sil, int(self.k_var.get())))

    def _show_pca(self):
        if not self._guard(): return
        self._activate_nav("PCA Scatter")
        self._display_fig(fig_pca(self.X_scaled, self.df["cluster"].values,
                                   self.km, int(self.k_var.get())))

    def _show_heatmap(self):
        if not self._guard(): return
        self._activate_nav("Centroid Heatmap")
        self._display_fig(fig_heatmap(self.km, int(self.k_var.get())))

    def _show_profiles(self):
        if not self._guard(): return
        self._activate_nav("Cluster Profiles")
        self._display_fig(fig_profiles(self.df, int(self.k_var.get())))

    def _show_scatter(self, x_col, y_col):
        if not self._guard(): return
        self._display_fig(fig_scatter(self.df, x_col, y_col, int(self.k_var.get())))

    def _show_correlation(self):
        if self.df is None: return
        self._activate_nav("Correlation")
        self._display_fig(fig_correlation(self.df))

    def _show_dist(self, col):
        if self.df is None: return
        self._display_fig(fig_distribution(self.df, col))

    def _show_categories(self):
        if self.df is None: return
        self._activate_nav("Top Categories")
        self._display_fig(fig_top_categories(self.df))

    def _show_pie(self):
        if not self._guard(): return
        self._activate_nav("Segment Pie")
        self._display_fig(fig_segment_pie(self.df, int(self.k_var.get())))


if __name__ == "__main__":
    app = App()
    app.mainloop()
