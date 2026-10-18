"""
src/agent/plot_tools.py

Lightweight visualization utilities.

Purpose:
- Convert SQL query results into static charts
- Render plots off-screen (server-safe)
- Return images as Base64-encoded PNGs for UI transport

Design principles:
- Dataset-agnostic
- No database or LLM dependencies
- Deterministic and side-effect free
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import base64
from io import BytesIO
from typing import List
from functools import wraps
from threading import Lock
from datetime import datetime
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

_plot_lock = Lock()


def serialized_plot(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with _plot_lock:
            return fn(*args, **kwargs)
    return wrapped


def _shorten_labels(labels: List[str], max_len: int = 14) -> List[str]:
    """Shorten long x-axis labels safely."""
    out = []
    for l in labels:
        if len(l) <= max_len:
            out.append(l)
        else:
            out.append(l[: max_len - 3] + "...")
    return out


def _finalize_plot(title: str, xlabel: str, ylabel: str, ys=None, temporal=False):
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    ax = plt.gca()
    if "count" in ylabel.lower() or "number" in ylabel.lower():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ys and max(abs(y) for y in ys) >= 1000:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if temporal:
        first, last = (mdates.num2date(value) for value in ax.get_xlim())
        months = (last.year - first.year) * 12 + last.month - first.month
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, (months + 9) // 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


@serialized_plot
def plot_bar_png_base64(
    xs: List[str],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    width = max(8, min(len(xs) * 0.6, 18))
    plt.figure(figsize=(width, 5))

    xs_short = _shorten_labels(xs)

    plt.bar(xs_short, ys)
    _finalize_plot(title, xlabel, ylabel, ys=ys)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@serialized_plot
def plot_line_png_base64(
    xs: List[str],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    plt.figure(figsize=(10, 5))

    temporal = bool(xs) and all(re.fullmatch(r"\d{4}-\d{2}", x) for x in xs)
    plot_xs = [datetime.strptime(x, "%Y-%m") for x in xs] if temporal else _shorten_labels(xs)
    plt.plot(plot_xs, ys, marker="o")
    _finalize_plot(title, xlabel, ylabel, ys=ys, temporal=temporal)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

