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

import matplotlib.pyplot as plt



def _shorten_labels(labels: List[str], max_len: int = 14) -> List[str]:
    """Shorten long x-axis labels safely."""
    out = []
    for l in labels:
        if len(l) <= max_len:
            out.append(l)
        else:
            out.append(l[: max_len - 3] + "...")
    return out


def _finalize_plot(title: str, xlabel: str, ylabel: str):
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()


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
    _finalize_plot(title, xlabel, ylabel)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_line_png_base64(
    xs: List[str],
    ys: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    plt.figure(figsize=(10, 5))

    xs_short = _shorten_labels(xs)

    plt.plot(xs_short, ys, marker="o")
    _finalize_plot(title, xlabel, ylabel)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

