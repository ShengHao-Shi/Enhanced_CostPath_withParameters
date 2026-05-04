"""
Generate comparison figures for the README.

Run from the repository root:

    python docs/generate_readme_figures.py

Output files are written to docs/images/.

Requires: numpy, matplotlib
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Ensure repo root is on sys.path so package imports work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from pure_python.cost_aware_straighten_lcp import cost_aware_least_cost_path

_OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared raster
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
ROWS, COLS = 60, 80
START = (5, 5)
END = (54, 74)

# Base cost surface: smooth Perlin-like noise using sums of sinusoids
_y = np.linspace(0, 4 * np.pi, ROWS)
_x = np.linspace(0, 4 * np.pi, COLS)
_xx, _yy = np.meshgrid(_x, _y)
cost_surface = (
    2.0
    + 1.5 * np.sin(_yy * 0.7 + 1.3)
    + 1.5 * np.cos(_xx * 0.9 + 0.8)
    + 0.8 * np.sin(_xx * 1.7 + _yy * 1.1)
    + RNG.uniform(0.0, 0.4, (ROWS, COLS))
)
cost_surface = np.clip(cost_surface, 0.5, 10.0).astype(np.float32)

# Add a "ridge" of high cost to make routing decisions visible
cost_surface[22:38, 25:45] = 8.5


def _path_xy(path):
    """Convert list of (row, col) to (x_array, y_array) for plotting."""
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    return cols, rows  # note: x = col, y = row


def _plot_base(ax, title):
    """Draw cost surface background and return the image for a colorbar."""
    im = ax.imshow(
        cost_surface,
        cmap="YlOrRd",
        vmin=0.5,
        vmax=10.0,
        origin="upper",
        aspect="equal",
        alpha=0.85,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _mark_endpoints(ax):
    ax.plot(START[1], START[0], "go", ms=7, zorder=5, label="Start")
    ax.plot(END[1], END[0], "b^", ms=7, zorder=5, label="End")


# ---------------------------------------------------------------------------
# Figure 1: side-by-side path comparison
# ---------------------------------------------------------------------------
def fig_comparison():
    """Four panels: native LCP | + curvature | + distance | + straighten."""

    configs = [
        dict(
            label="Standard LCP\n(no parameters)",
            curvature_factor=0.0,
            max_turning_angle=180.0,
            distance_factor=0.0,
            straighten_factor=0.0,
            cost_tolerance=1.05,
            path_color="#e74c3c",
        ),
        dict(
            label="+ curvature_factor = 0.7\n(smooth turns)",
            curvature_factor=0.7,
            max_turning_angle=180.0,
            distance_factor=0.0,
            straighten_factor=0.0,
            cost_tolerance=1.05,
            path_color="#9b59b6",
        ),
        dict(
            label="+ distance_factor = 0.4\n(shorter detours)",
            curvature_factor=0.0,
            max_turning_angle=180.0,
            distance_factor=0.4,
            straighten_factor=0.0,
            cost_tolerance=1.05,
            path_color="#2980b9",
        ),
        dict(
            label="All parameters combined\n(curvature + distance + straighten)",
            curvature_factor=0.5,
            max_turning_angle=135.0,
            distance_factor=0.3,
            straighten_factor=0.3,
            cost_tolerance=1.05,
            path_color="#27ae60",
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "Enhanced Cost Path — Effect of Parameters",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, cfg in zip(axes, configs):
        result = cost_aware_least_cost_path(
            cost_surface,
            START,
            END,
            curvature_factor=cfg["curvature_factor"],
            max_turning_angle=cfg["max_turning_angle"],
            distance_factor=cfg["distance_factor"],
            straighten_factor=cfg["straighten_factor"],
            cost_tolerance=cfg["cost_tolerance"],
        )
        _plot_base(ax, cfg["label"])

        # Raw grid path (thin, faint)
        rx, ry = _path_xy(result["path"])
        ax.plot(rx, ry, color=cfg["path_color"], alpha=0.3, lw=0.8)

        # Smoothed path (thick)
        sx, sy = _path_xy(result["smoothed_path"])
        ax.plot(sx, sy, color=cfg["path_color"], lw=2.2, label="Smoothed path")

        _mark_endpoints(ax)

        # Stats annotation
        n_turns = result["turning_angles"]
        max_turn = max(n_turns) if n_turns else 0.0
        ax.text(
            0.02,
            0.02,
            f"pts: {len(result['path'])}\nmax turn: {max_turn:.0f}°",
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    # Shared legend
    handles = [
        mpatches.Patch(color="#cccccc", alpha=0.4, label="Grid path (raw)"),
        plt.Line2D([0], [0], color="gray", lw=2.2, label="Smoothed path"),
        plt.Line2D([0], [0], marker="o", color="g", ls="", ms=7, label="Start"),
        plt.Line2D([0], [0], marker="^", color="b", ls="", ms=7, label="End"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    out = os.path.join(_OUT_DIR, "comparison_parameters.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: algorithm pipeline illustration
# ---------------------------------------------------------------------------
def fig_pipeline():
    """Three-panel figure showing raw grid path → straightened → smoothed."""
    result = cost_aware_least_cost_path(
        cost_surface,
        START,
        END,
        curvature_factor=0.3,
        max_turning_angle=135.0,
        distance_factor=0.2,
        straighten_factor=0.3,
        cost_tolerance=1.05,
    )

    labels = [
        ("Step 1 — Grid Path\n(8-dir Dijkstra output)", result["path"], "#e74c3c"),
        (
            "Step 2 — Straightened Path\n(cost-aware LOS shortcutting)",
            result["straightened_path"],
            "#e67e22",
        ),
        (
            "Step 3 — Smoothed Path\n(Chaikin corner-cutting)",
            result["smoothed_path"],
            "#27ae60",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Algorithm Pipeline: Grid Path → Straightened → Smoothed",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    for ax, (title, path, color) in zip(axes, labels):
        _plot_base(ax, title)
        px, py = _path_xy(path)
        ax.plot(px, py, color=color, lw=2.0)
        _mark_endpoints(ax)

        ax.text(
            0.02,
            0.02,
            f"waypoints: {len(path)}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="g", ls="", ms=7, label="Start"),
        plt.Line2D([0], [0], marker="^", color="b", ls="", ms=7, label="End"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    out = os.path.join(_OUT_DIR, "pipeline_steps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: cost tolerance effect
# ---------------------------------------------------------------------------
def fig_cost_tolerance():
    """Show how cost_tolerance controls the trade-off between straightness and cost."""
    tolerances = [1.0, 1.05, 1.2, 2.0]
    colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]
    labels = ["1.0 (strict)", "1.05 (default)", "1.20", "2.0 (loose)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "Effect of cost_tolerance on Straightening\n"
        "(higher = more aggressive straightening, may cross costly cells)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    for ax, tol, color, lbl in zip(axes, tolerances, colors, labels):
        result = cost_aware_least_cost_path(
            cost_surface,
            START,
            END,
            straighten_factor=0.4,
            cost_tolerance=tol,
        )
        _plot_base(ax, f"cost_tolerance = {lbl}")

        rx, ry = _path_xy(result["path"])
        ax.plot(rx, ry, color=color, alpha=0.25, lw=0.8)

        sx, sy = _path_xy(result["straightened_path"])
        ax.plot(sx, sy, color=color, lw=2.2)

        _mark_endpoints(ax)

        n_wp = len(result["straightened_path"])
        ax.text(
            0.02, 0.02,
            f"waypoints: {n_wp}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    out = os.path.join(_OUT_DIR, "cost_tolerance_effect.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating README figures …")
    fig_comparison()
    fig_pipeline()
    fig_cost_tolerance()
    print("Done. Images saved to docs/images/")
