#!/usr/bin/env python3
"""
Find a minimum-energy path (MEP) on a 2D energy surface using the string method.

Input energy file format:
    X   Y   Energy

Blank lines are ignored. Lines whose first non-space character is '#' or '@' are
ignored. Inline comments beginning with '#' or '@' are also ignored.

Example:
    python string_mep_from_energy.py \
        -f energy.dat \
        -end [0.67,15.3] [2.5,21.9] \
        -path [4,4] [18,9] [13.83,17.1] \
        --n-points 80 --n-iter 3000 --dt 0.05 --smooth 0.5

Notes:
    - The two -end points are always fixed.
    - The optional -path points are initial-guess waypoints. If the first and
      last -path points are not the same as the endpoints, the endpoints are
      automatically prepended/appended.
    - If -path is not given, a linear initial string is used between endpoints.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter



@dataclass
class Surface:
    x_grid: np.ndarray
    y_grid: np.ndarray
    energy: np.ndarray          # shape: (n_y, n_x)
    energy_interp: RegularGridInterpolator
    grad_x_interp: RegularGridInterpolator
    grad_y_interp: RegularGridInterpolator
    x_min: float
    x_max: float
    y_min: float
    y_max: float


# ------------------ INPUT PARSING ------------------
def parse_point(text: str) -> Tuple[float, float]:
    """Parse one point written as '[x,y]', '(x,y)', 'x,y', or 'x y'."""
    cleaned = text.strip().strip("[](){}")
    parts = [p for p in re.split(r"[,:;\s]+", cleaned) if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Could not parse point {text!r}. Use a format like [1.0,2.0]."
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Could not parse numeric values in point {text!r}."
        ) from exc


def same_point(p: Sequence[float], q: Sequence[float], atol: float = 1e-10) -> bool:
    return bool(np.allclose(np.asarray(p, dtype=float), np.asarray(q, dtype=float), atol=atol, rtol=0.0))


def load_energy_file(filename: str) -> np.ndarray:
    """
    Load an energy.dat-like file with at least three columns: X, Y, Energy.

    Skips blank rows and rows beginning with '#' or '@'. Supports whitespace or
    comma separated columns. Inline comments starting with '#' or '@' are removed.
    """
    rows: List[Tuple[float, float, float]] = []
    with open(filename, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue

            # Remove inline comments/metainfo after actual data.
            line = re.split(r"[#@]", line, maxsplit=1)[0].strip()
            if not line:
                continue

            parts = [p for p in re.split(r"[,\s]+", line) if p]
            if len(parts) < 3:
                raise ValueError(
                    f"Line {line_number} of {filename!r} has fewer than three columns: {raw_line.rstrip()!r}"
                )
            try:
                x, y, e = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} of {filename!r} contains non-numeric data: {raw_line.rstrip()!r}"
                ) from exc

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(e):
                rows.append((x, y, e))

    if len(rows) < 4:
        raise ValueError(f"Need at least four finite X,Y,Energy rows in {filename!r}.")

    return np.asarray(rows, dtype=float)


def average_duplicate_points(data: np.ndarray) -> np.ndarray:
    """Average energy values for duplicate (x, y) coordinates."""
    accumulator = {}
    for x, y, e in data:
        key = (float(x), float(y))
        if key not in accumulator:
            accumulator[key] = [0.0, 0]
        accumulator[key][0] += float(e)
        accumulator[key][1] += 1

    averaged = [(x, y, total / count) for (x, y), (total, count) in accumulator.items()]
    return np.asarray(averaged, dtype=float)


# ------------------ SURFACE CONSTRUCTION ------------------
def build_surface(
    data: np.ndarray,
    grid_size: int | None = None,
    smooth_sigma: float = 0.0,
    shift_min_to_zero: bool = True,
) -> Surface:
    """
    Build a regular 2D grid and interpolation objects from X,Y,Energy data.

    If data lie on a complete rectangular grid, that grid is used directly.
    Otherwise, scattered data are interpolated to a regular grid using linear
    interpolation plus nearest-neighbour filling for missing values.
    """
    data = average_duplicate_points(data)
    x = data[:, 0]
    y = data[:, 1]
    e = data[:, 2]

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    expected_grid_points = len(unique_x) * len(unique_y)
    observed_points = len(data)

    is_complete_grid = observed_points == expected_grid_points
    if is_complete_grid:
        index_x = {value: idx for idx, value in enumerate(unique_x)}
        index_y = {value: idx for idx, value in enumerate(unique_y)}
        energy_grid = np.full((len(unique_y), len(unique_x)), np.nan, dtype=float)
        for xi, yi, ei in data:
            energy_grid[index_y[yi], index_x[xi]] = ei
        is_complete_grid = np.all(np.isfinite(energy_grid))
    else:
        energy_grid = np.empty((0, 0), dtype=float)

    if is_complete_grid:
        x_grid = unique_x
        y_grid = unique_y
        print(f"Detected complete rectangular grid: {len(x_grid)} x {len(y_grid)}")
    else:
        if grid_size is None:
            grid_size = max(100, min(400, int(np.sqrt(observed_points))))
        x_grid = np.linspace(np.min(x), np.max(x), grid_size)
        y_grid = np.linspace(np.min(y), np.max(y), grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)

        print(
            "Input is not a complete rectangular grid; "
            f"interpolating scattered data to {grid_size} x {grid_size}."
        )
        energy_linear = griddata((x, y), e, (X, Y), method="linear")
        energy_nearest = griddata((x, y), e, (X, Y), method="nearest")
        energy_grid = np.where(np.isfinite(energy_linear), energy_linear, energy_nearest)

    if shift_min_to_zero:
        energy_grid = energy_grid - np.nanmin(energy_grid)

    if smooth_sigma > 0.0:
        energy_grid = gaussian_filter(energy_grid, sigma=smooth_sigma)
        if shift_min_to_zero:
            energy_grid = energy_grid - np.nanmin(energy_grid)

    dZdy, dZdx = np.gradient(energy_grid, y_grid, x_grid)

    energy_interp = RegularGridInterpolator(
        (y_grid, x_grid),
        energy_grid,
        method="linear",
        bounds_error=False,
        fill_value=float(np.nanmax(energy_grid)),
    )
    grad_x_interp = RegularGridInterpolator(
        (y_grid, x_grid),
        dZdx,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    grad_y_interp = RegularGridInterpolator(
        (y_grid, x_grid),
        dZdy,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    return Surface(
        x_grid=x_grid,
        y_grid=y_grid,
        energy=energy_grid,
        energy_interp=energy_interp,
        grad_x_interp=grad_x_interp,
        grad_y_interp=grad_y_interp,
        x_min=float(x_grid[0]),
        x_max=float(x_grid[-1]),
        y_min=float(y_grid[0]),
        y_max=float(y_grid[-1]),
    )


# ------------------ PATH INITIALISATION ------------------
def clip_path(path: np.ndarray, surface: Surface) -> np.ndarray:
    clipped = np.asarray(path, dtype=float).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], surface.x_min, surface.x_max)
    clipped[:, 1] = np.clip(clipped[:, 1], surface.y_min, surface.y_max)
    return clipped


def build_waypoints(endpoints: Sequence[Tuple[float, float]], path_points: Sequence[Tuple[float, float]] | None) -> np.ndarray:
    """
    Construct waypoints. Endpoints are always the fixed start/end.

    If -path is omitted, use only endpoints. If -path includes endpoints already,
    do not duplicate them. Otherwise treat -path points as intermediate hints.
    """
    start = np.asarray(endpoints[0], dtype=float)
    end = np.asarray(endpoints[1], dtype=float)

    if not path_points:
        return np.vstack([start, end])

    path = [np.asarray(p, dtype=float) for p in path_points]

    waypoints: List[np.ndarray] = []
    if same_point(path[0], start):
        waypoints.extend(path)
    else:
        waypoints.append(start)
        waypoints.extend(path)

    if not same_point(waypoints[-1], end):
        waypoints.append(end)

    return np.vstack(waypoints)


def build_initial_path(waypoints: np.ndarray, n_points: int, surface: Surface) -> np.ndarray:
    """
    Piecewise-linear initial path through waypoints with uniform bead spacing.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("Waypoints must have shape (n_waypoints, 2).")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    diffs = np.diff(waypoints, axis=0)
    chords = np.sqrt(np.sum(diffs**2, axis=1))
    if np.any(chords < 1e-14):
        raise ValueError("Two consecutive waypoints are identical or almost identical.")

    arc_waypoints = np.insert(np.cumsum(chords), 0, 0.0)
    arc_waypoints /= arc_waypoints[-1]
    arc_beads = np.linspace(0.0, 1.0, n_points)

    path = np.column_stack([
        np.interp(arc_beads, arc_waypoints, waypoints[:, 0]),
        np.interp(arc_beads, arc_waypoints, waypoints[:, 1]),
    ])
    return clip_path(path, surface)


def reparameterise(path: np.ndarray, n_points: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Redistribute beads with equal arc-length spacing; only endpoints are fixed."""
    dists = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    arc = np.insert(np.cumsum(dists), 0, 0.0)
    if arc[-1] < 1e-14:
        new_path = path.copy()
        new_path[0] = start
        new_path[-1] = end
        return new_path

    arc /= arc[-1]
    target = np.linspace(0.0, 1.0, n_points)
    new_path = np.column_stack([
        np.interp(target, arc, path[:, 0]),
        np.interp(target, arc, path[:, 1]),
    ])
    new_path[0] = start
    new_path[-1] = end
    return new_path


def path_energy(path: np.ndarray, surface: Surface) -> np.ndarray:
    """Evaluate energy on path. Path is stored as (x,y); interpolator expects (y,x)."""
    query = np.column_stack([
        np.clip(path[:, 1], surface.y_min, surface.y_max),
        np.clip(path[:, 0], surface.x_min, surface.x_max),
    ])
    return surface.energy_interp(query)


def path_arc_length(path: np.ndarray) -> np.ndarray:
    dists = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    return np.insert(np.cumsum(dists), 0, 0.0)


# ------------------ STRING METHOD ------------------
def string_method(
    waypoints: np.ndarray,
    surface: Surface,
    n_points: int = 80,
    n_iter: int = 3000,
    dt: float = 0.05,
    convergence_tol: float = 1e-6,
    check_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the simple string method on the interpolated energy surface.

    First and last waypoints are fixed endpoints. Middle waypoints only define
    the initial path and then evolve freely.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    start = waypoints[0].copy()
    end = waypoints[-1].copy()

    string = build_initial_path(waypoints, n_points, surface)
    initial_path = string.copy()
    previous_string = string.copy()

    for iteration in range(1, n_iter + 1):
        query = np.column_stack([
            np.clip(string[:, 1], surface.y_min, surface.y_max),
            np.clip(string[:, 0], surface.x_min, surface.x_max),
        ])
        gx = surface.grad_x_interp(query)
        gy = surface.grad_y_interp(query)

        string[1:-1, 0] -= dt * gx[1:-1]
        string[1:-1, 1] -= dt * gy[1:-1]
        string = clip_path(string, surface)
        string = reparameterise(string, n_points, start, end)
        string = clip_path(string, surface)

        # Re-enforce exact fixed endpoints after clipping/reparameterisation.
        string[0] = start
        string[-1] = end

        if check_every > 0 and iteration % check_every == 0:
            max_disp = np.max(np.linalg.norm(string - previous_string, axis=1))
            energies = path_energy(string, surface)
            print(
                f"  iter {iteration:6d} | max disp: {max_disp:.3e} "
                f"| max E on path: {np.max(energies):.6g}"
            )
            if max_disp < convergence_tol:
                print(f"Converged at iteration {iteration}.")
                break
            previous_string = string.copy()

    return initial_path, string


# ------------------ OUTPUT ------------------
def save_path_data(path: np.ndarray, energy: np.ndarray, filename: str) -> None:
    arc = path_arc_length(path)
    energy_shifted = energy - np.min(energy)
    bead_index = np.arange(1, len(path) + 1)
    output = np.column_stack([bead_index, path[:, 0], path[:, 1], arc, energy, energy_shifted])
    header = "index  X  Y  arc_length  Energy  Energy_minus_min"
    np.savetxt(filename, output, fmt=["%d", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f"], header=header)


def plot_results(
    surface: Surface,
    initial_path: np.ndarray,
    mep: np.ndarray,
    waypoints: np.ndarray,
    prefix: str,
    show_initial: bool = True,
    xlabel: str = "X",
    ylabel: str = "Y",
    energy_label: str = r"$F/k_BT$",
    dpi: int = 300,
) -> None:
    mep_energy = path_energy(mep, surface)
    mep_energy_shifted = mep_energy - np.min(mep_energy)
    bead_index = np.arange(1, len(mep) + 1)

    X, Y = np.meshgrid(surface.x_grid, surface.y_grid)

    # Combined figure: surface with path + energy profile along path.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(2.0)
    cf = ax.contourf(X, Y, surface.energy, levels=30, cmap='brg')

    if show_initial:
        ax.plot(initial_path[:, 0], initial_path[:, 1], "w--", lw=1.5, alpha=0.9, label="Initial path")
    ax.plot(mep[:, 0], mep[:, 1], "k-o", lw=1.2, markersize=4, label="MEP")

    # Plot user-provided waypoints.
    if len(waypoints) > 2:
        ax.scatter(waypoints[1:-1, 0], waypoints[1:-1, 1], c="cyan", s=70,
                   edgecolors="black", zorder=6, label="Initial hints")
    ax.scatter(waypoints[0, 0], waypoints[0, 1], c="white", s=120,
               edgecolors="black", zorder=7, label="Start")
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], c="yellow", s=120,
               edgecolors="black", zorder=7, label="End")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel(energy_label, fontsize=18, rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=14)
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14, direction="out", length=8, width=1.5)
    ax.tick_params(axis="both", which="minor", direction="out", length=4, width=1.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=6)
    ax.legend(frameon=True, fontsize=11, loc="best")

    ax2 = axes[1]
    ax2.patch.set_edgecolor("black")
    ax2.patch.set_linewidth(2.0)
    ax2.plot(bead_index, mep_energy_shifted, "k-o", markersize=4, lw=1.8)
    ax2.set_xlabel("Path point", fontsize=18)
    ax2.set_ylabel(energy_label, fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=14, direction="out", length=8, width=1.5)
    ax2.tick_params(axis="both", which="minor", direction="out", length=4, width=1.2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.locator_params(axis="x", nbins=6)
    ax2.locator_params(axis="y", nbins=6)

    plt.tight_layout()
    fig.savefig(f"{prefix}_surface_and_energy.png", bbox_inches="tight", dpi=dpi)
    fig.savefig(f"{prefix}_surface_and_energy.svg", bbox_inches="tight")
    plt.close(fig)

    # Separate path-on-surface figure.
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, surface.energy, levels=30, cmap='brg')
    if show_initial:
        ax.plot(initial_path[:, 0], initial_path[:, 1], "w--", lw=1.5, alpha=0.9, label="Initial path")
    ax.plot(mep[:, 0], mep[:, 1], "k-o", lw=1.2, markersize=4, label="MEP")
    ax.scatter(waypoints[0, 0], waypoints[0, 1], c="white", s=120, edgecolors="black", zorder=7)
    ax.scatter(waypoints[-1, 0], waypoints[-1, 1], c="yellow", s=120, edgecolors="black", zorder=7)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel(energy_label, fontsize=16, rotation=270, labelpad=22)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()
    fig.savefig(f"{prefix}_path_on_surface.png", bbox_inches="tight", dpi=dpi)
    #fig.savefig(f"{prefix}_path_on_surface.svg", bbox_inches="tight")
    plt.close(fig)

    # Separate energy-profile figure.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bead_index, mep_energy_shifted, "k-o", markersize=4, lw=1.8)
    ax.set_xlabel("Path point", fontsize=16)
    ax.set_ylabel(energy_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.tight_layout()
    fig.savefig(f"{prefix}_energy_profile.png", bbox_inches="tight", dpi=dpi)
    #fig.savefig(f"{prefix}_energy_profile.svg", bbox_inches="tight")
    plt.close(fig)


def write_surface_grid(surface: Surface, filename: str) -> None:
    X, Y = np.meshgrid(surface.x_grid, surface.y_grid)
    output = np.column_stack([X.ravel(), Y.ravel(), surface.energy.ravel()])
    np.savetxt(filename, output, fmt="%.10f", header="X  Y  Energy")


# ------------------ CLI ------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a minimum-energy path on a 2D energy surface using the string method."
    )
    parser.add_argument("-f", "--file", default="energy.dat", help="Input file with columns X Y Energy. Default: energy.dat")
    parser.add_argument(
        "-end",
        "--endpoints",
        required=True,
        nargs=2,
        type=parse_point,
        metavar=("START", "END"),
        help="Fixed endpoints, e.g. -end [0.67,15.3] [2.5,21.9]",
    )
    parser.add_argument(
        "-path",
        "--path",
        nargs="*",
        type=parse_point,
        default=None,
        metavar="POINT",
        help=(
            "Optional initial path waypoints, e.g. -path [4,4] [18,9]. "
            "If omitted, the initial path is linear between endpoints."
        ),
    )
    parser.add_argument("--n-points", type=int, default=80, help="Number of beads on the string. Default: 80")
    parser.add_argument("--n-iter", type=int, default=3000, help="Maximum string iterations. Default: 3000")
    parser.add_argument("--dt", type=float, default=0.05, help="Gradient-descent step size. Default: 0.05")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance for max bead displacement. Default: 1e-6")
    parser.add_argument("--check-every", type=int, default=200, help="Print/check convergence every N iterations. Default: 200")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Grid size used only for scattered/non-rectangular input data. Default: automatic.",
    )
    parser.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing sigma applied to the energy grid. Default: 0")
    parser.add_argument("--prefix", default="MEP", help="Prefix for output files. Default: MEP")
    parser.add_argument("--xlabel", default="X", help="X-axis label for plots. Default: X")
    parser.add_argument("--ylabel", default="Y", help="Y-axis label for plots. Default: Y")
    parser.add_argument("--energy-label", default=r"$F/k_BT$", help=r"Energy label for plots. Default: $F/k_BT$")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG figures. Default: 300")
    parser.add_argument("--no-initial", action="store_true", help="Do not draw the initial path on figures.")
    parser.add_argument("--save-grid", action="store_true", help="Also save the regularized/smoothed energy grid used internally.")
    parser.add_argument("--no-shift", action="store_true", help="Do not shift the minimum surface energy to zero.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading energy surface from: {args.file}")
    data = load_energy_file(args.file)
    print(f"Loaded {len(data)} finite data points.")

    surface = build_surface(
        data,
        grid_size=args.grid_size,
        smooth_sigma=args.smooth,
        shift_min_to_zero=not args.no_shift,
    )

    waypoints = build_waypoints(args.endpoints, args.path)
    waypoints = clip_path(waypoints, surface)

    print("Fixed endpoints:")
    print(f"  start = ({waypoints[0, 0]:.8g}, {waypoints[0, 1]:.8g})")
    print(f"  end   = ({waypoints[-1, 0]:.8g}, {waypoints[-1, 1]:.8g})")
    print(f"Number of initial waypoints including endpoints: {len(waypoints)}")
    print(f"Running string method with {args.n_points} beads, {args.n_iter} max iterations, dt={args.dt:g}")

    initial_path, mep = string_method(
        waypoints=waypoints,
        surface=surface,
        n_points=args.n_points,
        n_iter=args.n_iter,
        dt=args.dt,
        convergence_tol=args.tol,
        check_every=args.check_every,
    )

    initial_energy = path_energy(initial_path, surface)
    mep_energy = path_energy(mep, surface)

    os.makedirs(os.path.dirname(args.prefix), exist_ok=True) if os.path.dirname(args.prefix) else None

    save_path_data(initial_path, initial_energy, f"{args.prefix}_initial_path.dat")
    save_path_data(mep, mep_energy, f"{args.prefix}_path.dat")
    np.savetxt(f"{args.prefix}_waypoints.dat", waypoints, fmt="%.10f", header="X  Y")

    if args.save_grid:
        write_surface_grid(surface, f"{args.prefix}_surface_grid.dat")

    plot_results(
        surface=surface,
        initial_path=initial_path,
        mep=mep,
        waypoints=waypoints,
        prefix=args.prefix,
        show_initial=not args.no_initial,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        energy_label=args.energy_label,
        dpi=args.dpi,
    )

    barrier = float(np.max(mep_energy) - np.min(mep_energy))
    print("Done.")
    print(f"MEP barrier along saved path: {barrier:.6g}")
    print("Saved files:")
    print(f"  {args.prefix}_path.dat")
    print(f"  {args.prefix}_initial_path.dat")
    print(f"  {args.prefix}_waypoints.dat")
    print(f"  {args.prefix}_surface_and_energy.png / .svg")
    print(f"  {args.prefix}_path_on_surface.png / .svg")
    print(f"  {args.prefix}_energy_profile.png / .svg")
    if args.save_grid:
        print(f"  {args.prefix}_surface_grid.dat")


if __name__ == "__main__":
    main()
