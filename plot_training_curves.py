"""Utility for visualizing GRPO learning and reward curves."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning and reward curves from GRPO logs.")
    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=None,
        help="Path to trainer_state.json. Defaults to the newest file under grpo_runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV and SVG artifacts.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply exponential moving average smoothing to the plots.",
    )
    return parser.parse_args()


def locate_trainer_state(candidate: Path | None) -> Path:
    if candidate is not None:
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No trainer_state.json at {candidate}")
    grpo_dir = Path("grpo_runs")
    if not grpo_dir.exists():
        raise FileNotFoundError("grpo_runs directory not found.")
    matches = sorted(grpo_dir.rglob("trainer_state.json"))
    if not matches:
        raise FileNotFoundError("Could not locate trainer_state.json under grpo_runs/.")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_log_history(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    history = data.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"trainer_state log_history is not a list in {path}")
    return history  # type: ignore[return-value]


def extract_learning_rows(history: Sequence[Dict]) -> List[Dict[str, float]]:
    learning_rows: List[Dict[str, float]] = []
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        row: Dict[str, float] = {"step": step}
        for key in ("epoch", "loss", "reward", "reward_std", "entropy", "grad_norm", "learning_rate"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                row[key] = float(value)
        if len(row) > 1:
            learning_rows.append(row)
    return learning_rows


def extract_reward_series(history: Sequence[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    reward_series: Dict[str, List[Tuple[float, float]]] = {}
    for entry in history:
        step = entry.get("step")
        if step is None:
            continue
        for key, value in entry.items():
            if not isinstance(value, (int, float)):
                continue
            if not key.startswith("rewards/") or not key.endswith("/mean"):
                continue
            reward_name = key.split("/")[1]
            reward_name = reward_name.removesuffix("_reward")
            reward_series.setdefault(reward_name, []).append((step, float(value)))
    return reward_series


def write_csv(rows: Sequence[Dict[str, float]], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_reward_long_csv(reward_series: Dict[str, List[Tuple[float, float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reward_name", "step", "value"])
        writer.writeheader()
        for reward_name, points in sorted(reward_series.items()):
            for step, value in points:
                writer.writerow({"reward_name": reward_name, "step": step, "value": value})


@dataclass
class Series:
    label: str
    points: List[Tuple[float, float]]
    color: str


def _scale_points(
    points: Iterable[Tuple[float, float]],
    width: float,
    height: float,
    margin: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> List[Tuple[float, float]]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    scaled = []
    for x_val, y_val in points:
        x_norm = (x_val - x_min) / x_span
        y_norm = (y_val - y_min) / y_span
        x_px = margin + x_norm * plot_width
        y_px = height - margin - y_norm * plot_height
        scaled.append((x_px, y_px))
    return scaled


def _format_tick(value: float) -> str:
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.2e}"
    if abs(value - int(value)) < 1e-6:
        return f"{int(value)}"
    return f"{value:.2f}"


def render_svg(
    series: Sequence[Series],
    path: Path,
    title: str,
    x_label: str = "Step",
    y_label: str = "Value",
    y_bounds: Tuple[float, float] | None = None,
) -> None:
    if not series:
        raise ValueError(f"No data available for {title}")
    x_values = [x for s in series for x, _ in s.points]
    y_values = [y for s in series for _, y in s.points]
    if not x_values or not y_values:
        raise ValueError(f"Missing values for {title}")
    x_min, x_max = min(x_values), max(x_values)
    y_min = min(y_values) if y_bounds is None else y_bounds[0]
    y_max = max(y_values) if y_bounds is None else y_bounds[1]
    if y_min == y_max:
        delta = max(abs(y_min), 1.0)
        y_min -= 0.1 * delta
        y_max += 0.1 * delta
    width, height, margin = 1100.0, 600.0, 70.0
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="35" text-anchor="middle" font-size="20">{title}</text>',
    ]
    # Axes
    x_axis_y = height - margin
    svg_lines.append(
        f'<line x1="{margin}" y1="{x_axis_y}" x2="{width - margin}" y2="{x_axis_y}" stroke="#333" stroke-width="1"/>'
    )
    svg_lines.append(
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{x_axis_y}" stroke="#333" stroke-width="1"/>'
    )

    # Ticks
    tick_count = 6
    for idx in range(tick_count):
        frac = idx / (tick_count - 1)
        x_val = x_min + frac * (x_max - x_min)
        x_px = margin + frac * (width - 2 * margin)
        svg_lines.append(
            f'<line x1="{x_px:.2f}" y1="{x_axis_y}" x2="{x_px:.2f}" y2="{x_axis_y+5}" stroke="#333" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x_px:.2f}" y="{x_axis_y+20}" text-anchor="middle" font-size="12">{_format_tick(x_val)}</text>'
        )
    for idx in range(tick_count):
        frac = idx / (tick_count - 1)
        y_val = y_min + frac * (y_max - y_min)
        y_px = height - margin - frac * (height - 2 * margin)
        svg_lines.append(
            f'<line x1="{margin-5}" y1="{y_px:.2f}" x2="{margin}" y2="{y_px:.2f}" stroke="#333" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{margin-10}" y="{y_px+4:.2f}" text-anchor="end" font-size="12">{_format_tick(y_val)}</text>'
        )

    svg_lines.append(
        f'<text x="{width/2:.1f}" y="{height-15:.1f}" text-anchor="middle" font-size="14">{x_label}</text>'
    )
    svg_lines.append(
        f'<text x="20" y="{height/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90, 20, {height/2:.1f})">{y_label}</text>'
    )

    # Data series
    for idx, s in enumerate(series):
        if not s.points:
            continue
        points = sorted(s.points)
        scaled = _scale_points(points, width, height, margin, (x_min, x_max), (y_min, y_max))
        if len(scaled) == 1:
            x_px, y_px = scaled[0]
            svg_lines.append(
                f'<circle cx="{x_px:.2f}" cy="{y_px:.2f}" r="3" fill="{s.color}" stroke="{s.color}"/>'
            )
        else:
            start = scaled[0]
            path_cmds = [f"M {start[0]:.2f} {start[1]:.2f}"]
            for x_px, y_px in scaled[1:]:
                path_cmds.append(f"L {x_px:.2f} {y_px:.2f}")
            svg_lines.append(
                f'<path d="{" ".join(path_cmds)}" fill="none" stroke="{s.color}" stroke-width="2"/>'
            )

    # Legend
    legend_x, legend_y = width - margin - 180, margin + 10
    for idx, s in enumerate(series):
        y_pos = legend_y + idx * 22
        svg_lines.append(
            f'<rect x="{legend_x}" y="{y_pos-12}" width="14" height="14" fill="{s.color}" stroke="{s.color}"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x+20}" y="{y_pos:.1f}" font-size="13">{s.label}</text>'
        )

    svg_lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg_lines), encoding="utf-8")


def build_learning_series(rows: Sequence[Dict[str, float]]) -> List[Series]:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    series: List[Series] = []
    loss_points = [(row["step"], row["loss"]) for row in rows if "loss" in row]
    if loss_points:
        series.append(Series("Loss", loss_points, palette[0]))
    reward_points = [(row["step"], row["reward"]) for row in rows if "reward" in row]
    if reward_points:
        series.append(Series("Mean Reward", reward_points, palette[1]))
    return series


def build_reward_series(reward_series: Dict[str, List[Tuple[float, float]]]) -> List[Series]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    series: List[Series] = []
    for idx, (name, points) in enumerate(sorted(reward_series.items())):
        series.append(Series(name, points, palette[idx % len(palette)]))
    return series


def smooth_points(points: List[Tuple[float, float]], weight: float = 0.9) -> List[Tuple[float, float]]:
    if not points:
        return []
    points = sorted(points, key=lambda x: x[0])
    smoothed = []
    last = points[0][1]
    smoothed.append((points[0][0], last))
    for x, y in points[1:]:
        last = last * weight + y * (1 - weight)
        smoothed.append((x, last))
    return smoothed


def smooth_series_list(series_list: List[Series], weight: float = 0.9) -> List[Series]:
    return [Series(s.label, smooth_points(s.points, weight), s.color) for s in series_list]


def main() -> None:
    args = parse_args()
    trainer_state = locate_trainer_state(args.trainer_state)
    history = load_log_history(trainer_state)

    learning_rows = extract_learning_rows(history)
    learning_csv_path = args.output_dir / "learning_curve.csv"
    write_csv(
        learning_rows,
        ["step", "epoch", "loss", "reward", "reward_std", "entropy", "grad_norm", "learning_rate"],
        learning_csv_path,
    )

    reward_series = extract_reward_series(history)
    reward_csv_path = args.output_dir / "reward_curve.csv"
    write_reward_long_csv(reward_series, reward_csv_path)

    learning_series = build_learning_series(learning_rows)
    if args.smooth:
        learning_series = smooth_series_list(learning_series)
    render_svg(
        learning_series,
        args.output_dir / "learning_curve.svg",
        title="GRPO Learning Curve",
        y_label="Value",
    )

    reward_chart_series = build_reward_series(reward_series)
    if args.smooth:
        reward_chart_series = smooth_series_list(reward_chart_series)
    render_svg(
        reward_chart_series,
        args.output_dir / "reward_curves.svg",
        title="Reward Functions Over Training",
        y_label="Mean Reward per Batch",
        y_bounds=(0.0, 1.0),
    )


if __name__ == "__main__":
    main()
