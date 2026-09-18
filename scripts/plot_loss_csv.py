#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def moving_average(values, window):
    result = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def write_svg(steps, values, label, output, smooth_window):
    smooth = moving_average(values, smooth_window)
    width, height = 1000, 600
    left, right, top, bottom = 85, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(steps)
    max_value = max(values)

    def points(series):
        return " ".join(
            f"{left + step / max_step * plot_width:.2f},"
            f"{top + (1 - value / max_value) * plot_height:.2f}"
            for step, value in zip(steps, series)
        )

    title = label.replace("_", " ").title()
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="16">Training step</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16" transform="rotate(-90 20 {height / 2})">{title}</text>',
        f'<text x="{left}" y="{height-bottom+25}" font-family="sans-serif" font-size="13">0</text>',
        f'<text x="{width-right}" y="{height-bottom+25}" text-anchor="end" font-family="sans-serif" font-size="13">{max_step}</text>',
        f'<text x="{left-10}" y="{top+5}" text-anchor="end" font-family="sans-serif" font-size="13">{max_value:.4g}</text>',
        f'<text x="{left-10}" y="{height-bottom+5}" text-anchor="end" font-family="sans-serif" font-size="13">0</text>',
        f'<polyline points="{points(values)}" fill="none" stroke="#93c5fd" stroke-width="1" opacity="0.45"/>',
        f'<polyline points="{points(smooth)}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
        f'<text x="{left}" y="{top+20}" font-family="sans-serif" font-size="13" fill="#2563eb">moving average ({smooth_window} steps)</text>',
        "</svg>",
    ]
    output.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot each loss column in a training CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--smooth-window", type=int, default=100)
    args = parser.parse_args()

    with args.csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No data rows in {args.csv}")

    steps = [int(row["step"]) for row in rows]
    loss_columns = [name for name in rows[0] if "loss" in name]
    for column in loss_columns:
        values = [float(row[column]) for row in rows]
        output = args.csv.parent / f"{column}_curve.svg"
        write_svg(steps, values, column, output, args.smooth_window)
        print(output)


if __name__ == "__main__":
    main()
