"""Summarize FlyStress v1.0 sleep_results.csv for graphing and review."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def f(value: str | None, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return default


def resolve(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file(): return path
    candidate = path / "analysis" / "sleep_results.csv"
    if candidate.is_file(): return candidate
    raise FileNotFoundError(f"Could not find sleep_results.csv under {path}")


def summarize(results_path: Path) -> Path:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with results_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            groups[row.get("fly_name", "")].append(row)
    output = results_path.parent / "sleep_summary.csv"
    fields = ["fly_name", "well", "total_samples", "valid_samples", "unknown_samples",
              "awake_seconds", "asleep_seconds", "percent_asleep_of_classified",
              "sleep_bouts", "average_sleep_bout_seconds", "longest_sleep_bout_seconds",
              "first_sleep_onset_seconds", "total_adjusted_distance_px",
              "maximum_rolling_distance_px", "maximum_immobile_duration_seconds"]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for name in sorted(k for k in groups if k):
            rows = sorted(groups[name], key=lambda r: f(r.get("elapsed_seconds")))
            states = [r.get("fly_state", "UNKNOWN").upper() for r in rows]
            times = [f(r.get("elapsed_seconds")) for r in rows]
            positive_steps = [b-a for a, b in zip(times, times[1:]) if b > a]
            fallback_step = sorted(positive_steps)[len(positive_steps)//2] if positive_steps else 1.0
            durations = [(times[i+1]-times[i]) if i+1 < len(times) and times[i+1] > times[i] else fallback_step
                         for i in range(len(times))]
            awake_duration = sum(dt for state, dt in zip(states, durations) if state == "AWAKE")
            asleep_duration = sum(dt for state, dt in zip(states, durations) if state == "ASLEEP")
            unknown = sum(state == "UNKNOWN" for state in states)
            valid = sum(state in {"AWAKE", "ASLEEP"} for state in states)
            bouts: list[float] = []; current = 0.0; first = ""
            for row, state, dt in zip(rows, states, durations):
                if state == "ASLEEP":
                    if current == 0.0 and first == "": first = row.get("elapsed_seconds", "")
                    current += dt
                elif current > 0.0:
                    bouts.append(current); current = 0.0
            if current > 0.0: bouts.append(current)
            classified_duration = awake_duration + asleep_duration
            writer.writerow({
                "fly_name": name, "well": rows[0].get("well", ""), "total_samples": len(rows),
                "valid_samples": valid, "unknown_samples": unknown,
                "awake_seconds": f"{awake_duration:.3f}", "asleep_seconds": f"{asleep_duration:.3f}",
                "percent_asleep_of_classified": f"{100*asleep_duration/classified_duration:.2f}" if classified_duration else "",
                "sleep_bouts": len(bouts),
                "average_sleep_bout_seconds": f"{sum(bouts)/len(bouts):.3f}" if bouts else "0.000",
                "longest_sleep_bout_seconds": f"{max(bouts):.3f}" if bouts else "0.000",
                "first_sleep_onset_seconds": first,
                "total_adjusted_distance_px": f"{sum(f(r.get('distance_px')) for r in rows):.3f}",
                "maximum_rolling_distance_px": f"{max((f(r.get('rolling_distance_px')) for r in rows), default=0):.3f}",
                "maximum_immobile_duration_seconds": f"{max((f(r.get('immobile_duration_seconds')) for r in rows), default=0):.3f}",
            })
    print(f"Sleep summary saved to: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args(); summarize(resolve(args.path))

if __name__ == "__main__": main()
