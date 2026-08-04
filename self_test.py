"""Quick no-camera validation for FlyStress Track v1.0."""
from sleep_analysis.fly_detection import FlyDetection
from sleep_analysis.single_fly_tracker import SingleFlyTracker


def main() -> None:
    tracker = SingleFlyTracker(["A1"], jitter_threshold_px=3.0,
                               rolling_window_seconds=300.0, sleep_duration_seconds=300.0,
                               max_position_jump_px=45.0, max_valid_sample_gap_seconds=2.5)
    detection = [FlyDetection("A1", 10.0, 10.0, 0.0, 0.0, 50, 20)]
    result = None
    for second in range(301):
        result = tracker.update("A1", detection, float(second), True)
    assert result is not None and result.state == "ASLEEP"
    assert result.immobile_duration_seconds == 300.0
    missing = tracker.update("A1", [], 302.0, True)
    assert missing.state == "UNKNOWN"
    assert missing.immobile_duration_seconds == 300.0
    print("FlyStress v1.0 self-test passed.")


if __name__ == "__main__":
    main()
