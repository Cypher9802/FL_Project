import yaml
from pathlib import Path
from federated.privacy import PrivacyTracker

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

def main():
    tracker = PrivacyTracker(
        target_epsilon=cfg["privacy"]["epsilon"],
        target_delta=cfg["privacy"]["delta"]
    )

    # Simulate privacy budget expenditure
    sample_rate = cfg["federated"]["clients_per_round"] / cfg["federated"]["batch_size"]  # example
    for round_num in range(1, cfg["federated"]["rounds"] + 1):
        tracker.add_round(
            noise_multiplier=cfg["privacy"]["noise_multiplier"],
            sample_rate=sample_rate
        )

    report = tracker.get_privacy_report()
    print("Privacy Analysis Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
