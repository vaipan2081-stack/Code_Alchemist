import pandas as pd


def get_grid_decision(timestamp, predicted_demand_kw):
    timestamp = pd.Timestamp(timestamp)
    hour = timestamp.hour

    if predicted_demand_kw > 160 or 18 <= hour <= 22:
        grid_mode = "Renewable Storage Mode"
    else:
        grid_mode = "Main Grid Mode"

    critical_overload = predicted_demand_kw >= 200

    return {
        "grid_mode": grid_mode,
        "critical_overload": critical_overload,
    }


def apply_controller(predictions_df):
    required_columns = ["timestamp", "target_demand_kw"]

    missing = [
        column for column in required_columns
        if column not in predictions_df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = predictions_df.copy()

    timestamps = pd.to_datetime(results["timestamp"], errors="coerce")
    demands = pd.to_numeric(results["target_demand_kw"], errors="coerce")

    if timestamps.isna().any() or demands.isna().any():
        raise ValueError("The file contains invalid timestamps or demands.")

    decisions = [
        get_grid_decision(timestamp, demand)
        for timestamp, demand in zip(timestamps, demands)
    ]

    results["grid_mode"] = [item["grid_mode"] for item in decisions]
    results["critical_overload"] = [
        item["critical_overload"] for item in decisions
    ]

    return results

def simulate_battery(
    decisions_df,
    capacity_kwh=2000.0,
    starting_charge_pct=80.0,
    charge_rate_kw=100.0,
):
    results = decisions_df.copy()

    energy_kwh = capacity_kwh * starting_charge_pct / 100
    charge_levels = []
    insufficient_storage = []

    for _, row in results.iterrows():
        if row["grid_mode"] == "Renewable Storage Mode":
            required_kwh = max(float(row["target_demand_kw"]), 0.0)
            insufficient = required_kwh > energy_kwh
            energy_kwh = max(0.0, energy_kwh - required_kwh)
        else:
            insufficient = False
            energy_kwh = min(capacity_kwh, energy_kwh + charge_rate_kw)

        charge_levels.append(round(100 * energy_kwh / capacity_kwh, 2))
        insufficient_storage.append(insufficient)

    results["battery_charge_pct"] = charge_levels
    results["insufficient_storage"] = insufficient_storage

    return results


if __name__ == "__main__":
    predictions = pd.read_csv("predictions_24_hours.csv")
    results = apply_controller(predictions)

    results.to_csv("grid_decisions_24_hours.csv", index=False)

    print(results.to_string(index=False))
    print("\nSaved results to grid_decisions_24_hours.csv")