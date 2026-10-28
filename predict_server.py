import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import threading
import time


QUICKML_ENDPOINT_URL = (
    "https://api.catalyst.zoho.in/quickml/v1/project/"
    "90812000000013024/endpoints/predict"
)

CATALYST_ORG = "60093611309"
ZOHO_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
TOKEN_FILE = Path(__file__).parent / ".zoho_tokens.json"

REQUIRED_COLUMNS = [
    "timestamp",
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
]

_token_lock = threading.Lock()
_cached_token = None
_token_expires_at = 0.0

def get_access_token():
    global _cached_token, _token_expires_at

    with _token_lock:
        now = time.monotonic()

        if _cached_token and now < _token_expires_at:
            return _cached_token

        client_id = os.getenv("ZOHO_CLIENT_ID")
        client_secret = os.getenv("ZOHO_CLIENT_SECRET")
        refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")

        if not client_id or not client_secret or not refresh_token:
            raise RuntimeError("Zoho credentials are not configured.")

        response = requests.post(
            ZOHO_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"Zoho token request failed with HTTP {response.status_code}."
            )

        result = response.json()
        access_token = result.get("access_token")

        if not access_token:
            raise RuntimeError(
                "Zoho did not return an access token. "
                f"Error: {result.get('error', 'unknown')}"
            )

        expires_in = int(result.get("expires_in", 3600))

        _cached_token = access_token
        _token_expires_at = time.monotonic() + max(0, expires_in - 120)

        return access_token

def prepare_features(input_df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in input_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = input_df[REQUIRED_COLUMNS].copy()

    timestamps = pd.to_datetime(features["timestamp"], errors="coerce")

    if timestamps.isna().any():
        raise ValueError("One or more timestamps are missing or invalid.")

    for column in REQUIRED_COLUMNS[1:]:
        features[column] = pd.to_numeric(features[column], errors="coerce")

        if features[column].isna().any():
            raise ValueError(f"Column {column} has missing or invalid values.")

        if not np.isfinite(features[column]).all():
            raise ValueError(f"Column {column} contains infinite values.")

    hours = timestamps.dt.hour

    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["dayofweek"] = timestamps.dt.dayofweek

    features["timestamp"] = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")

    return features


def predict(input_df: pd.DataFrame):
    features = prepare_features(input_df)

    if features.empty:
        return np.array([], dtype=float)

    endpoint_key = os.getenv("QUICKML_ENDPOINT_KEY")

    if not endpoint_key:
        raise RuntimeError(
            "QUICKML_ENDPOINT_KEY must be set in the terminal environment."
        )

    access_token = get_access_token()

    headers = {
        "X-QUICKML-ENDPOINT-KEY": endpoint_key,
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "CATALYST-ORG": CATALYST_ORG,
        "Environment": "Development",
    }

    predictions = []
    rows = features.to_dict(orient="records")
    batch_size = 50

    with requests.Session() as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]

            response = session.post(
                QUICKML_ENDPOINT_URL,
                headers=headers,
                json={"data": batch},
                timeout=60,
            )

            if not response.ok:
                raise RuntimeError(
                    f"QuickML request failed for rows "
                    f"{start + 1}–{start + len(batch)}: "
                    f"HTTP {response.status_code}. "
                    f"Response: {response.text[:500]}"
                )

            result = response.json().get("result")

            if not isinstance(result, list) or len(result) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} predictions for the batch, "
                    f"but received {len(result) if isinstance(result, list) else 'an invalid result'}."
                )

            batch_predictions = np.asarray(result, dtype=float)

            if not np.isfinite(batch_predictions).all():
                raise ValueError("QuickML returned an invalid prediction.")

            predictions.extend(batch_predictions.tolist())

    return np.asarray(predictions, dtype=float)




def main():
    parser = argparse.ArgumentParser(
        description="Generate energy demand predictions using QuickML."
    )

    parser.add_argument(
        "--input",
        default="sample_test_input.csv",
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path for the output CSV file.",
    )

    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    predictions = predict(input_df)

    output_df = pd.DataFrame(
        {
            "timestamp": input_df["timestamp"],
            "target_demand_kw": predictions,
        }
    )

    output_df.to_csv(args.output, index=False)
    print(f"Saved {len(output_df)} predictions to {args.output}")


if __name__ == "__main__":
    main()