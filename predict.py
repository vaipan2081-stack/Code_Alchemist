import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


QUICKML_ENDPOINT_URL = (
    "https://api.catalyst.zoho.in/quickml/v1/project/"
    "90812000000013024/endpoints/predict"
)

PROXY_URL = (
    "https://smart-energy-dashboard-50047227333."
    "development.catalystappsail.in/api/predict"
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


def get_access_token():
    client_id = os.getenv("ZOHO_CLIENT_ID")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "ZOHO_CLIENT_ID and ZOHO_CLIENT_SECRET must be set."
        )

    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")

    if not refresh_token and TOKEN_FILE.exists():
        with TOKEN_FILE.open("r", encoding="utf-8") as file:
            tokens = json.load(file)

        refresh_token = tokens.get("refresh_token")

    if not refresh_token:
        raise RuntimeError(
            "No refresh token found in the environment or local token file."
        )

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
    response.raise_for_status()

    result = response.json()
    access_token = result.get("access_token")

    if not access_token:
        raise RuntimeError(
            "Could not refresh the Zoho access token: "
            f"{result.get('error', 'Unknown error')}"
        )

    return access_token


def prepare_features(input_df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in input_df.columns
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


def validate_predictions(result, expected_count):
    if not isinstance(result, list) or len(result) != expected_count:
        raise ValueError(
            f"Expected {expected_count} predictions, but received "
            f"{len(result) if isinstance(result, list) else 'an invalid result'}."
        )

    predictions = np.asarray(result, dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError("The endpoint returned invalid predictions.")

    return predictions


def predict_via_proxy(input_df):
    # Validate and normalize the input before converting it to JSON.
    features = prepare_features(input_df)
    rows = features[REQUIRED_COLUMNS].to_dict(orient="records")
    predictions = []
    batch_size = 24

    with requests.Session() as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]

            for attempt in range(3):
                try:
                    response = session.post(
                        PROXY_URL,
                        json={"data": batch},
                        timeout=120,
                    )

                    # Retry only temporary server/rate-limit responses.
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        response.raise_for_status()

                    # Other HTTP errors should fail immediately.
                    response.raise_for_status()
                    break

                except (requests.ConnectionError, requests.Timeout):
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))

                except requests.HTTPError as error:
                    status = error.response.status_code

                    if status not in (408, 429, 500, 502, 503, 504):
                        raise

                    if attempt == 2:
                        raise

                    time.sleep(2 * (attempt + 1))

            result = response.json().get("predictions")
            batch_predictions = validate_predictions(result, len(batch))
            predictions.extend(batch_predictions.tolist())

    return np.asarray(predictions, dtype=float)


def predict_direct(input_df):
    features = prepare_features(input_df)

    if features.empty:
        return np.array([], dtype=float)

    endpoint_key = os.getenv("QUICKML_ENDPOINT_KEY")

    if not endpoint_key:
        raise RuntimeError("QUICKML_ENDPOINT_KEY must be set.")

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
            batch_predictions = validate_predictions(result, len(batch))
            predictions.extend(batch_predictions.tolist())

    return np.asarray(predictions, dtype=float)


def predict(input_df: pd.DataFrame):
    direct_credentials = all(
        os.getenv(name)
        for name in [
            "ZOHO_CLIENT_ID",
            "ZOHO_CLIENT_SECRET",
            "QUICKML_ENDPOINT_KEY",
        ]
    )

    if direct_credentials:
        return predict_direct(input_df)

    return predict_via_proxy(input_df)


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