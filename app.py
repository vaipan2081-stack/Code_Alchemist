import io
import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from controller import apply_controller, simulate_battery
from predict import predict


app = Flask(__name__)

REQUIRED_COLUMNS = [
    "timestamp",
    "ambient_temp_c",
    "humidity_pct",
    "solar_irradiance_wm2",
]


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/forecast")
def forecast():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Please upload a CSV file."}), 400

    try:
        data = uploaded_file.read()

        if len(data) > 1_000_000:
            return jsonify({"error": "The CSV file must be smaller than 1 MB."}), 400

        input_df = pd.read_csv(io.BytesIO(data))

        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in input_df.columns
        ]

        if missing:
            return jsonify(
                {"error": f"Missing required columns: {', '.join(missing)}"}
            ), 400

        if len(input_df) != 24:
            return jsonify(
                {"error": "Please provide exactly 24 hourly rows."}
            ), 400

        timestamps = pd.to_datetime(input_df["timestamp"], errors="coerce")

        if timestamps.isna().any():
            return jsonify({"error": "One or more timestamps are invalid."}), 400

        if timestamps.duplicated().any():
            return jsonify({"error": "The CSV contains duplicate timestamps."}), 400

        if not timestamps.is_monotonic_increasing:
            return jsonify(
                {"error": "Please arrange timestamps in ascending order."}
            ), 400

        if not (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all():
            return jsonify(
                {"error": "Timestamps must be exactly one hour apart."}
            ), 400

        predictions = predict(input_df)

        prediction_df = pd.DataFrame(
            {
                "timestamp": input_df["timestamp"],
                "target_demand_kw": predictions,
            }
        )

        decisions = apply_controller(prediction_df)
        decisions = simulate_battery(decisions)

        return jsonify(
            {
                "forecast": decisions.to_dict(orient="records"),
            }
        )

    except (ValueError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Forecast request failed")
        return jsonify(
            {"error": "Could not generate the forecast. Check the server logs."}
        ), 500

@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object."}), 400

    rows = payload.get("data")

    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        return jsonify(
            {"error": "Provide between 1 and 1000 rows in data."}
        ), 400

    if not all(isinstance(row, dict) for row in rows):
        return jsonify({"error": "Each row must be a JSON object."}), 400

    try:
        input_df = pd.DataFrame(rows)
        predictions = predict(input_df)

        return jsonify({"predictions": predictions.tolist()})

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction API request failed")
        return jsonify({"error": "Could not generate predictions."}), 500



if __name__ == "__main__":
        port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", 8080))
        app.run(host="0.0.0.0", port=port)