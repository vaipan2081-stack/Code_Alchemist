# Smart Energy Microgrid & Demand Forecasting
**Team:** Code Alchemist

A Zoho Catalyst application that predicts hourly electricity demand and
applies the HackSphere microgrid transition rules.

## Live application

https://smart-energy-dashboard-50047227333.development.catalystappsail.in

The current application is deployed in the Catalyst development environment.

## How it works

1. The user uploads a CSV with 24 consecutive hourly telemetry readings.
2. The backend calculates time features and sends the inputs to the
   published Zoho Catalyst QuickML regression endpoint.
3. The controller applies the required grid-mode and overload rules.
4. The dashboard displays the forecast, decisions, alerts and a simulated
   battery state of charge.

## Input format

The CSV must contain these columns:

- `timestamp`
- `ambient_temp_c`
- `humidity_pct`
- `solar_irradiance_wm2`

The dashboard expects 24 rows in ascending order, exactly one hour apart.

## Transition rules

- Renewable Storage Mode: demand above 160 kW, or timestamp hour from
  18:00 to 22:00 inclusive.
- Main Grid Mode: demand at or below 160 kW and outside that time window.
- Critical overload: demand at or above 200 kW.

## Battery simulation

Battery state of charge is simulated; it is not live battery telemetry.

The current assumptions are:

- Capacity: 2,000 kWh.
- Starting charge: 80%.
- In storage mode, the predicted demand is drawn over one hour.
- In main grid mode, the battery charges at 100 kW.
- Charging losses and solar generation are not modelled.

The dashboard flags hours when the simulated battery cannot supply the
full predicted demand. This warning does not change the prescribed
grid-mode decision.

## Files

- `app.py` — Flask application and forecast API.
- `predict.py` — QuickML inference and CSV export.
- `controller.py` — transition rules and battery simulation.
- `templates/index.html` — dashboard.
- `requirements.txt` — Python dependencies.
- `test_eval.py` — organizer-provided self-test.

## Local setup

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For direct QuickML access, configure the following environment variables.
They are not needed when using the deployed proxy.

- `ZOHO_CLIENT_ID`
- `ZOHO_CLIENT_SECRET`
- `ZOHO_REFRESH_TOKEN`
- `QUICKML_ENDPOINT_KEY`

The values must be supplied securely. Do not commit credentials or token
files to a public repository.

Start the dashboard:

```bash
python3 app.py
```

Open `http://localhost:8080`.

## Prediction script

Run the self-test:

```bash
python3 test_eval.py
```

Generate a predictions CSV:

```bash
python3 predict.py --input sample_test_input.csv --output predictions.csv
```

The output contains `timestamp` and `target_demand_kw`, with one prediction
per input row.

## Model

The current QuickML model is a Random Forest regression model. Its inputs
include weather readings and time features derived from the timestamp.
QuickML reported an evaluation RMSE of approximately 10.61 kW; the
evaluation split method has not been independently confirmed.

## Evaluation without local credentials

When QuickML credentials are not available as environment variables,
`predict.py` sends input rows to the deployed Catalyst application's
`/api/predict` endpoint. The application calls the published QuickML
model using credentials configured in Catalyst and returns the predictions.

This allows the self-test and prediction script to run without placing
credentials in the source package. Internet access and availability of
the deployed Catalyst development service are required.

If the QuickML credentials are configured locally, `predict.py` calls
the QuickML endpoint directly.

## Deployed server version

`predict_server.py` is the prediction module used in the deployed
AppSail build. It calls QuickML directly and caches its OAuth access
token in memory to reduce token refresh requests.

For an AppSail build using this version, package `predict_server.py`
as `predict.py`, alongside `app.py`. The deployed service requires the
four Zoho and QuickML environment variables listed above.

The root `predict.py` supports direct access and the credential-free
proxy mode used by the evaluator. Keep the deployed server on direct
QuickML access to avoid a proxy calling itself.