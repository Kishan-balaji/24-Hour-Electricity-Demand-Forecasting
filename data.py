
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def prepare_hourly_data(df):
    
    #Converting the date/time or x_Timestamp column to a pandas supported formatted.
    df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])
    df = df.set_index("x_Timestamp").sort_index()

    # Aggregate from 3-min → hourly energy.
    hourly = df["t_kWh"].resample("h").sum()

    # converting to continuous hourly index.
    full_index = pd.date_range(start=hourly.index.min(), end=hourly.index.max(), freq="h", tz=hourly.index.tz)
    hourly = hourly.reindex(full_index)

    # Handle missing values by inerpolating linearly.
    hourly = hourly.interpolate(method="linear", limit_direction="both")

    # Cap extreme outliers at 99th percentile to reduce irregular plots or output.
    cap = hourly.quantile(0.99)
    hourly = np.minimum(hourly, cap)
    hourly = pd.Series(hourly, index=full_index, name="hourly_kwh")
    return hourly

def seasonal_naive_forecast(series):
    last_24 = series[-24:].values
    return last_24.copy()

def ridge_forecast(series, weather_df=None):
    df = pd.DataFrame({"y": series})
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    #Lag features
    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # 24-hour rolling mean
    df["roll24"] = df["y"].rolling(24, min_periods=1).mean()

    df = df.dropna()

    X = df[["sin_hour", "cos_hour", "dayofweek", "lag_1", "lag_2", "lag_3", "roll24"]]
    y = df["y"]

    model = Ridge(alpha=0.2)
    model.fit(X, y)

    resid_std = np.std(y - model.predict(X))

    # Prepare recursive forecasting starting from the last available row
    last_idx = df.index[-1]
    last_row = df.iloc[-1:].copy()
    preds = []
    for h in range(24):
        next_time = last_idx + pd.Timedelta(hours=1)
        hour = next_time.hour
        dayofweek = next_time.dayofweek
        sin_hour = np.sin(2 * np.pi * hour / 24)
        cos_hour = np.cos(2 * np.pi * hour / 24)

        x_input = {
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "dayofweek": dayofweek,
            "lag_1": last_row["lag_1"].values[0],
            "lag_2": last_row["lag_2"].values[0],
            "lag_3": last_row["lag_3"].values[0],
            "roll24": last_row["roll24"].values[0],
        }

        if weather_df is not None and "temperature" in weather_df.columns:
            x_input["temp"] = weather_df["temperature"].iloc[h]

        feature_cols = ["sin_hour", "cos_hour", "dayofweek", "lag_1", "lag_2", "lag_3", "roll24"]
        X_row = pd.DataFrame([x_input])[feature_cols]
        y_pred = model.predict(X_row)[0]
        preds.append(y_pred)

        last_row["lag_3"] = last_row["lag_2"]
        last_row["lag_2"] = last_row["lag_1"]
        last_row["lag_1"] = y_pred
        last_row["roll24"] = (last_row["roll24"] * 24 - last_row["lag_3"].values[0] + y_pred) / 24

        last_idx = next_time

    preds = np.array(preds)

    
    #Low Energy demand
    y_p10 = preds - 1.28 * resid_std
    #Normal Energy demand
    y_p50 = preds
    #High Energy demand
    y_p90 = preds + 1.28 * resid_std

    #This rescales the next 24-hour predictions so that the total daily energy matches recent averages(last 3 days mean).
    daily_mean = series.resample("D").sum()
    recent_mean = daily_mean[-3:].mean()
    forecast_total = y_p50.sum()
    if forecast_total > 0:
        scale_factor = recent_mean / forecast_total
    else:
        scale_factor = 1.0

    y_p10 *= scale_factor
    y_p50 *= scale_factor
    y_p90 *= scale_factor

    return {"p10": y_p10, "p50": y_p50, "p90": y_p90}

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    #Average physical deviation per hour
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.sum(np.abs(y_true))

    #Total deviation relative to actual energy used 
    wape = (np.sum(np.abs(y_true - y_pred)) / denom) if denom != 0 else np.nan

    #Symmetric % error that treats over-prediction and under-prediction equally
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    return {"MAE": mae, "WAPE": wape, "sMAPE": smape}
