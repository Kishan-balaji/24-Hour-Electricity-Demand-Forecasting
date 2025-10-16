import requests
import pandas as pd

def get_weather_forecast(city):

    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    geo = requests.get(geo_url, timeout=10).json()
    if "results" not in geo or not geo["results"]:
        raise ValueError(f"City '{city}' not found via Open-Meteo geocoding API.")

    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m&forecast_days=2&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("hourly", {})

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"]
    })

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

    df = df.set_index("timestamp").sort_index()
    
    return df.head(24)
