import os
import requests
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found")

LAT = 24.8607
LON = 67.0011

client = MongoClient(MONGO_URI)
db = client["aqi_db"]
collection = db["aqi_data"]

# ---------------- FETCH AIR POLLUTION HISTORY ---------------- #

def fetch_air_pollution(start_unix, end_unix):
    url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}"
        f"&start={start_unix}&end={end_unix}"
        f"&appid={API_KEY}"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json().get("list", [])

# ---------------- FETCH WEATHER (TIMEMACHINE) ---------------- #

def fetch_weather_for_day(dt_unix):
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={LAT}&lon={LON}"
        f"&dt={dt_unix}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# ---------------- MAIN BACKFILL ---------------- #

def backfill(days=60):

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    start_unix = int(start.timestamp())
    end_unix = int(end.timestamp())

    print("Fetching pollution history...")
    pollution_data = fetch_air_pollution(start_unix, end_unix)

    if not pollution_data:
        print("No pollution data returned.")
        return

    records = []

    for entry in pollution_data:

        dt_utc = datetime.fromtimestamp(entry["dt"], timezone.utc)
        dt = dt_utc.replace(tzinfo=None)

        # Optional: reduce volume (every 4 hours)
        if dt.hour % 4 != 0:
            continue

        print(f"Processing {dt}")

        # Fetch weather for this timestamp
        weather_data = fetch_weather_for_day(entry["dt"])

        weather_hour = None
        for hour in weather_data.get("data", []):
            if hour["dt"] == entry["dt"]:
                weather_hour = hour
                break

        if not weather_hour:
            continue

        record = {
            "timestamp": dt,

            # Pollutants
            "pm25": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"],
            "so2": entry["components"]["so2"],
            "co": entry["components"]["co"],

            # Weather
            "temperature": weather_hour.get("temp"),
            "humidity": weather_hour.get("humidity"),
            "pressure": weather_hour.get("pressure"),
            "wind_speed": weather_hour.get("wind_speed"),
            "wind_deg": weather_hour.get("wind_deg"),
            "clouds": weather_hour.get("clouds"),
            "visibility": weather_hour.get("visibility"),

            # AQI CLASS DIRECTLY FROM API (1–5)
            "aqi": entry["main"]["aqi"]
        }

        records.append(record)

        time.sleep(1)  # Respect API rate limits

    if records:
        collection.insert_many(records)
        print(f"\nInserted {len(records)} records into MongoDB.")
    else:
        print("No valid records collected.")

if __name__ == "__main__":
    backfill(days=60)
