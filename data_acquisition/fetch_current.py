import os
import requests
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

LAT = 28.6139   
LON = 77.2090

client = MongoClient(MONGO_URI)
db = client["aqi_db"]
collection = db["aqi_data"]

def fetch_data():
    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"

    air_response = requests.get(air_url).json()
    weather_response = requests.get(weather_url).json()

    air = air_response["list"][0]
    weather = weather_response

    document = {
        "timestamp": datetime.utcnow(),

        # Pollutants
        "pm25": air["components"]["pm2_5"],
        "pm10": air["components"]["pm10"],
        "no2": air["components"]["no2"],
        "o3": air["components"]["o3"],
        "so2": air["components"]["so2"],
        "co": air["components"]["co"],

        # Weather
        "temperature": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "pressure": weather["main"]["pressure"],
        "wind_speed": weather["wind"]["speed"],
        "wind_deg": weather["wind"]["deg"],
        "clouds": weather["clouds"]["all"],
        "visibility": weather.get("visibility", None),

        # AQI class from API (1–5)
        "aqi": air["main"]["aqi"]
    }

    collection.insert_one(document)
    print("Data inserted successfully")

if __name__ == "__main__":
    fetch_data()
