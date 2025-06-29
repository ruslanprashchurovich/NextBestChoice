import requests
from datetime import datetime

base_url = "http://127.0.0.1:8000/post/recommendations/"

params = {"id": 201, "time": datetime(2021, 9, 8, 14, 47).isoformat(), "limit": 5}

response = requests.get(base_url, params=params)
print(response.json())
