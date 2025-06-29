import api_base
from fastapi.testclient import TestClient
from datetime import datetime

client = TestClient(api_base.app)
user_id = 203
time = datetime(2021, 12, 25)

try:
    r = client.get(
        f"/post/recommendations/",
        params={"id": user_id, "time": time, "limit": 5},
    )

except Exception as e:
    raise ValueError(f"Ошибка при выполнение запроса {type(e)} {str(e)}")

print(r.json())
