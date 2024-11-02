from locust import HttpUser, task, between
import json

class WebsiteTestUser(HttpUser):
    wait_time = between(0.5, 3.0)

    @task(1)
    def test1(self):
        self.client.get("https://mlflaskapp.azurewebsites.net/")

    @task(2)
    def test2(self):
        json_data = {
            "CHAS": {"0": 0},
            "RM": {"0": 6.575},
            "TAX": {"0": 296.0},
            "PTRATIO": {"0": 15.3},
            "B": {"0": 396.9},
            "LSTAT": {"0": 4.98}
        }
        
        # Convert JSON data to string
        payload = json.dumps(json_data)
        self.client.post("https://mlflaskapp.azurewebsites.net/predict", data=payload, headers={"Content-Type": "application/json"})