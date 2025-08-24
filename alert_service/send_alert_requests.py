import requests

url = "http://localhost:8000/alert"

payload = {
    "type": "error",
    "id": 123,
    "value": 95
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response:", response.json())
