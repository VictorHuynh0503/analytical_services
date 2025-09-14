import requests

# Replace with your Supabase project details
url = "https://ycnjgpfabhtnjymjpqcp.supabase.co"
api_key = "YeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InljbmpncGZhYmh0bmp5bWpwcWNwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc3ODc2OTgsImV4cCI6MjA3MzM2MzY5OH0.pJlk1n2HAWyixR5XGfXx5qMi5y1KkcpDv0b1W0KgydU"

# The REST endpoint for your table
table_name = "news"
endpoint = f"{url}/rest/v1/{table_name}"

# Set headers (important: API key + JSON response)
headers = {
    "apikey": api_key,
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Example: Get all rows from table
response = requests.get(endpoint, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("Data:", data)
else:
    print("Error:", response.status_code, response.text)
