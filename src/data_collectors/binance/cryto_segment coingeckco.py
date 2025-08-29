import requests
import pandas as pd

def fetch_coin_list():
    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

df_coins = fetch_coin_list()
print(df_coins.head())

def fetch_categories():
    url = "https://api.coingecko.com/api/v3/coins/categories"
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data)

df_categories = fetch_categories()
print(df_categories.head())

import time

def fetch_coin_details(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "id": data["id"],
            "symbol": data["symbol"],
            "name": data["name"],
            "categories": data.get("categories", [])
        }
    else:
        return None

# Example: fetch Ethereum
eth_data = fetch_coin_details("ethereum")
print(eth_data)

coin_metadata = []

for coin_id in df_coins["id"].head(100):  # limit for testing first 100
    details = fetch_coin_details(coin_id)
    if details:
        coin_metadata.append(details)
    time.sleep(1.5)  # avoid rate limit

df_coin_details = pd.DataFrame(coin_metadata)
print(df_coin_details.head())
