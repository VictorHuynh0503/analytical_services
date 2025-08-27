import requests

def get_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 10,  # Adjust the number of cryptocurrencies you want to fetch
        "page": 1,
        "sparkline": False,
        "locale": "en"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data:", response.status_code)
        return []

def get_crypto_details(crypto_id):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching details for {crypto_id}:", response.status_code)
        return None

def main():
    cryptos = get_crypto_data()
    
    for crypto in cryptos:
        print("----------------------------------------")
        print(f"Name: {crypto['name']}")
        print(f"Symbol: {crypto['symbol'].upper()}")
        print(f"Market Cap Rank: {crypto['market_cap_rank']}")
        print(f"Current Price: ${crypto['current_price']}")
        
        details = get_crypto_details(crypto['id'])
        if details:
            categories = details.get("categories", [])
            if categories:
                print("Use Cases / Categories:", ", ".join(categories))
            else:
                print("Use Cases / Categories: Not available")
            
            print("Description:", details["description"].get("en", "No description available."))
        
        print("----------------------------------------\n")

if __name__ == "__main__":
    main()
