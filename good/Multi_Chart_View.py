import requests
import time

TOKEN_ADDRESS = "5NhN6zzDkzwXFGPFqtpTopV4ttBeZ6CWy1oRL9Rkpump"
URL = f"https://api.dexscreener.com/latest/dex/tokens/{TOKEN_ADDRESS}"

def get_live_stats():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        pairs = data.get('pairs')
        if pairs:
            pair = pairs[0]  # First active trading pair
            
            price_usd = pair.get('priceUsd', '0')
            market_cap = pair.get('marketCap', 0)
            price_changes = pair.get('priceChange', {})
            
            print("=" * 40)
            print(f"Token: {pair.get('baseToken', {}).get('name')} (${pair.get('baseToken', {}).get('symbol')})")
            print(f"Price USD: ${price_usd}")
            print(f"Market Cap: ${market_cap:,.2f}")
            print("-" * 40)
            print(f"5m Change : {price_changes.get('m5', 0):+.2f}%")
            print(f"1h Change : {price_changes.get('h1', 0):+.2f}%")
            print(f"6h Change : {price_changes.get('h6', 0):+.2f}%")
            print(f"24h Change: {price_changes.get('h24', 0):+.2f}%")
            print("=" * 40)
        else:
            print("No liquidity pair found yet.")
    else:
        print(f"Error fetching data: {response.status_code}")

if __name__ == "__main__":
    get_live_stats()