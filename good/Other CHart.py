import requests
import time
from collections import deque

TOKEN_ADDRESS = "5NhN6zzDkzwXFGPFqtpTopV4ttBeZ6CWy1oRL9Rkpump"
URL = f"https://api.dexscreener.com/latest/dex/tokens/{TOKEN_ADDRESS}"

# Store rolling history: (timestamp, price)
price_history = deque()

def get_pct_change(seconds_ago, current_price):
    now = time.time()
    # Find price closest to (now - seconds_ago)
    for ts, price in price_history:
        if now - ts >= seconds_ago:
            if price == 0: return 0.0
            return ((current_price - price) / price) * 100
    return 0.0

print(f"Starting live micro-tracker for {TOKEN_ADDRESS}...")

while True:
    try:
        res = requests.get(URL, timeout=3).json()
        pairs = res.get('pairs')
        if pairs:
            current_price = float(pairs[0].get('priceUsd', 0))
            now = time.time()
            
            price_history.append((now, current_price))
            
            # Keep maximum 15 minutes of memory
            while price_history and (now - price_history[0][0]) > 900:
                price_history.popleft()
            
            # Print live calculated percentage changes
            print("\031[H\033[J", end="") # Clear console
            print(f"--- Live Price: ${current_price:.8f} ---")
            print(f"1s  Change: {get_pct_change(1, current_price):+.2f}%")
            print(f"15s Change: {get_pct_change(15, current_price):+.2f}%")
            print(f"30s Change: {get_pct_change(30, current_price):+.2f}%")
            print(f"1m  Change: {get_pct_change(60, current_price):+.2f}%")
            print(f"5m  Change: {get_pct_change(300, current_price):+.2f}%")
            
    except Exception as e:
        print(f"Polling error: {e}")
        
    time.sleep(1) # Poll every 1 second