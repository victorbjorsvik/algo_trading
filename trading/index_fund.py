################
###INDEX FUND###
################

import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import math
import time

# import data for SP500 companies(could be done by API, but not free)
api_key = "VK0T80OKAA8M8RHM"
stocks = pd.read_csv("constituents_csv.csv")
print(stocks.head())
time.sleep(300)
for i, row in stocks[-2:].iterrows():
    stock = stocks.iloc[i]["Symbol"]
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Make API call for market cap and closing price for last observation of stock
    price = float(data["Global Quote"][r"08. previous close"])
    url2 = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={stock}&apikey={api_key}'
    r2 = requests.get(url2)
    data2 = r2.json()
    market_cap = int(data2["MarketCapitalization"])
    stocks["Price"] = price
    stocks["MCap"] = market_cap
    time.sleep(61)

print(stocks[-2:])
