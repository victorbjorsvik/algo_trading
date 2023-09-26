import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yfin
from pandas_datareader import data as pdr

# Initialize Yahoo Finance
yfin.pdr_override()

# Define time range
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()
spx = pdr.get_data_yahoo('SPY', start, end)
us_treasury_5y = pdr.get_data_yahoo('IEI', start, end)
columns = [
'Date', 'Close_Price',
'Bid1_Price', 'Bid1_Quantity',
'Bid2_Price', 'Bid2_Quantity',
'Bid3_Price', 'Bid3_Quantity','Ask1_Price', 'Ask1_Quantity',
'Ask2_Price', 'Ask2_Quantity',
'Ask3_Price', 'Ask3_Quantity'
]
spx_order_book_rows = []
iei_order_book_rows = []

# Helper function to create a single day's order book
def create_order_book_row(date, close_price):
    bid1_price = close_price - np.random.uniform(0.05, 3)
    bid2_price = bid1_price - np.random.uniform(0.05, 3)
    bid3_price = bid2_price - np.random.uniform(0.05, 3)
    ask1_price = close_price + np.random.uniform(0.05, 3)
    ask2_price = ask1_price + np.random.uniform(0.05, 3)
    ask3_price = ask2_price + np.random.uniform(0.05, 3)
    bid1_quantity = np.random.randint(1, 10000)
    bid2_quantity = np.random.randint(1, 10000)
    bid3_quantity = np.random.randint(1, 10000)
    ask1_quantity = np.random.randint(1, 10000)
    ask2_quantity = np.random.randint(1, 10000)
    ask3_quantity = np.random.randint(1, 10000)

    return {'Date': date,
    'Close_Price': close_price,
    'Bid1_Price': bid1_price,
    'Bid1_Quantity': bid1_quantity,
    'Bid2_Price': bid2_price,
    'Bid2_Quantity': bid2_quantity,
    'Bid3_Price': bid3_price,
    'Bid3_Quantity': bid3_quantity,
    'Ask1_Price': ask1_price,
    'Ask1_Quantity': ask1_quantity,
    'Ask2_Price': ask2_price,
    'Ask2_Quantity': ask2_quantity,
    'Ask3_Price': ask3_price,
    'Ask3_Quantity': ask3_quantity
    }

# Generate order book for SPY
for date, close_price in spx['Close'].items():
spx_order_book_rows.append(create_order_book_row(date, close_price))
# Generate order book for IEI
for date, close_price in us_treasury_5y['Close'].items():
iei_order_book_rows.append(create_order_book_row(date, close_price))
# Create DataFrames
spx_order_book_df = pd.DataFrame(spx_order_book_rows)
iei_order_book_df = pd.DataFrame(iei_order_book_rows)


def trading_algorithm(spx_order_book: pd.DataFrame, iei_order_book:
pd.DataFrame, current_position: pd.DataFrame) -> tuple:

"""
Executes a sample trading algorithm to provide trading decisions for
SPX and IEI assets.
This function is intended to serve as a template and should be
modified to include
specific trading logic.
Parameters
----------
spx_order_book : pd.DataFrame
DataFrame containing historical data and a simulated order book
for the SPX asset.
The DataFrame is expected to have the following columns:
- 'Date': The date of the data
- 'Close_Price': The closing price of the asset
- 'Bid1_Price': The highest bid price in the order book
- 'Bid1_Qty': The quantity at the highest bid price
- 'Bid2_Price': The second-highest bid price in the order book
- 'Bid2_Qty': The quantity at the second-highest bid price
... (additional bid levels)
- 'Ask1_Price': The lowest ask price in the order book
- 'Ask1_Qty': The quantity at the lowest ask price
- 'Ask2_Price': The second-lowest ask price in the order book
- 'Ask2_Qty': The quantity at the second-lowest ask price
... (additional ask levels)
iei_order_book : pd.DataFrame
DataFrame containing historical data and a simulated order book
for the IEI asset.
The DataFrame is expected to have the following columns:
- 'Date': The date of the data
- 'Close_Price': The closing price of the asset
- 'Bid1_Price': The highest bid price in the order book
- 'Bid1_Qty': The quantity at the highest bid price
- 'Bid2_Price': The second-highest bid price in the order book
- 'Bid2_Qty': The quantity at the second-highest bid price
... (additional bid levels)
- 'Ask1_Price': The lowest ask price in the order book
- 'Ask1_Qty': The quantity at the lowest ask price
- 'Ask2_Price': The second-lowest ask price in the order book
- 'Ask2_Qty': The quantity at the second-lowest ask price
... (additional ask levels)
current_position : pd.DataFrame
DataFrame containing the current trading position.
Expected columns are:
- 'Cash': Available cash
- 'SPX Position': Current position in SPX
- 'IEI Position': Current position in IEI
- 'PnL': Profit and Loss
Returns
-------
tuple
A tuple containing trading decisions as follows:
(SPX Buying Signal, SPX Price, SPX Qty, IEI Buying Signal, IEI
Price, IEI Qty)
- SPX Buying Signal (int): Trading signal for SPX. Values can be
Sell (-1), Hold (0), or Buy (1).
- SPX Price (float): Trading price for SPX. Relevant only if SPX
Buying Signal is not 0.
- SPX Qty (int): Number of SPX shares to buy/sell. Relevant only
if SPX Buying Signal is not 0.
- IEI Buying Signal (int): Trading signal for IEI. Values can be
Sell (-1), Hold (0), or Buy (1).
- IEI Price (float): Trading price for IEI. Relevant only if IEI
Buying Signal is not 0.
- IEI Qty (int): Number of IEI shares to buy/sell. Relevant only
if IEI Buying Signal is not 0.
Example
-------
>>> trading_algorithm(spx_order_book_df, iei_order_book_df,
current_position_df)
(1, 4030.50, 10, -1, 114.25, 20)
Note
----
The trading logic included in this function is simplistic and for
demonstration purposes only.
Implement your own trading logic for a production environment.
"""
# Replace with your specific trading logic for SPX
spx_latest_close = spx_order_book['Close_Price'].iloc[-1]
spx_signal = 0 # 0 = Hold
spx_qty = 0 # Quantity to buy/sell
if spx_latest_close > 4000:
spx_signal = 1 # 1 = Buy
spx_qty = 10
elif spx_latest_close < 3500:
spx_signal = -1 # -1 = Sell
spx_qty = 10
# Replace with your specific trading logic for IEI
iei_latest_close = iei_order_book['Close_Price'].iloc[-1]
iei_signal = 0 # 0 = Hold
iei_qty = 0 # Quantity to buy/sell
if iei_latest_close > 120:
iei_signal = 1 # 1 = Buy
iei_qty = 20
elif iei_latest_close < 115:
iei_signal = -1 # -1 = Sell
iei_qty = 20
return (spx_signal, spx_latest_close, spx_qty, iei_signal,
iei_latest_close, iei_qty)
