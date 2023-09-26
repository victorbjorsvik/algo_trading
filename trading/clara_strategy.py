import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yfin
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# Initialize Yahoo Finance
yfin.pdr_override()

# Define time range
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

# Get data
spx = pdr.get_data_yahoo('SPY', start, end)
us_treasury_5y = pdr.get_data_yahoo('IEI', start, end)

# Define columns
columns = [
    'Date', 'Close_Price',
    'Bid1_Price', 'Bid1_Quantity',
    'Bid2_Price', 'Bid2_Quantity',
    'Bid3_Price', 'Bid3_Quantity',
    'Ask1_Price', 'Ask1_Quantity',
    'Ask2_Price', 'Ask2_Quantity',
    'Ask3_Price', 'Ask3_Quantity'
]

# Create empty lists to store rows for each order book
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

    return {
        'Date': date,
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

#######################################################################################################
################################ OUR TRADING ALGO #####################################################
#######################################################################################################

import math

"""
*We need to define our assumptions (SMART)
*We should include some charts


"""
# Define stop-loss and take-profit percentages
stop_loss_percentage = 0.1  # Increase the stop-loss percentage
take_profit_percentage = 0.1  # Increase the take-profit percentage

# Define windows for MA
short_window = 50
long_window = 200

def calculate_moving_average(data, window):
    return data['Close_Price'].rolling(window=window).mean()

def trading_algorithm(spx_order_book: pd.DataFrame, iei_order_book: pd.DataFrame, current_position: pd.DataFrame = None) -> tuple:
    # Parameters for risk management
    max_position_size = 5000  # Increase the maximum position size in dollars
    short_ma_window = short_window  # Short moving average period (adjust as needed)
    long_ma_window = long_window  # Long moving average period (adjust as needed)

    # Initialize cash balance to $1,000,000 if current_position is None or 'Cash' column is missing
    if current_position is None or 'Cash' not in current_position:
        current_position = pd.DataFrame({'Cash': [1000000.0]})

    # Extract current cash balance
    current_cash = current_position['Cash'].iloc[-1]

    # Calculate moving averages for SPX and IEI for both short and long periods
    spx_short_ma = calculate_moving_average(spx_order_book, short_ma_window)
    spx_long_ma = calculate_moving_average(spx_order_book, long_ma_window)
    iei_short_ma = calculate_moving_average(iei_order_book, short_ma_window)
    iei_long_ma = calculate_moving_average(iei_order_book, long_ma_window)

    # Trading logic for SPX
    spx_latest_close = spx_order_book['Close_Price'].iloc[-1]
    spx_signal = 0  # 0 = Hold
    spx_qty = 0  # Quantity to buy/sell

    if spx_latest_close > spx_short_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        spx_qty = int(current_cash / (spx_latest_close - spx_order_book['Bid1_Price'].iloc[-1]))
        spx_signal = 1 if spx_qty > 0 else 0  # 1 = Buy if position size is positive

    elif spx_latest_close < spx_short_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        spx_qty = int(current_cash / (spx_order_book['Ask1_Price'].iloc[-1] - spx_latest_close))
        spx_signal = -1 if spx_qty > 0 else 0  # -1 = Sell if position size is positive

    # Implement stop-loss logic for SPX
    if spx_signal == 1:
        stop_loss_price = spx_latest_close * (1 - stop_loss_percentage)
        spx_signal = 0 if spx_order_book['Bid1_Price'].iloc[-1] <= stop_loss_price else 1

    elif spx_signal == -1:
        stop_loss_price = spx_latest_close * (1 + stop_loss_percentage)
        spx_signal = 0 if spx_order_book['Ask1_Price'].iloc[-1] >= stop_loss_price else -1

    # Initialize take-profit flag for SPX
    spx_take_profit = False

    # Implement take-profit logic for SPX
    if spx_signal == 1:
        take_profit_price = spx_latest_close * (1 + take_profit_percentage)
        spx_take_profit = spx_order_book['Ask1_Price'].iloc[-1] >= take_profit_price

    elif spx_signal == -1:
        take_profit_price = spx_latest_close * (1 - take_profit_percentage)
        spx_take_profit = spx_order_book['Bid1_Price'].iloc[-1] <= take_profit_price

    # Check if take-profit condition is met for SPX
    if spx_take_profit:
        spx_signal = 0  # Exit SPX position

    # Trading logic for IEI
    iei_latest_close = iei_order_book['Close_Price'].iloc[-1]
    iei_signal = 0  # 0 = Hold
    iei_qty = 0  # Quantity to buy/sell

    if iei_latest_close > iei_short_ma.iloc[-1] and iei_latest_close > iei_long_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        iei_qty = int(current_cash / (iei_latest_close - iei_order_book['Bid1_Price'].iloc[-1]))
        iei_signal = 1 if iei_qty > 0 else 0  # 1 = Buy if position size is positive

    elif iei_latest_close < iei_short_ma.iloc[-1] and iei_latest_close < iei_long_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        iei_qty = int(current_cash / (iei_order_book['Ask1_Price'].iloc[-1] - iei_latest_close))
        iei_signal = -1 if iei_qty > 0 else 0  # -1 = Sell if position size is positive

    # Implement stop-loss logic for IEI
    if iei_signal == 1:
        stop_loss_price = iei_latest_close * (1 - stop_loss_percentage)
        iei_signal = 0 if iei_order_book['Bid1_Price'].iloc[-1] <= stop_loss_price else 1

    elif iei_signal == -1:
        stop_loss_price = iei_latest_close * (1 + stop_loss_percentage)
        iei_signal = 0 if iei_order_book['Ask1_Price'].iloc[-1] >= stop_loss_price else -1

    # Initialize take-profit flag for IEI
    iei_take_profit = False

    # Implement take-profit logic for IEI
    if iei_signal == 1:
        take_profit_price = iei_latest_close * (1 + take_profit_percentage)
        iei_take_profit = iei_order_book['Ask1_Price'].iloc[-1] >= take_profit_price

    elif iei_signal == -1:
        take_profit_price = iei_latest_close * (1 - take_profit_percentage)
        iei_take_profit = iei_order_book['Bid1_Price'].iloc[-1] <= take_profit_price

    # Check if take-profit condition is met for IEI
    if iei_take_profit:
        iei_signal = 0  # Exit IEI position


    # Calculate volatility for SPX and IEI
    spx_volatility = spx_order_book['Close_Price'].rolling(window=short_ma_window).std() * math.sqrt(252)  # Annualized volatility
    iei_volatility = iei_order_book['Close_Price'].rolling(window=short_ma_window).std() * math.sqrt(252)  # Annualized volatility


    # Reduce the impact of volatility on position sizing adjustments
    spx_qty *= max(1.0, min(1.0 + (spx_volatility.iloc[-1] - 0.05), 2.0))  # Adjust based on SPX volatility (0.05 is a baseline)
    iei_qty *= max(1.0, min(1.0 + (iei_volatility.iloc[-1] - 0.025), 2.0))  # Adjust based on IEI volatility (0.025 is a baseline)

    # Ensure position size does not exceed the maximum allowed
    spx_qty = min(spx_qty, max_position_size)
    iei_qty = min(iei_qty, max_position_size)

    return (spx_signal, spx_latest_close, spx_qty, iei_signal, iei_latest_close, iei_qty)

def format_output(function):
    spx_signal, spx_latest_close, spx_qty, iei_signal, iei_latest_close, iei_qty = function

    return (
    f"""
    SPX Latest Close: {spx_latest_close}"
    SPX Quantity: {spx_qty}
    IEI Signal: {iei_signal}
    IEI Latest Close: {iei_latest_close}
    IEI Quantity: {iei_qty}""")

def plot_trading_strategy(spx_order_book, iei_order_book):
    # Plot SPX and IEI closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(spx_order_book['Date'], spx_order_book['Close_Price'], label='SPX Close Price', color='blue')
    plt.plot(iei_order_book['Date'], iei_order_book['Close_Price'], label='IEI Close Price', color='green')
    plt.title('SPX and IEI Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot SPX short and long moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(spx_order_book['Date'], calculate_moving_average(spx_order_book, short_window), label='SPX Short MA', color='orange')
    plt.plot(spx_order_book['Date'], calculate_moving_average(spx_order_book, long_window), label='SPX Long MA', color='red')
    plt.title('SPX Short and Long Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot IEI short and long moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(iei_order_book['Date'], calculate_moving_average(iei_order_book, short_window), label='IEI Short MA', color='purple')
    plt.plot(iei_order_book['Date'], calculate_moving_average(iei_order_book, long_window), label='IEI Long MA', color='blue')
    plt.title('IEI Short and Long Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


# Call function to get the output of the strategy
print(format_output(trading_algorithm(spx_order_book_df, iei_order_book_df, spx_order_book_df)))

# Call the function to plot the graphs
plot_trading_strategy(spx_order_book_df, iei_order_book_df)


