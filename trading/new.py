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

# Get data
msci_world = pdr.get_data_yahoo('ACWI', start, end)
us_treasury_10y = pdr.get_data_yahoo('^TNX', start, end)



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
msci_order_book_rows = []
tnx_order_book_rows = []

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

# Generate order book for MSCI
for date, close_price in msci_world['Close'].items():
    msci_order_book_rows.append(create_order_book_row(date, close_price))

# Generate order book for ^TNX
for date, close_price in us_treasury_10y['Close'].items():
    tnx_order_book_rows.append(create_order_book_row(date, close_price))

# Create DataFrames
msci_order_book_df = pd.DataFrame(msci_order_book_rows)
tnx_order_book_df = pd.DataFrame(tnx_order_book_rows)



import matplotlib.pyplot as plt
import math

"""
*We need to define our assumptions (SMART)
*We should include some charts (DONE!)


"""
# Define stop-loss and take-profit percentages
stop_loss_percentage = 0.1  # Increase the stop-loss percentage
take_profit_percentage = 0.1  # Increase the take-profit percentage

# Define windows for MA
short_window = 50
long_window = 200

def calculate_moving_average(data, window):
    return data['Close_Price'].rolling(window=window).mean()


def trading_algorithm(tnx_order_book: pd.DataFrame, msci_order_book: pd.DataFrame, current_position: pd.DataFrame, transactions:pd.DataFrame()) -> tuple:
    # msci_order_book_df = pd.DataFrame(msci_order_book_rows)
    msci_order_book = msci_order_book.copy()
    tnx_order_book = tnx_order_book.copy()


    # Parameters for risk management
    max_position_size = 1000  # Increase the maximum position size in dollars
    short_ma_window = short_window  # Short moving average period (adjust as needed)
    long_ma_window = long_window  # Long moving average period (adjust as needed)

    # Extract current cash balance
    current_cash = current_position['Cash'].iloc[-1]

    # Calculate moving averages for MSCI and TNX for both short and long periods
    tnx_short_ma = calculate_moving_average(tnx_order_book, short_ma_window)
    tnx_long_ma = calculate_moving_average(tnx_order_book, long_ma_window)
    msci_short_ma = calculate_moving_average(msci_order_book, short_ma_window)
    msci_long_ma = calculate_moving_average(msci_order_book, long_ma_window)

    # Trading logic for msci
    msci_latest_close = msci_order_book['Close_Price'].iloc[-1]
    msci_signal = 0  # 0 = Hold
    msci_qty = int(current_cash / msci_order_book['Bid1_Price'].iloc[-1] * 0.05)

    if msci_latest_close > msci_short_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        msci_signal = 1 if msci_qty > 0 else 0  # 1 = Buy if position size is positive

    elif msci_latest_close < msci_short_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        msci_signal = -1 if msci_qty > 0 else 0  # -1 = Sell if position size is positive

    # Implement stop-loss logic for msci
    if msci_signal == 1:
        stop_loss_price = msci_latest_close * (1 - stop_loss_percentage)
        msci_signal = 0 if msci_order_book['Bid1_Price'].iloc[-1] <= stop_loss_price else 1

    elif msci_signal == -1:
        stop_loss_price = msci_latest_close * (1 + stop_loss_percentage)
        msci_signal = 0 if msci_order_book['Ask1_Price'].iloc[-1] >= stop_loss_price else -1

    # Initialize take-profit flag for msci
    msci_take_profit = False

    # Implement take-profit logic for msci
    if msci_signal == 1:
        take_profit_price = msci_latest_close * (1 + take_profit_percentage)
        msci_take_profit = msci_order_book['Ask1_Price'].iloc[-1] >= take_profit_price

    elif msci_signal == -1:
        take_profit_price = msci_latest_close * (1 - take_profit_percentage)
        msci_take_profit = msci_order_book['Bid1_Price'].iloc[-1] <= take_profit_price

    # Check if take-profit condition is met for msci
    if msci_take_profit:
        msci_signal = 0  # Exit msci position

    # Trading logic for tnx
    tnx_latest_close = tnx_order_book['Close_Price'].iloc[-1]
    tnx_signal = 0  # 0 = Hold
    tnx_qty = int(current_cash / msci_order_book['Bid1_Price'].iloc[-1] * 0.05)

    if tnx_latest_close > tnx_short_ma.iloc[-1] and tnx_latest_close > tnx_long_ma.iloc[-1] and current_cash >= (tnx_latest_close - tnx_order_book['Bid1_Price'].iloc[-1]):
        # Calculate position size based on available cash (allowing for larger positions)
        tnx_signal = 1 if tnx_qty > 0 else 0  # 1 = Buy if position size is positive

    elif tnx_latest_close < tnx_short_ma.iloc[-1] and tnx_latest_close < tnx_long_ma.iloc[-1]:
        # Calculate position size based on available cash (allowing for larger positions)
        tnx_signal = -1 if tnx_qty > 0 else 0  # -1 = Sell if position size is positive

    # Implement stop-loss logic for tnx
    if tnx_signal == 1:
        stop_loss_price = tnx_latest_close * (1 - stop_loss_percentage)
        tnx_signal = 0 if tnx_order_book['Bid1_Price'].iloc[-1] <= stop_loss_price else 1

    elif tnx_signal == -1:
        stop_loss_price = tnx_latest_close * (1 + stop_loss_percentage)
        tnx_signal = 0 if tnx_order_book['Ask1_Price'].iloc[-1] >= stop_loss_price else -1

    # Initialize take-profit flag for tnx
    tnx_take_profit = False

    # Implement take-profit logic for tnx
    if tnx_signal == 1:
        take_profit_price = tnx_latest_close * (1 + take_profit_percentage)
        tnx_take_profit = tnx_order_book['Ask1_Price'].iloc[-1] >= take_profit_price

    elif tnx_signal == -1:
        take_profit_price = tnx_latest_close * (1 - take_profit_percentage)
        tnx_take_profit = tnx_order_book['Bid1_Price'].iloc[-1] <= take_profit_price

    # Check if take-profit condition is met for tnx
    if tnx_take_profit:
        tnx_signal = 0  # Exit MSCI position


    # Calculate volatility for msci and MSCI
    tnx_volatility = tnx_order_book['Close_Price'].rolling(window=short_ma_window).std() * math.sqrt(252)  # Annualized volatility
    msci_volatility = msci_order_book['Close_Price'].rolling(window=short_ma_window).std() * math.sqrt(252)  # Annualized volatility


    # Reduce the impact of volatility on position sizing adjustments
    tnx_qty *= max(1.0, min(1.0 + (tnx_volatility.iloc[-1] - 0.05), 2.0))  # Adjust based on TNX volatility (0.05 is a baseline)
    msci_qty *= max(1.0, min(1.0 + (msci_volatility.iloc[-1] - 0.025), 2.0))  # Adjust based on MSCI volatility (0.025 is a baseline)

    # Ensure position size does not exceed the maximum allowed
    tnx_qty = min(tnx_qty, max_position_size)
    msci_qty = min(msci_qty, max_position_size)

    # TODO What to do with correlation?
    # Correlation of both
    corrAll = tnx_order_book.corrwith(msci_order_book, axis=0,numeric_only=True)
    corrClose = corrAll["Close_Price"]

    return (tnx_signal, tnx_latest_close, tnx_qty, msci_signal, msci_latest_close, msci_qty)


def format_output(function):
    tnx_signal, tnx_latest_close, tnx_qty, msci_signal, msci_latest_close, msci_qty = function

    return (
    f"""
    TNX Signal: {tnx_signal}
    TNX Latest Close: {tnx_latest_close}
    TNX Quantity: {tnx_qty}
    MSCI Signal: {msci_signal}
    MSCI Latest Close: {msci_latest_close}
    MSCI Quantity: {msci_qty}""")


def plot_trading_strategy(tnx_order_book, msci_order_book):
    # Plot TNX and MSCI closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(tnx_order_book['Date'], tnx_order_book['Close_Price'], label='TNX Close Price', color='blue')
    plt.plot(msci_order_book['Date'], msci_order_book['Close_Price'], label='MSCI Close Price', color='green')
    plt.title('TNX and MSCI Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot TNX short and long moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(tnx_order_book['Date'], calculate_moving_average(tnx_order_book, short_window), label='TNX Short MA', color='orange')
    plt.plot(tnx_order_book['Date'], calculate_moving_average(tnx_order_book, long_window), label='TNX Long MA', color='red')
    plt.title('TNX Short and Long Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot MSCI short and long moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(msci_order_book['Date'], calculate_moving_average(msci_order_book, short_window), label='MSCI Short MA', color='purple')
    plt.plot(msci_order_book['Date'], calculate_moving_average(msci_order_book, long_window), label='MSCI Long MA', color='blue')
    plt.title('MSCI Short and Long Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################################################
#                               Testing                                       #
###############################################################################

# Initial Position
current_position = pd.DataFrame({
    'Cash': [100000],  # initial cash
    'TNX Position': [0],  # initial SPX position
    'MSCI Position': [0],  # initial IEI position
    'PnL': [0]  # initial PnL
})

# Initialize transactions DataFrame
columns = ['Date', 'Qty', 'Price', 'Symbol', 'Signal']
transactions = pd.DataFrame(columns=columns)

print(tnx_order_book_df['Date'])

for idx, date in enumerate(tnx_order_book_df['Date']):
    # Select the data up to and including the current date
    tnx_current = tnx_order_book_df.iloc[:idx+1]
    msci_current = msci_order_book_df.iloc[:idx+1]

    # Call the trading algorithm to get trading signals
    tnx_signal, tnx_price, tnx_qty, msci_signal, msci_price, msci_qty = trading_algorithm(tnx_current, msci_current, current_position, transactions)
    # Update transactions DataFrame
    if tnx_signal != 0:  # If there's an TNX transaction
        new_transaction = pd.DataFrame({
            'Date': [date],
            'Symbol': ['TNX'],
            'Signal': [tnx_signal],
            'Price': [tnx_price],
            'Qty': [tnx_qty]
        })
        transactions = pd.concat([transactions, new_transaction], ignore_index=True)

    if msci_signal != 0:  # If there's an MSCI transaction
        new_transaction = pd.DataFrame({
            'Date': [date],
            'Symbol': ['ACWI'],
            'Signal': [msci_signal],
            'Price': [msci_price],
            'Qty': [msci_qty]
        })
        transactions = pd.concat([transactions, new_transaction], ignore_index=True)

    # Update positions based on the trading signals
    current_position['TNX Position'] += tnx_signal * tnx_qty
    current_position['MSCI Position'] += msci_signal * msci_qty

    # Update cash based on the trading operations
    current_position['Cash'] -= tnx_signal * tnx_price * tnx_qty
    current_position['Cash'] -= msci_signal * msci_price * msci_qty

    # Update PnL
    tnx_transactions = transactions[transactions['Symbol'] == 'TNX']
    msci_transactions = transactions[transactions['Symbol'] == 'ACWI']

    tnx_cost_basis = (tnx_transactions['Price'] * tnx_transactions['Qty'] * tnx_transactions['Signal']).sum()
    msci_cost_basis = (msci_transactions['Price'] * msci_transactions['Qty'] * msci_transactions['Signal']).sum()

    tnx_latest_close = tnx_current['Close_Price'].iloc[-1]
    msci_latest_close = msci_current['Close_Price'].iloc[-1]

    tnx_pnl = (current_position['TNX Position'] * tnx_latest_close) - tnx_cost_basis
    msci_pnl = (current_position['MSCI Position'] * msci_latest_close) - msci_cost_basis

    current_position['PnL'] = tnx_pnl + msci_pnl

print(current_position.head())

print(transactions.head())







    # Call function to get the output of the strategy
print(format_output(trading_algorithm(tnx_order_book_df, msci_order_book_df, current_position, transactions)))

# Call the function to plot the graphs
#plot_trading_strategy(tnx_order_book_df, msci_order_book_df)

