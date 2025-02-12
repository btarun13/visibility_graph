
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def backtest_buy_hold(data, initial_capital=1000):
    """
    Backtests a buy-and-hold strategy based on a 'tag' column.

    Buys and holds only when tag is 1, otherwise does not hold.

    Args:
        data (pd.DataFrame): DataFrame with 'date', 'open', and 'tag' columns.
        initial_capital (float): Starting portfolio amount.

    Returns:
        pd.DataFrame: DataFrame with added 'position', 'returns', 'portfolio_value',
                      and 'cumulative_returns' columns.
        dict: Dyictionary containing performance metrics.
    """

    df = data.copy()
    df.sort_index(inplace=True)

    df['position'] = 0
    df['position'] = np.where(df['tag'] == 1, 1, 0)  # 1 for buy and hold, 0 for no holding

    df['returns'] = 0.0
    df['portfolio_value'] = initial_capital

    shares_held = 0  # Track the number of shares held
    cash = initial_capital  # Track the cash available

    for i in range(1, len(df)):
        if df['position'][i - 1] == 1 and shares_held == 0:  # Buy if tag is 1 and no shares held
            shares_held = cash / df['open'][i-1] # Buy as many shares as possible with available cash at previous day's open price.
            cash = 0  # Invest all cash

        if shares_held > 0:  # If shares are held, calculate returns
            df['returns'][i] = (df['open'][i] - df['open'][i - 1]) / df['open'][i - 1]

        # Portfolio Value Update
        df['portfolio_value'][i] = cash + shares_held * df['open'][i]  # Portfolio value is cash + value of shares
    
    df['cumulative_returns'] = (df['portfolio_value'] / initial_capital) - 1

    # Performance Metrics Calculation (Same as before)
    total_return = df['cumulative_returns'].iloc[-1]
    peak = df['portfolio_value'].cummax()
    drawdown = (peak - df['portfolio_value']) / peak
    max_drawdown = drawdown.max()

    metrics = {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        # ... (Add other metrics)
    }
        # --- Plotting ---
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Plot 1: Portfolio Value
    plt.subplot(2, 1, 1)  # Create the first subplot
    plt.plot(df.index, df['portfolio_value'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')

    # Plot 2: Cumulative Returns
    plt.subplot(2, 1, 2)  # Create the second subplot
    plt.plot(df.index, df['cumulative_returns'])
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


    return df, metrics

