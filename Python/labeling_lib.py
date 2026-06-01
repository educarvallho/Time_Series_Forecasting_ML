import random
import numpy as np
import pandas as pd
from numba import njit
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


# TREND OR NEUTRAL BASED LABELING
@njit
def calculate_labels(close_data, markup, min_val, max_val):
    labels = []
    for i in range(len(close_data) - max_val):
        rand = random.randint(min_val, max_val)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        if (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels(dataset, markup, min = 1, max = 15) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price movements.

    This function calculates labels indicating buy, sell, or hold signals 
    based on future price movements relative to a given markup percentage.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy and sell signals.
        min (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Hold (price change doesn't meet criteria)
                            - 1: Buy (future price increases by at least 'markup' within 'max' days) 
                       - Rows where 'labels' is 2 (sell signal) are removed.
                       - Rows with missing values (NaN) are removed. 
    """

    # Extract closing prices from the dataset
    close_data = dataset['close'].values

    # Calculate buy/hold labels based on future price movements
    labels = calculate_labels(close_data, markup, min, max)

    # Trim the dataset to match the length of calculated labels
    dataset = dataset.iloc[:len(labels)].copy() 

    # Add the calculated labels as a new column
    dataset['labels'] = labels

    # Remove rows with NaN values (potentially introduced in 'calculate_labels')
    dataset = dataset.dropna()

    # Remove rows where the label is 2 (sell signal). 
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)

    return dataset

@njit
def calculate_labels_clusters(close_data, clusters, markup):
    labels = []
    current_cluster = clusters[0]
    last_price = close_data[0]
    for i in range(1, len(close_data)):
        next_cluster = clusters[i]
        if next_cluster != current_cluster and (abs(close_data[i] - last_price) > markup):
            if close_data[i] > last_price:
                labels.append(0.0)
            else:
                labels.append(1.0)
            current_cluster = next_cluster
            last_price = close_data[i]
        else:
            labels.append(2.0)

    if len(labels) < len(close_data):
        labels.append(2.0)
    return labels

def get_labels_clusters(dataset, markup, num_clusters=20) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=num_clusters)
    dataset['cluster'] = kmeans.fit_predict(dataset[['close']])

    close_data = dataset['close'].values
    clusters = dataset['cluster'].values

    labels = calculate_labels_clusters(close_data, clusters, markup)

    dataset['labels'] = labels
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    dataset = dataset.drop(columns=['cluster'])
    return dataset

@njit
def calculate_labels_one_direction(close_data, markup, min, max, direction):
    labels = []
    for i in range(len(close_data) - max):
        rand = random.randint(min, max)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        if direction == "sell":
            if (future_pr + markup) < curr_pr:
                labels.append(1.0)
            else:
                labels.append(0.0)
        if direction == "buy":
            if (future_pr - markup) > curr_pr:
                labels.append(1.0)
            else:
                labels.append(0.0)
    return labels

def get_labels_one_direction(dataset, markup, min = 1, max = 15, direction = 'buy') -> pd.DataFrame:
    close_data = dataset['close'].values
    labels = calculate_labels_one_direction(close_data, markup, min, max, direction)
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    return dataset

@njit
def calculate_signals(prices, window_sizes, threshold_pct):
    max_window = max(window_sizes)
    signals = []
    for i in range(max_window, len(prices)):
        long_signals = 0
        short_signals = 0
        for window_size in window_sizes:
            window = prices[i-window_size:i]
            resistance = max(window)
            support = min(window)
            current_price = prices[i]
            if current_price > resistance * (1 + threshold_pct):
                long_signals += 1
            elif current_price < support * (1 - threshold_pct):
                short_signals += 1
        if long_signals > short_signals:
            signals.append(0.0) 
        elif short_signals > long_signals:
            signals.append(1.0)
        else:
            signals.append(2.0)
    return signals

def get_labels_multi_window(dataset, window_sizes=[20, 50, 100], threshold_pct=0.02) -> pd.DataFrame:
    prices = dataset['close'].values
    signals = calculate_signals(prices, window_sizes, threshold_pct)
    signals = [2.0] * max(window_sizes) + signals
    dataset['labels'] = signals
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    return dataset

@njit
def calculate_labels_validated_levels(prices, window_size, threshold_pct, min_touches):
    resistance_touches = {}
    support_touches = {}
    labels = []
    for i in range(window_size, len(prices)):
        window = prices[i-window_size:i]
        current_price = prices[i]

        potential_resistance = np.max(window)
        potential_support = np.min(window)

        for level in resistance_touches:
            if abs(current_price - level) <= level * threshold_pct:
                resistance_touches[level] += 1

        for level in support_touches:
            if abs(current_price - level) <= level * threshold_pct:
                support_touches[level] += 1

        if potential_resistance not in resistance_touches:
            resistance_touches[potential_resistance] = 1
        if potential_support not in support_touches:
            support_touches[potential_support] = 1

        valid_resistance = [level for level, touches in resistance_touches.items() if touches >= min_touches]
        valid_support = [level for level, touches in support_touches.items() if touches >= min_touches]

        if valid_resistance and current_price > min(valid_resistance) * (1 + threshold_pct):
            labels.append(0.0)
        elif valid_support and current_price < max(valid_support) * (1 - threshold_pct):
            labels.append(1.0) 
        else:
            labels.append(2.0)

    return labels

def get_labels_validated_levels(dataset, window_size=20, threshold_pct=0.02, min_touches=2) -> pd.DataFrame:
    prices = dataset['close'].values
    
    labels = calculate_labels_validated_levels(prices, window_size, threshold_pct, min_touches)
    
    labels = [2.0] * window_size + labels
    dataset['labels'] = labels
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    return dataset


# MEAN REVERSION WITH RESTRICTIONS BASED LABELING
@njit
def calculate_labels_mean_reversion(close, lvl, markup, min_l, max_l, q):
    labels = np.empty(len(close) - max_l, dtype=np.float64)
    for i in range(len(close) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close[i]
        curr_lvl = lvl[i]
        future_pr = close[i + rand]

        if curr_lvl > q[1] and (future_pr + markup) < curr_pr:
            labels[i] = 1.0
        elif curr_lvl < q[0] and (future_pr - markup) > curr_pr:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_mean_reversion(dataset, markup, min_l=1, max_l=15, rolling=0.5, quantiles=[.45, .55], method='spline', shift=0) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on mean reversion principles.

    This function calculates trading signals (buy/sell) based on the deviation of
    the price from a chosen moving average or smoothing method. It identifies
    potential buy opportunities when the price deviates significantly below its 
    smoothed trend, anticipating a reversion to the mean.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy signals.
        min_l (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max_l (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.
        rolling (float, optional): Rolling window size for smoothing/averaging. 
                                     If method='spline', this controls the spline smoothing factor.
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        method (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data forward (positive) or backward (negative).
                                 Useful for creating a lag/lead effect. Defaults to 0.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate the price deviation ('lvl') based on the chosen method
    if method == 'mean':
        dataset['lvl'] = (dataset['close'] - dataset['close'].rolling(rolling).mean())
    elif method == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling) 
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift
        dataset['lvl'] = dataset['close'] - yHat_shifted
        dataset = dataset.dropna()  # Remove NaN values potentially introduced by spline/shift
    elif method == 'savgol':
        smoothed_prices = savgol_filter(dataset['close'].values, window_length=int(rolling), polyorder=3)
        dataset['lvl'] = dataset['close'] - smoothed_prices

    dataset = dataset.dropna()  # Remove NaN values before proceeding
    q = dataset['lvl'].quantile(quantiles).to_list()  # Calculate quantiles for the 'reversion zone'

    # Prepare data for label calculation
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels 
    labels = calculate_labels_mean_reversion(close, lvl, markup, min_l, max_l, q) 

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)  # Remove sell signals (if any)
    return dataset.drop(columns=['lvl'])  # Remove the temporary 'lvl' column 

@njit
def calculate_labels_mean_reversion_multi(close_data, lvl_data, q, markup, min_l, max_l, windows):
    labels = []
    for i in range(len(close_data) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close_data[i]
        future_pr = close_data[i + rand]

        buy_condition = True
        sell_condition = True
        qq = 0
        for _ in windows:  # Loop over each window, variable unused
            curr_lvl = lvl_data[i, qq]            
            if not (curr_lvl >= q[qq, 1]):  # Access q as 2D array
                sell_condition = False
            if not (curr_lvl <= q[qq, 0]):
                buy_condition = False
            qq += 1
    
        if sell_condition and (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif buy_condition and (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_multi(dataset, markup, min_l=1, max_l=15, windows=[0.2, 0.3, 0.5], quantiles=[.45, .55]) -> pd.DataFrame:
    q = np.empty((len(windows), 2))  # Initialize as 2D NumPy array
    lvl_data = np.empty((dataset.shape[0], len(windows)))

    for i, rolling in enumerate(windows):
        x = np.arange(dataset.shape[0])
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling)
        yHat = spl(np.linspace(x.min(), x.max(), x.shape[0]))
        lvl_data[:, i] = dataset['close'] - yHat
        # Store quantiles directly into the NumPy array
        quantile_values = np.quantile(lvl_data[:, i], quantiles)
        q[i, 0] = quantile_values[0]
        q[i, 1] = quantile_values[1]

    dataset = dataset.dropna()
    close_data = dataset['close'].values

    # Convert windows to a tuple for Numba compatibility (optional)
    labels = calculate_labels_mean_reversion_multi(close_data, lvl_data, q, markup, min_l, max_l, tuple(windows))

    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset[dataset.labels != 2.0]
    
    return dataset

@njit
def calculate_labels_mean_reversion_v(close_data, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, markup, min_l, max_l):
    labels = []
    for i in range(len(close_data) - max_l):
        rand = random.randint(min_l, max_l)
        curr_pr = close_data[i]
        curr_lvl = lvl_data[i]
        curr_vol_group = volatility_group[i]
        future_pr = close_data[i + rand]

        # Access quantiles directly from arrays
        low_q = quantile_groups_low[int(curr_vol_group)]
        high_q = quantile_groups_high[int(curr_vol_group)]

        if curr_lvl > high_q and (future_pr + markup) < curr_pr:
            labels.append(1.0)
        elif curr_lvl < low_q and (future_pr - markup) > curr_pr:
            labels.append(0.0)
        else:
            labels.append(2.0)
    return labels

def get_labels_mean_reversion_v(dataset, markup, min_l=1, max_l=15, rolling=0.5, quantiles=[.45, .55], method='spline', shift=1, volatility_window=20) -> pd.DataFrame:
    """
    Generates trading labels based on mean reversion principles, incorporating
    volatility-based adjustments to identify buy opportunities.

    This function calculates trading signals (buy/sell), taking into account the 
    volatility of the asset. It groups the data into volatility bands and calculates 
    quantiles for each band. This allows for more dynamic "reversion zones" that 
    adjust to changing market conditions.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        markup (float): The percentage markup used to determine buy signals.
        min_l (int, optional): Minimum number of consecutive days the markup must hold. Defaults to 1.
        max_l (int, optional): Maximum number of consecutive days the markup is considered. Defaults to 15.
        rolling (float, optional): Rolling window size or spline smoothing factor (see 'method'). 
                                     Defaults to 0.5.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        method (str, optional): Method for calculating the price deviation:
                                 - 'mean': Deviation from the rolling mean.
                                 - 'spline': Deviation from a smoothed spline.
                                 - 'savgol': Deviation from a Savitzky-Golay filter.
                                 Defaults to 'spline'.
        shift (int, optional): Shift the smoothed price data (lag/lead effect). Defaults to 1.
        volatility_window (int, optional): Window size for calculating volatility. Defaults to 20.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl', 'volatility', 'volatility_group' columns are removed.
    """

    # Calculate Volatility
    dataset['volatility'] = dataset['close'].pct_change().rolling(window=volatility_window).std()
    
    # Divide into 20 groups by volatility 
    dataset['volatility_group'] = pd.qcut(dataset['volatility'], q=20, labels=False)
    
    # Calculate price deviation ('lvl') based on the chosen method
    if method == 'mean':
        dataset['lvl'] = (dataset['close'] - dataset['close'].rolling(rolling).mean())
    elif method == 'spline':
        x = np.array(range(dataset.shape[0]))
        y = dataset['close'].values
        spl = UnivariateSpline(x, y, k=3, s=rolling)
        yHat = spl(np.linspace(min(x), max(x), num=x.shape[0]))
        yHat_shifted = np.roll(yHat, shift=shift) # Apply the shift 
        dataset['lvl'] = dataset['close'] - yHat_shifted
        dataset = dataset.dropna() 
    elif method == 'savgol':
        smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=5)
        dataset['lvl'] = dataset['close'] - smoothed_prices

    dataset = dataset.dropna()
    
    # Calculate quantiles for each volatility group
    quantile_groups = {}
    quantile_groups_low = []
    quantile_groups_high = []
    for group in range(20):
        group_data = dataset[dataset['volatility_group'] == group]['lvl']
        quantiles_values = group_data.quantile(quantiles).to_list()
        quantile_groups[group] = quantiles_values
        quantile_groups_low.append(quantiles_values[0])
        quantile_groups_high.append(quantiles_values[1])

    # Prepare data for label calculation (potentially using Numba)
    close_data = dataset['close'].values
    lvl_data = dataset['lvl'].values
    volatility_group = dataset['volatility_group'].values
    
    # Convert quantile groups to numpy arrays
    quantile_groups_low = np.array(quantile_groups_low)
    quantile_groups_high = np.array(quantile_groups_high)

    # Calculate buy/sell labels 
    labels = calculate_labels_mean_reversion_v(close_data, lvl_data, volatility_group, quantile_groups_low, quantile_groups_high, markup, min_l, max_l)
    
    # Process dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index) # Remove sell signals
    
    # Remove temporary columns and return
    return dataset.drop(columns=['lvl', 'volatility', 'volatility_group'])


# FILTERING BASED LABELING W/O RESTRICTIONS
@njit
def calculate_labels_filter(close, lvl, q):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl = lvl[i]

        if curr_lvl > q[1]:
            labels[i] = 1.0
        elif curr_lvl < q[0]:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_filter(dataset, rolling=200, quantiles=[.45, .55], polyorder=3) -> pd.DataFrame:
    """
    Generates labels for a financial dataset based on price deviation from a Savitzky-Golay filter.

    This function applies a Savitzky-Golay filter to the closing prices to generate a smoothed
    price trend. It then calculates trading signals (buy/sell) based on the deviation of the 
    actual price from this smoothed trend. Buy signals are generated when the price is 
    significantly below the smoothed trend, anticipating a potential price reversal. 

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling (int, optional): Window size for the Savitzky-Golay filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - The temporary 'lvl' column is removed. 
    """

    # Calculate smoothed prices using the Savitzky-Golay filter
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling, polyorder=polyorder)
    
    # Calculate the difference between the actual closing prices and the smoothed prices
    diff = dataset['close'] - smoothed_prices
    dataset['lvl'] = diff  # Add the difference as a new column 'lvl' to the DataFrame
    
    # Remove any rows with NaN values 
    dataset = dataset.dropna()
    
    # Calculate the quantiles of the 'lvl' column (price deviation)
    q = dataset['lvl'].quantile(quantiles).to_list() 

    # Extract the closing prices and the calculated 'lvl' values as NumPy arrays
    close = dataset['close'].values
    lvl = dataset['lvl'].values
    
    # Calculate buy/sell labels using the 'calculate_labels_filter' function 
    labels = calculate_labels_filter(close, lvl, q) 

    # Trim the dataset to match the length of the calculated labels
    dataset = dataset.iloc[:len(labels)].copy()
    
    # Add the calculated labels as a new 'labels' column to the DataFrame
    dataset['labels'] = labels
    
    # Remove any rows with NaN values
    dataset = dataset.dropna()
    
    # Remove rows where the 'labels' column has a value of 2.0 (sell signals)
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    
    # Return the modified DataFrame with the 'lvl' column removed
    return dataset.drop(columns=['lvl']) 

@njit
def calc_labels_multiple_filters(close, lvls, qs):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        label_found = False
        
        for j in range(len(lvls)):
            curr_lvl = lvls[j][i]
            curr_q_low = qs[j][0][i]
            curr_q_high = qs[j][1][i]
            
            if curr_lvl > curr_q_high:
                labels[i] = 1.0
                label_found = True
                break
            elif curr_lvl < curr_q_low:
                labels[i] = 0.0
                label_found = True
                break
                
        if not label_found:
            labels[i] = 2.0
            
    return labels

def get_labels_multiple_filters(dataset, rolling_periods=[200, 400, 600], quantiles=[.45, .55], window=100, polyorder=3) -> pd.DataFrame:
    """
    Generates trading signals (buy/sell) based on price deviation from multiple 
    smoothed price trends calculated using a Savitzky-Golay filter with different
    rolling periods and rolling quantiles. 

    This function applies a Savitzky-Golay filter to the closing prices for each 
    specified 'rolling_period'. It then calculates the price deviation from these
    smoothed trends and determines dynamic "reversion zones" using rolling quantiles.
    Buy signals are generated when the price is within these reversion zones 
    across multiple timeframes.

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling_periods (list, optional): List of rolling window sizes for the Savitzky-Golay filter. 
                                           Defaults to [200, 400, 600].
        quantiles (list, optional): Quantiles to define the "reversion zone". Defaults to [.05, .95].
        window (int, optional): Window size for calculating rolling quantiles. Defaults to 100.
        polyorder (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed. 
    """
    
    # Create a copy of the dataset to avoid modifying the original
    dataset = dataset.copy()
    
    # Lists to store price deviation levels and quantiles for each rolling period
    all_levels = []
    all_quantiles = []
    
    # Calculate smoothed price trends and rolling quantiles for each rolling period
    for rolling in rolling_periods:
        # Calculate smoothed prices using the Savitzky-Golay filter
        smoothed_prices = savgol_filter(dataset['close'].values, 
                                      window_length=rolling, 
                                      polyorder=polyorder)
        # Calculate the price deviation from the smoothed prices
        diff = dataset['close'] - smoothed_prices
        
        # Create a temporary DataFrame to calculate rolling quantiles
        temp_df = pd.DataFrame({'diff': diff})
        
        # Calculate rolling quantiles for the price deviation
        q_low = temp_df['diff'].rolling(window=window).quantile(quantiles[0])
        q_high = temp_df['diff'].rolling(window=window).quantile(quantiles[1])
        
        # Store the price deviation and quantiles for the current rolling period
        all_levels.append(diff)
        all_quantiles.append([q_low.values, q_high.values])
    
    # Convert lists to NumPy arrays for faster calculations (potentially using Numba)
    lvls_array = np.array(all_levels)
    qs_array = np.array(all_quantiles)
    
    # Calculate buy/sell labels using the 'calc_labels_multiple_filters' function 
    labels = calc_labels_multiple_filters(dataset['close'].values, lvls_array, qs_array)
    
    # Add the calculated labels to the DataFrame
    dataset['labels'] = labels
    
    # Remove rows with NaN values and sell signals (labels == 2.0)
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index)
    
    # Return the DataFrame with the new 'labels' column
    return dataset

@njit
def calc_labels_bidirectional(close, lvl1, lvl2, q1, q2):
    labels = np.empty(len(close), dtype=np.float64)
    for i in range(len(close)):
        curr_lvl1 = lvl1[i]
        curr_lvl2 = lvl2[i]

        if curr_lvl1 > q1[1]:
            labels[i] = 1.0
        elif curr_lvl2 < q2[0]:
            labels[i] = 0.0
        else:
            labels[i] = 2.0
    return labels

def get_labels_filter_bidirectional(dataset, rolling1=200, rolling2=200, quantiles=[.45, .55], polyorder=3) -> pd.DataFrame:
    """
    Generates trading labels based on price deviation from two Savitzky-Golay filters applied
    in opposite directions (forward and reversed) to the closing price data.

    This function calculates trading signals (buy/sell) based on the price's 
    position relative to smoothed price trends generated by two Savitzky-Golay filters 
    with potentially different window sizes (`rolling1`, `rolling2`). 

    Args:
        dataset (pd.DataFrame): DataFrame containing financial data with a 'close' column.
        rolling1 (int, optional): Window size for the first Savitzky-Golay filter. Defaults to 200.
        rolling2 (int, optional): Window size for the second Savitzky-Golay filter. Defaults to 200.
        quantiles (list, optional): Quantiles to define the "reversion zones". Defaults to [.45, .55].
        polyorder (int, optional): Polynomial order for both Savitzky-Golay filters. Defaults to 3.

    Returns:
        pd.DataFrame: The original DataFrame with a new 'labels' column and filtered rows:
                       - 'labels' column: 
                            - 0: Buy
                            - 1: Sell
                       - Rows where 'labels' is 2 (no signal) are removed.
                       - Rows with missing values (NaN) are removed.
                       - Temporary 'lvl1' and 'lvl2' columns are removed.
    """

    # Apply the first Savitzky-Golay filter (forward direction)
    smoothed_prices = savgol_filter(dataset['close'].values, window_length=rolling1, polyorder=polyorder)
    
    # Apply the second Savitzky-Golay filter (could be in reverse direction if rolling2 is negative)
    smoothed_prices2 = savgol_filter(dataset['close'].values, window_length=rolling2, polyorder=polyorder)

    # Calculate price deviations from both smoothed price series
    diff1 = dataset['close'] - smoothed_prices
    diff2 = dataset['close'] - smoothed_prices2

    # Add price deviations as new columns to the DataFrame
    dataset['lvl1'] = diff1
    dataset['lvl2'] = diff2
    
    # Remove rows with NaN values 
    dataset = dataset.dropna()

    # Calculate quantiles for the "reversion zones" for both price deviation series
    q1 = dataset['lvl1'].quantile(quantiles).to_list()
    q2 = dataset['lvl2'].quantile(quantiles).to_list()

    # Extract relevant data for label calculation
    close = dataset['close'].values
    lvl1 = dataset['lvl1'].values
    lvl2 = dataset['lvl2'].values
    
    # Calculate buy/sell labels using the 'calc_labels_bidirectional' function
    labels = calc_labels_bidirectional(close, lvl1, lvl2, q1, q2)

    # Process the dataset and labels
    dataset = dataset.iloc[:len(labels)].copy()
    dataset['labels'] = labels
    dataset = dataset.dropna()
    dataset = dataset.drop(dataset[dataset.labels == 2.0].index) # Remove bad signals (if any)
    
    # Return the DataFrame with temporary columns removed
    return dataset.drop(columns=['lvl1', 'lvl2']) 
