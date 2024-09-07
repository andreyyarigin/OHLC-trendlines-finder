import numpy as np
import ast
import pandas as pd
import datetime
import time

# Convert otohlcvcc_df to np.array
def convert_df_to_array(df):
    tohlcvcc_df = df.drop(df.columns[0], axis=1)  # Drop the 'open_time' column
    array = tohlcvcc_df.values
    return array


def point_location(x1, x2, y1, y2, x, y):
    """
    Determines the position of a point (x, y) relative to a line defined by points A (x1, y1) and B (x2, y2).

    INPUT:
    x1, x2, y1, y2 - Coordinates of two points A and B that define the line (trend line)
    x, y - Coordinates of the point whose position relative to the line is of interest

    OUTPUT:
    -1 if the point (x, y) is below the line
    0 if the point (x, y) is on the line
    1 if the point (x, y) is above the line
    """
    # Calculate the value of the expression (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)
    delta = (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)

    if delta > 0:
        return -1  # Point is below the line
    elif delta < 0:
        return 1   # Point is above the line
    else:
        return 0   # Point is on the line
    

def find_trend_intervals(tohlcvcc_array):
    """
    Finds all ascending and descending trend intervals.

    INPUT:
    tohlcvcc_array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]

    OUTPUT:
    trend_intervals_list (list) - List of trend intervals in the format [[list of ascending trend intervals], [list of descending trend intervals]]
    """

    array = tohlcvcc_array[:, (0, 2, 3, 7)]
    array_length = array.shape[0]
    delta = array_length

    min_max_index_list = []
    min_max_type_list = []

    high_idx = np.argmax(array[:, 1])  # Index of the first global maximum
    low_idx = np.argmin(array[:, 2])   # Index of the first global minimum
    extremum = min(high_idx, low_idx)  # Index of the first global extremum

    while delta > 2:  # Minimum length of remaining interval for finding global extrema

        if extremum == high_idx:  # If the current extremum is a maximum

            extremum_idx = int(array[extremum, 3])  # Index of the global maximum candle
            min_max_index_list.append(extremum_idx)  # Add index to extremum index list
            min_max_type_list.append(1)  # Add extremum type (1: 'max', 0: 'min')

            # Find the next minimum after the current maximum
            low_idx_pre = np.argmin(array[extremum_idx + 1:, 2]) + 1  # Index of the next global minimum in the remaining array
            low_idx = extremum_idx + low_idx_pre  # Adjust index
            extremum = low_idx

        else:  # If the current extremum is a minimum

            extremum_idx = int(array[extremum, 3])  # Index of the global minimum candle
            min_max_index_list.append(extremum_idx)  # Add index to extremum index list
            min_max_type_list.append(0)  # Add extremum type (1: 'max', 0: 'min')

            # Find the next maximum after the current minimum
            high_idx_pre = np.argmax(array[extremum_idx + 1:, 1]) + 1  # Index of the next global maximum
            high_idx = extremum_idx + high_idx_pre  # Adjust index
            extremum = high_idx

        delta = array_length - extremum  # Length of the remaining interval for finding extrema

    # Create the array of trend intervals
    min_max_index_list_second = min_max_index_list[1:]
    min_max_index_list.pop()
    min_max_type_list.pop(0)
    min_max_array = np.array([min_max_index_list, min_max_index_list_second, min_max_type_list], dtype=int).T
    uptrend_intervals_list = []
    downtrend_intervals_list = []

    # Form lists of ascending and descending trend interval indices
    for row in min_max_array:
        first_id, last_id, direction = row
        if last_id - first_id >= 3:  # Minimum length of the trend interval
            if direction == 1:
                uptrend_intervals_list.append([first_id, last_id])
            elif direction == 0:
                downtrend_intervals_list.append([first_id, last_id])

    return [uptrend_intervals_list, downtrend_intervals_list]


def find_uptrendlines_on_interval(tohlcvcc_array, trendinterval_first_idx, trendinterval_last_idx):
    """
    Finds uptrend lines within a specified uptrend interval

    INPUT:
    tohlcvcc_array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    trendinterval_first_idx (int) - Starting index of the trend interval
    trendinterval_last_idx (int) - Ending index of the trend interval

    OUTPUT:
    uptrendlines_list (list) - List of uptrend lines as pairs of indices
    """
    array = tohlcvcc_array[trendinterval_first_idx: trendinterval_last_idx, (0, 3, 7)]

    x1 = array[0, 0]
    y1 = array[0, 1]

    delta = array.shape[0]
    array_points = np.array([array[0]])  # Initial point for trend lines within the interval
    array = np.delete(array, 0, axis=0)  # Remove the first point from the array

    while delta >= 3:
        slope_coeffs = (array[:, 1] - y1) / (array[:, 0] - x1)  # Vector of slope coefficients
        array = np.column_stack((array, slope_coeffs))  # Add slope coefficients column to the array
        min_slope_index = np.argmin(array[:, 3])  # Index of the row with the minimum slope coefficient
        min_slope_row = array[min_slope_index, :3]  # Row corresponding to the minimum slope coefficient
        array_points = np.vstack([array_points, min_slope_row])  # Add new point to the array_points

        array = array[min_slope_index:, :3]

        x1 = array[0, 0]
        y1 = array[0, 1]

        array = np.delete(array, 0, axis=0)  # Remove the first point from the array
        delta = array.shape[0]

    index_column = array_points[:, -1]
    uptrendlines_list = [[int(index_column[i]), int(index_column[i + 1])] for i in range(len(index_column) - 1)]

    return uptrendlines_list


def find_downtrendlines_on_interval(tohlcvcc_array, trendinterval_first_idx, trendinterval_last_idx):
    """
    Finds downtrend lines within a specified downtrend interval

    INPUT:
    tohlcvcc_array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    trendinterval_first_idx (int) - Starting index of the trend interval
    trendinterval_last_idx (int) - Ending index of the trend interval

    OUTPUT:
    downtrendlines_list (list) - List of downtrend lines as pairs of indices
    """
    array = tohlcvcc_array[trendinterval_first_idx: trendinterval_last_idx, (0, 2, 7)]

    x1 = array[0, 0]
    y1 = array[0, 1]

    delta = array.shape[0]
    array_points = np.array([array[0]])  # Initial point for trend lines within the interval
    array = np.delete(array, 0, axis=0)  # Remove the first point from the array

    while delta >= 3:
        slope_coeffs = abs(array[:, 1] - y1) / (array[:, 0] - x1)  # Vector of slope coefficients
        array = np.column_stack((array, slope_coeffs))  # Add slope coefficients column to the array
        min_slope_index = np.argmin(array[:, 3])  # Index of the row with the minimum slope coefficient
        min_slope_row = array[min_slope_index, :3]  # Row corresponding to the minimum slope coefficient
        array_points = np.vstack([array_points, min_slope_row])  # Add new point to the array_points
        array = array[min_slope_index:, :3]

        x1 = array[0, 0]
        y1 = array[0, 1]

        array = np.delete(array, 0, axis=0)  # Remove the first point from the array
        delta = array.shape[0]

    index_column = array_points[:, -1]
    downtrendlines_list = [[int(index_column[i]), int(index_column[i + 1])] for i in range(len(index_column) - 1)]

    return downtrendlines_list


def find_all_uptrend_lines(array, uptrend_intervals_list):
    """
    Determines all uptrend lines for all uptrend intervals.

    INPUT:
    array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    uptrend_intervals_list (list) - List of uptrend intervals as pairs of indices

    OUTPUT:
    uptrendlines_df (pd.DataFrame) - DataFrame with all uptrend lines and their details
    """
    uptrendlines_df = pd.DataFrame(columns=['up/down', 'trend_interval', 'a_id', 'b_id'])
    for uptrend_interval in uptrend_intervals_list:
        uptrendlines_list = find_uptrendlines_on_interval(array, uptrend_interval[0], uptrend_interval[1])
        if len(uptrendlines_list) > 0:
            for uptrendline in uptrendlines_list:
                uptrend_interval_str = str(uptrend_interval)
                uptrendline_df = pd.DataFrame({
                    'up/down': 'up',
                    'trend_interval': uptrend_interval_str,
                    'a_id': uptrendline[0],
                    'b_id': uptrendline[1]
                }, index=[0])
                uptrendlines_df = pd.concat([uptrendlines_df, uptrendline_df], ignore_index=True)

    return uptrendlines_df


def is_uptrendline_actual(array, x1, x2, y1, y2, trend_int):
    """
    Checks if an uptrend line is still valid.

    INPUT:
    array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    x1, x2, y1, y2 - Coordinates of two points defining the trend line
    trend_int (str) - Trend interval as a string

    OUTPUT:
    int - 1 if the uptrend line is still valid, 0 otherwise
    """
    # Define the current candle's time (last one) to determine the relevance of trendlines
    x_current = array[-1,0]
    
    trend_int_high_id = ast.literal_eval(trend_int)[1]
    trend_int_high = array[trend_int_high_id][2]
    y_intersection = (x_current * (y2 - y1) / (x2 - x1) + y1 - x1 * ((y2 - y1) / (x2 - x1)))

    return 1 if y_intersection < trend_int_high else 0

def find_all_downtrend_lines(array, downtrend_intervals_list):
    """
    Determines all downtrend lines for all downtrend intervals.

    INPUT:
    array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    downtrend_intervals_list (list) - List of downtrend intervals as pairs of indices

    OUTPUT:
    downtrendlines_df (pd.DataFrame) - DataFrame with all downtrend lines and their details
    """
    downtrendlines_df = pd.DataFrame(columns=['up/down', 'trend_interval', 'a_id', 'b_id'])
    for downtrend_interval in downtrend_intervals_list:
        downtrendlines_list = find_downtrendlines_on_interval(array, downtrend_interval[0], downtrend_interval[1])
        if len(downtrendlines_list) > 0:
            for downtrendline in downtrendlines_list:
                downtrend_interval_str = str(downtrend_interval)
                downtrendline_df = pd.DataFrame({
                    'up/down': 'down',
                    'trend_interval': downtrend_interval_str,
                    'a_id': downtrendline[0],
                    'b_id': downtrendline[1]
                }, index=[0])
                downtrendlines_df = pd.concat([downtrendlines_df, downtrendline_df], ignore_index=True)
    return downtrendlines_df


def is_downtrendline_actual(array, x1, x2, y1, y2, trend_int):
    """
    Checks if a downtrend line is still valid.

    INPUT:
    array (np.array) - Array with columns [time, open, high, low, close, volume, candle_type, candle_index]
    x1, x2, y1, y2 - Coordinates of two points defining the trend line
    trend_int (str) - Trend interval as a string

    OUTPUT:
    int - 1 if the downtrend line is still valid, 0 otherwise
    """
    # Define the current candle's time (last one) to determine the relevance of trendlines
    x_current = array[-1,0]

    trend_int_low_id = ast.literal_eval(trend_int)[1]
    trend_int_low = array[trend_int_low_id][3]
    y_intersection = (x_current * (y2 - y1) / (x2 - x1) + y1 - x1 * ((y2 - y1) / (x2 - x1)))

    return 1 if y_intersection > trend_int_low else 0


def find_all_trendlines(df):
    """
    Creates a DataFrame of all trendlines that are valid at the time of the last candle's close in the input OHLC DataFrame.

    INPUT:
    OHLC DataFrame:
    'timestamp' /int/
    'open' /float/
    'high' /float/
    'low' /float/
    'close' /float/
    'volume' /float/

    OUTPUT:
    A DataFrame with detailed information on all trendlines valid at the time of the last candle's close in the input OHLC DataFrame.

    Result columns:

    'up/down' /str/ - ascending or descending ('up' / 'down')
    'a_time' /datetime/ - time at point A
    'b_time' /datetime/ - time at point B
    'trend_interval' /list/ - trend interval where the trendline was formed
    'time_base' /timedelta/ - time interval between points A and B on which the trendline is based
    'candles_base' /int/ - the number of candles between points A and B
    'a_price' /float/ - price at point A
    'b_price' /float/ - price at point B
    'a_id' /int/ - ID of the candle at point A
    'b_id' /int/ - ID of the candle at point B
    'break_price' /float/ - price at which the trendline breaks (price at point A)
    'enclosured' /int/ - whether the trendline is enclosed at this interval (0 - not enclosed, 1 - first enclosed, 2 - second, etc.)
    'irrelevant_price' /float/ - The price at the high/low point of the trend interval. This is used to calculate the price level at which the trendline becomes irrelevant.
                                An ascending trendline becomes irrelevant if the intersection point of the current price and trendline is above the trend interval's high.
                                A descending trendline becomes irrelevant if the intersection point is below the trend interval's low.
    'tg_alpha' /float/ - The "tangent" of the trendline's slope, calculated as: (percentage price change) / (number of candles between points A and B)
    'a_timestamp' /int/ - timestamp at point A
    'b_timestamp' /int/ - timestamp at point B
    """

    # Create np.array from DataFrame
    tohlcvcc_array = convert_df_to_array (df)

    # List of trend intervals
    trend_intervals_list = find_trend_intervals(tohlcvcc_array)

    # Lists of ascending and descending trend intervals
    up_intervals_list = trend_intervals_list[0]
    down_intervals_list = trend_intervals_list[1]

    # Find all uptrend lines
    all_uptrend_lines_df = find_all_uptrend_lines(tohlcvcc_array, up_intervals_list)

    if not all_uptrend_lines_df.empty:
        # Add additional columns for uptrend lines
        all_uptrend_lines_df['a_price'] = all_uptrend_lines_df['a_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['b_price'] = all_uptrend_lines_df['b_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['a_time'] = all_uptrend_lines_df['a_id'].apply(lambda id: df.at[id, 'open_time'])
        all_uptrend_lines_df['b_time'] = all_uptrend_lines_df['b_id'].apply(lambda id: df.at[id, 'open_time'])
        all_uptrend_lines_df['a_timestamp'] = all_uptrend_lines_df['a_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_uptrend_lines_df['b_timestamp'] = all_uptrend_lines_df['b_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_uptrend_lines_df['break_price'] = all_uptrend_lines_df['a_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['candles_base'] = all_uptrend_lines_df.apply(lambda x: x['b_id'] - x['a_id'], axis=1)
        all_uptrend_lines_df['enclosured'] = all_uptrend_lines_df.apply(lambda x: 0 if x['a_id'] == ast.literal_eval(x['trend_interval'])[0] else 1, axis=1)
        all_uptrend_lines_df['irrelevant_price'] = all_uptrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'low'])
        all_uptrend_lines_df['time_base'] = all_uptrend_lines_df.apply(lambda x: x['b_time'] - x['a_time'], axis=1)
        all_uptrend_lines_df['tg_alpha'] = all_uptrend_lines_df.apply(lambda x: abs(x['b_price'] - x['a_price']) / x['b_price'] / x['candles_base'], axis=1)
        all_uptrend_lines_df['is_actual'] = all_uptrend_lines_df.apply(lambda x: is_uptrendline_actual(tohlcvcc_array, x['a_timestamp'], x['b_timestamp'], x['a_price'], x['b_price'], x['trend_interval']), axis=1)

        # Remove non-actual uptrend lines
        all_uptrend_lines_df = all_uptrend_lines_df[all_uptrend_lines_df['is_actual'] == 1]
        all_uptrend_lines_df.drop('is_actual', axis=1, inplace=True)

    # Find all downtrend lines
    all_downtrend_lines_df = find_all_downtrend_lines(tohlcvcc_array, down_intervals_list)

    if not all_downtrend_lines_df.empty:
        # Add additional columns for downtrend lines
        all_downtrend_lines_df['a_price'] = all_downtrend_lines_df['a_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['b_price'] = all_downtrend_lines_df['b_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['a_time'] = all_downtrend_lines_df['a_id'].apply(lambda id: df.at[id, 'open_time'])
        all_downtrend_lines_df['b_time'] = all_downtrend_lines_df['b_id'].apply(lambda id: df.at[id, 'open_time'])
        all_downtrend_lines_df['a_timestamp'] = all_downtrend_lines_df['a_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_downtrend_lines_df['b_timestamp'] = all_downtrend_lines_df['b_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_downtrend_lines_df['break_price'] = all_downtrend_lines_df['a_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['candles_base'] = all_downtrend_lines_df.apply(lambda x: x['b_id'] - x['a_id'], axis=1)
        all_downtrend_lines_df['enclosured'] = all_downtrend_lines_df.apply(lambda x: 0 if x['a_id'] == ast.literal_eval(x['trend_interval'])[0] else 1, axis=1)
        all_downtrend_lines_df['irrelevant_price'] = all_downtrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'high'])
        all_downtrend_lines_df['time_base'] = all_downtrend_lines_df.apply(lambda x: x['b_time'] - x['a_time'], axis=1)
        all_downtrend_lines_df['tg_alpha'] = all_downtrend_lines_df.apply(lambda x: abs(x['b_price'] - x['a_price']) / x['b_price'] / x['candles_base'], axis=1)
        all_downtrend_lines_df['is_actual'] = all_downtrend_lines_df.apply(lambda x: is_downtrendline_actual(tohlcvcc_array, x['a_timestamp'], x['b_timestamp'], x['a_price'], x['b_price'], x['trend_interval']), axis=1)

        # Remove non-actual downtrend lines
        all_downtrend_lines_df = all_downtrend_lines_df[all_downtrend_lines_df['is_actual'] == 1]
        all_downtrend_lines_df.drop('is_actual', axis=1, inplace=True)

    # Combine uptrend and downtrend lines into a single DataFrame
    actual_trendlines_df = pd.concat([all_uptrend_lines_df, all_downtrend_lines_df], ignore_index=True)

    # Define column order for the final DataFrame
    columns_list = ['up/down',
                    'a_time',
                    'b_time',
                    'trend_interval',
                    'time_base',
                    'candles_base',
                    'a_price',
                    'b_price',
                    'a_id',
                    'b_id',
                    'break_price',
                    'enclosured',
                    'irrelevant_price',
                    'tg_alpha',
                    'a_timestamp',
                    'b_timestamp']
    actual_trendlines_df = actual_trendlines_df[columns_list]

    return actual_trendlines_df

