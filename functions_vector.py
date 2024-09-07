def find_trend_intervals (tohlcvcc_array):
    """
    Находит все восходящие и нисходящие трендовые интервалы

    Аргументы:
    tohlcvcc_array (np.array) - массив с данными по столбцам [time, open, high, low, close, volume, candle_type, candle_index]

    Возвращает:
    trend_intervals_list (list) - список трендовых интервалов в формате [[список восходящих трендовых интервалов],[список нисходящих трендовых интервалов]]
    """

    array = tohlcvcc_array[:, (0, 2, 3, 7)]

    array_lenght = array.shape[0]

    delta = array_lenght

    min_max_index_list = []
    min_max_type_list = []


    high_idx = np.argmax(array[:, 1])  # индекс первого глобального максимума
    low_idx = np.argmin(array[:, 2])  # индекс первого глобального минимума

    extremum = min(high_idx, low_idx)  # первый глобальный экстремум

    while delta > 2:

        if extremum == high_idx:  # Если текущий экстремум - максимум

            extremum_idx = int(array[extremum, 3])   # индекс свечи
            min_max_index_list.append(extremum_idx) # добавили индекс свечи-экстремума в список индексов
            min_max_type_list.append(1) # добавили тип экстремума

            # Нахождение следующего минимума после текущего максимума
            low_idx_pre = np.argmin(array[extremum_idx + 1:, 2]) + 1 # индекс следующего глобального минимума на остаточном массиве

            low_idx = extremum_idx + low_idx_pre # корректировка индекса

            extremum = low_idx

        else:  # Текущий экстремум - минимум

            extremum_idx = int(array[extremum, 3])  # индекс свечи
            min_max_index_list.append(extremum_idx)
            min_max_type_list.append(0)

            # Нахождение следующего максимума после текущего минимума
            high_idx_pre = np.argmax(array[extremum_idx + 1 :, 1]) + 1 # индекс следующего глобального максимума


            high_idx = extremum_idx + high_idx_pre # корректировка индекса


            extremum = high_idx

        delta = array_lenght - extremum


    min_max_index_list_second = min_max_index_list[1:]
    min_max_index_list.pop()
    min_max_type_list.pop(0)

    min_max_array = np.array([min_max_index_list, min_max_index_list_second, min_max_type_list], dtype=int).T

    uptrend_intervals_list = []
    downtrend_intervals_list = []

    for row in min_max_array:
        start, end, direction = row
        if end - start >= 3:  # Проверяем, что длина диапазона не меньше 3
            if direction == 1:
                uptrend_intervals_list.append([start, end])
            elif direction == 0:
                downtrend_intervals_list.append([start, end])

    return [uptrend_intervals_list, downtrend_intervals_list]


def point_location(x1, x2, y1, y2, x, y):
    """
    Определяет положение точки (x, y) относительно прямой, проведенной через точки A (x1, y1) и B (x2, y2).

    Аргументы:
    x1, x2, y1, y2 - координаты двух точек A и B, через которые проходит прямая (трендовая)
    x, y - координаты точки, положение которой относительно прямой нас интересует

    Возвращает:
    -1 - если точка (x, y) находится ниже прямой
     0 - если точка (x, y) находится на прямой
     1 - если точка (x, y) находится выше прямой
    """
    # Вычисляем значение выражения (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)
    delta = (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)

    if delta > 0:
        return -1  # Точка ниже прямой
    elif delta < 0:
        return 1   # Точка выше прямой
    else:
        return 0   # Точка на прямой


def find_uptrendlines_on_interval (tohlcvcc_array, trendinterval_first_idx, trendinterval_last_idx):

    array = tohlcvcc_array[trendinterval_first_idx: trendinterval_last_idx, (0,3,7)]

    x1 = array[0,0]
    y1 = array[0,1]

    delta = array.shape[0]

    array_points = np.array([array[0]]) # первая строка (точка) результирующего array_points (содержит в себе последовательность точек - оснований трендовых лучей на трендовом интервале)

    array = np.delete(array, 0, axis=0) # удаляем строку, соответствующую первой точке трендовой линии

    while delta >=3 :

        slope_coeffs = (array[:, 1] - y1)/(array[:, 0] - x1) # вектор угловых коэффициентов

        array = np.column_stack((array, slope_coeffs)) # добавляем столбец с угловымми коэффициентами к первоначальному array

        min_slope_index = np.argmin(array[:, 3])  # индекс строки с минимальным угловым коэффициентом

        min_slope_row = array[min_slope_index, :3] # стока массива, соответствующая минимальному угловому коэффициенту

        array_points = np.vstack([array_points, min_slope_row]) # следующая строка (точка) результирующего array_points (содержит в себе последовательность точек - оснований трендовых лучей на трендовом интервале)

        array = array [min_slope_index :,: 3]

        x1 = array[0,0]
        y1 = array[0,1]

        array = np.delete(array, 0, axis=0) # удаляем строку, соответствующую первой точке трендовой линии

        delta = array.shape[0]

    index_column = array_points[:, -1]
    uptrendlines_list = [[int(index_column[i]), int(index_column[i + 1])] for i in range(len(index_column) - 1)]

    return uptrendlines_list

def find_downtrendlines_on_interval (tohlcvcc_array, trendinterval_first_idx, trendinterval_last_idx):

    array = tohlcvcc_array[ trendinterval_first_idx: trendinterval_last_idx, (0,2,7)]

    x1 = array[0,0]
    y1 = array[0,1]

    delta = array.shape[0]

    array_points = np.array([array[0]]) # первая строка (точка) результирующего array_points (содержит в себе последовательность точек - оснований трендовых лучей на трендовом интервале)

    array = np.delete(array, 0, axis=0) # удаляем строку, соответствующую первой точке трендовой линии

    while delta >=3 :

        slope_coeffs = abs(array[:, 1] - y1)/(array[:, 0] - x1) # вектор угловых коэффициентов

        array = np.column_stack((array, slope_coeffs)) # добавляем столбец с угловымми коэффициентами к первоначальному array

        min_slope_index = np.argmin(array[:, 3])  # индекс строки с минимальным угловым коэффициентом

        min_slope_row = array[min_slope_index, :3] # стока массива, соответствующая минимальному угловому коэффициенту

        array_points = np.vstack([array_points, min_slope_row]) # следующая строка (точка) результирующего array_points (содержит в себе последовательность точек - оснований трендовых лучей на трендовом интервале)

        array = array [min_slope_index :,: 3]

        x1 = array[0,0]
        y1 = array[0,1]

        array = np.delete(array, 0, axis=0) # удаляем строку, соответствующую первой точке трендовой линии

        delta = array.shape[0]

    index_column = array_points[:, -1]
    downtrendlines_list = [[int(index_column[i]), int(index_column[i + 1])] for i in range(len(index_column) - 1)]

    return downtrendlines_list


def find_all_trendlines_vector(df):
    """
    Функция find_all_trendlines (df) составляет датафрейм из всех трендовых линий, действительных на момент закрытия последней свечи входного OHLC датафрейма.

    НА ВХОД:
    OHLC датафрейм

    НА ВЫХОД:
    датафрейм с подробной информацией по всем трендовым линиям, действительным на момент закрытия последней свечи входного OHLC датафрейма

    Колонки результата:
    'Up/Down' /str/ - восходящая или нисходящая ('up' / 'down')
    'A_time' /time/ - время точки A
    'B_time' /time/ - время точки B
    'Trnd_base' /int/ - "база" - на сколько свечей "опирается" трендовая (количество свечей между точками A и B)
    'Tg_Alpha' /float/ - тангенс угла наклона трендовой. Вычисляется как частное: (% изменения цены) / (количество свечей между точками A и B трендовой)
    'A_id' /int/ - индекс точки A
    'B_id' /int/ - индекс точки B
    'A_price' /float/ - цена точки A
    'B_price' /float/ - цена точки B
    'Encl.' /int/ - является ли трендовая вложенной на этом интервале (0 - не является, 1 - является 1-й вложенной, 2 - 2-й и т.д.)
    'Brk_price' /float/ - цена, при достижении которой ломается трендовая (цена точки A)
    'Trnd_int' /list/ - трендовый интервал
    'Irr_price' /float/ - цена хая/лоя трендового интервала. На основании этой цены рассчитывается уровень цены, при достижении которой трендовая становится неактуальной.
      Восходящая трендовая становится неактуальной, если точка пересечения перпендикуляра текущей цены и трендовой выше хая трендового интервала.
      Нисходящая трендовая становится неактуальной, если точка пересечения перпендикуляра текущей цены и трендовой ниже лоя трендового интервала.
    """
    # Преобразование входного датафрейма в массив для дальнейших расчетов
    array = convert_df_to_array(df)

    # Определение времени текущей свечи (последней) для актуальности трендовых линий
    x_current = df['timestamp'].iloc[-1]

    # Определение столбцов для результирующего датафрейма
    trend_lines_df_columns = [
        'up/down', 'a_timestamp', 'b_timestamp', 'first_point_time', 'second_point_time',
        'candles_base', 'tg_alpha', 'first_point_id', 'second_point_id', 'first_point_price',
        'second_point_price', 'enclosured', 'break_price', 'irrelevant_price', 'trend_interval',
        'time_base', 'is_actual']


    # Получение датафрейма трендовых периодов
    trend_intervals_list = find_trend_intervals(array)

    # Определение списков восходящих и нисходящих трендовых интервалов
    up_intervals_list = trend_intervals_list[0]
    down_intervals_list = trend_intervals_list[1]

    def find_all_uptrend_lines(array, uptrend_intervals_list):
        """
        Нахождение всех восходящих трендовых линий на основе предоставленных интервалов.
        """
        uptrendlines_df = pd.DataFrame(columns=['up/down', 'trend_interval', 'first_point_id', 'second_point_id'])
        for uptrend_interval in uptrend_intervals_list:
            uptrendlines_list = find_uptrendlines_on_interval(array, uptrend_interval[0], uptrend_interval[1])
            if len(uptrendlines_list) > 0:
                for uptrendline in uptrendlines_list:
                    uptrend_interval_str = str(uptrend_interval)
                    uptrendline_df = pd.DataFrame({
                        'up/down': 'up',
                        'trend_interval': uptrend_interval_str,
                        'first_point_id': uptrendline[0],
                        'second_point_id': uptrendline[1]
                    }, index=[0])
                    uptrendlines_df = pd.concat([uptrendlines_df, uptrendline_df], ignore_index=True)
        return uptrendlines_df

    def is_uptrendline_actual(array, x1, x2, y1, y2, trend_int):
        """
        Проверка актуальности восходящей трендовой линии.
        """
        trend_int_high_id = ast.literal_eval(trend_int)[1]
        trend_int_high = array[trend_int_high_id][2]
        y_intersection = (x_current * (y2 - y1) / (x2 - x1) + y1 - x1 * ((y2 - y1) / (x2 - x1)))
        return 1 if y_intersection < trend_int_high else 0

    def find_all_downtrend_lines(array, downtrend_intervals_list):
        """
        Нахождение всех нисходящих трендовых линий на основе предоставленных интервалов.
        """
        downtrendlines_df = pd.DataFrame(columns=['up/down', 'trend_interval', 'first_point_id', 'second_point_id'])
        for downtrend_interval in downtrend_intervals_list:
            downtrendlines_list = find_downtrendlines_on_interval(array, downtrend_interval[0], downtrend_interval[1])
            if len(downtrendlines_list) > 0:
                for downtrendline in downtrendlines_list:
                    downtrend_interval_str = str(downtrend_interval)
                    downtrendline_df = pd.DataFrame({
                        'up/down': 'down',
                        'trend_interval': downtrend_interval_str,
                        'first_point_id': downtrendline[0],
                        'second_point_id': downtrendline[1]
                    }, index=[0])
                    downtrendlines_df = pd.concat([downtrendlines_df, downtrendline_df], ignore_index=True)
        return downtrendlines_df

    def is_downtrendline_actual(array, x1, x2, y1, y2, trend_int):
        """
        Проверка актуальности нисходящей трендовой линии.
        """
        trend_int_low_id = ast.literal_eval(trend_int)[1]
        trend_int_low = array[trend_int_low_id][3]
        y_intersection = (x_current * (y2 - y1) / (x2 - x1) + y1 - x1 * ((y2 - y1) / (x2 - x1)))
        return 1 if y_intersection > trend_int_low else 0

    # Нахождение всех восходящих трендовых линий
    all_uptrend_lines_df = find_all_uptrend_lines(array, up_intervals_list)

    if not all_uptrend_lines_df.empty:
        all_uptrend_lines_df['first_point_price'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['second_point_price'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['first_point_time'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_uptrend_lines_df['second_point_time'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_uptrend_lines_df['a_timestamp'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_uptrend_lines_df['b_timestamp'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_uptrend_lines_df['break_price'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['candles_base'] = all_uptrend_lines_df.apply(lambda x: x['second_point_id'] - x['first_point_id'], axis=1)
        all_uptrend_lines_df['enclosured'] = all_uptrend_lines_df.apply(lambda x: 0 if x['first_point_id'] == ast.literal_eval(x['trend_interval'])[0] else 1, axis=1)
        all_uptrend_lines_df['irrelevant_price'] = all_uptrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'low'])
        all_uptrend_lines_df['time_base'] = all_uptrend_lines_df.apply(lambda x: x['second_point_time'] - x['first_point_time'], axis=1)
        all_uptrend_lines_df['tg_alpha'] = all_uptrend_lines_df.apply(lambda x: abs(x['second_point_price'] - x['first_point_price']) / x['second_point_price'] / x['candles_base'], axis=1)
        all_uptrend_lines_df['is_actual'] = all_uptrend_lines_df.apply(lambda x: is_uptrendline_actual(array, x['a_timestamp'], x['b_timestamp'], x['first_point_price'], x['second_point_price'], x['trend_interval']), axis=1)

        # Удаление неактуальных восходящих трендовых линий
        all_uptrend_lines_df.drop(all_uptrend_lines_df[all_uptrend_lines_df['is_actual'] == 0].index, inplace=True)
        all_uptrend_lines_df.drop('is_actual', axis=1, inplace=True)

    # Нахождение всех нисходящих трендовых линий
    all_downtrend_lines_df = find_all_downtrend_lines(array, down_intervals_list)

    if not all_downtrend_lines_df.empty:
        all_downtrend_lines_df['first_point_price'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['second_point_price'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['first_point_time'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_downtrend_lines_df['second_point_time'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_downtrend_lines_df['a_timestamp'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_downtrend_lines_df['b_timestamp'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_downtrend_lines_df['break_price'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['candles_base'] = all_downtrend_lines_df.apply(lambda x: x['second_point_id'] - x['first_point_id'], axis=1)
        all_downtrend_lines_df['enclosured'] = all_downtrend_lines_df.apply(lambda x: 0 if x['first_point_id'] == ast.literal_eval(x['trend_interval'])[0] else 1, axis=1)
        all_downtrend_lines_df['irrelevant_price'] = all_downtrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'high'])
        all_downtrend_lines_df['time_base'] = all_downtrend_lines_df.apply(lambda x: x['second_point_time'] - x['first_point_time'], axis=1)
        all_downtrend_lines_df['tg_alpha'] = all_downtrend_lines_df.apply(lambda x: abs(x['second_point_price'] - x['first_point_price']) / x['second_point_price'] / x['candles_base'], axis=1)
        all_downtrend_lines_df['is_actual'] = all_downtrend_lines_df.apply(lambda x: is_downtrendline_actual(array, x['a_timestamp'], x['b_timestamp'], x['first_point_price'], x['second_point_price'], x['trend_interval']), axis=1)

        # Удаление неактуальных нисходящих трендовых линий
        all_downtrend_lines_df.drop(all_downtrend_lines_df[all_downtrend_lines_df['is_actual'] == 0].index, inplace=True)
        all_downtrend_lines_df.drop('is_actual', axis=1, inplace=True)

    # Объединение датафреймов восходящих и нисходящих трендовых линий
    actual_trendlines_df = pd.concat([all_uptrend_lines_df, all_downtrend_lines_df], ignore_index=True)

    # Определение порядка столбцов в итоговом датафрейме
    columns_list = [
        'up/down', 'first_point_time', 'second_point_time', 'trend_interval', 'time_base',
        'first_point_price', 'second_point_price', 'first_point_id', 'second_point_id',
        'break_price', 'candles_base', 'enclosured', 'irrelevant_price', 'tg_alpha',
        'a_timestamp', 'b_timestamp']
    actual_trendlines_df = actual_trendlines_df[columns_list]

    return actual_trendlines_df
