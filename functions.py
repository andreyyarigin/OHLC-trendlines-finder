import numpy as np
import ast
import pandas as pd
import datetime
import os
from pathlib import Path

csv_file_path = '/content/XRPUSDT_OHLCV_1D.csv'

df = pd.read_csv(csv_file_path, header = None, names = ('timestamp', 'open', 'high', 'low', 'close', 'volume'))

df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
df['candle_type'] = df.apply(lambda x: 1 if x['open'] <= x['close'] else 0, axis = 1)
df.reset_index(inplace = True, names = 'candle_id')
new_columns_order = ['open_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_type', 'candle_id']
otohlcvcc_df = df[new_columns_order]

# преобразование otohlcvcc_df в np.array
def convert_df_to_array(df):
    tohlcvcc_df = df.drop(df.columns[0], axis=1) # удаление столбца 'open_time'
    array = tohlcvcc_df.values
    return array

tohlcvcc_array = convert_df_to_array (otohlcvcc_df)


def find_trend_periods(otohlcvcc_df): 
    """
    Находит все восходящие и нисходящие трендовые интервалы

    Аргументы:
    otohlcvcc_df (pandas DataFrame) - датафрейм с обогащенными OHLC данными

    Возвращает:
    trend_periods_df (pandas DataFrame) - перечень трендовых интервалов
    """

    lendata = otohlcvcc_df.shape[0]  # длина (количество свечей) остаточного интервала (изначально - длина вхдящего датафрейма)
    df_vs_turn_points = otohlcvcc_df.copy()

    # Определение типа первого глобального экстремума: ('max') / ('min')
    high_id = df_vs_turn_points.high.idxmax()  # индекс первого глобального максимума (максимум на всем интервале изначального df)
    low_id = df_vs_turn_points.low.idxmin()  # индекс первого глобального минимума (минимум на всем интервале изначального df)

    extremum = min(high_id, low_id)  # первый глобальный экстремум - тот, чей индекс меньше

    entry = df_vs_turn_points.loc[extremum].copy()  # строка исходного датафрейма по найденному индексу первого глобального экстремума

    delta = lendata - extremum # длина оставшегося диапазона индексов от предыдущего глобального экстремума до конца остаточного интервала (критерий окончания поисков: delta<=2)

    trend_periods_up = []
    trend_periods_down = []

    while (lendata - extremum) > 2:

        if extremum == high_id:  # Текущий экстремум - максимум
            # Нахождение следующего минимума после текущего максимума
            next_extremum = df_vs_turn_points.low.iloc[extremum + 1:lendata].idxmin()
            entry_type = 'min'
            trend_period = (high_id, next_extremum)
            
            # Обновление low_id
            low_id = next_extremum
            
        else:  # Текущий экстремум - минимум
            # Нахождение следующего максимума после текущего минимума
            next_extremum = df_vs_turn_points.high.iloc[extremum + 1:lendata].idxmax()
            entry_type = 'max'
            trend_period = (low_id, next_extremum)
            
            # Обновление high_id
            high_id = next_extremum

        # Обновление данных для следующего цикла
        extremum = next_extremum
        delta = lendata - extremum

        # Добавление трендового периода, если длина интервала больше или равна 3
        if len(range(trend_period[0], trend_period[1])) >= 3:
            if entry_type == 'max':
                trend_periods_up.append(trend_period)
            else:
                trend_periods_down.append(trend_period)

    # Формирование результирующего DataFrame
    trend_periods = [('up', period) for period in trend_periods_up] + [('down', period) for period in trend_periods_down]
    
    trend_periods_data = []
    for trend_type, period in trend_periods:
        start_period, end_period = period
        start_time = otohlcvcc_df.loc[start_period, 'open_time'].date()
        end_time = otohlcvcc_df.loc[end_period, 'open_time'].date()
        time_range = (start_time, end_time)
        if trend_type == 'up':
            price_range = (otohlcvcc_df.loc[start_period, 'low'], otohlcvcc_df.loc[end_period, 'high'])
        else:
            price_range = (otohlcvcc_df.loc[start_period, 'high'], otohlcvcc_df.loc[end_period, 'low'])
        trend_periods_data.append([trend_type, period, time_range, price_range])

    trend_periods_df = pd.DataFrame(trend_periods_data, columns=['up/down', 'index_range', 'time_range', 'price_range'])

    return trend_periods_df


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



def find_first_candle_above_line(array, point1, point2):
    """
    Итерируя по координатам low свечей на интервале, функция находит первую свечу, low которой находится выше линии, построенной
    по двум точкам: low первой свечи (point1) и low предпоследней свечи (point2-1) трендового интервала.
    
    Проверяет координаты low свечей в обратном порядке в диапазоне [point2-2, point1) (не включая point1). 
    Если находит свечу, удовлетворяющую следующим условиям:
    1) low этой свечи находится выше построенной линии.
    2) Линия не пересекает последнюю свечу трендового интервала (low последней свечи (point2) выше линии),
    то функция возвращает индекс свечи, следующей за найденной, как потенциальную вторую точку трендовой линии.

    Аргументы:
    array -- массив данных Numpy размером (i, j), содержащий даннные по свечам 
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс первой свечи трендового интервала
    point2 -- индекс последней свечи трендового интервала
    
    Возвращает:
    point2_possible (int) -- индекс свечи, следующей за найденной, как потенциальная вторая точка трендовой линии
    """

    # Координаты первой точки (low point1)
    x1 = array[point1][0] # timestamp первой свечи
    y1 = array[point1][3] # low первой свечи
    
    # Координаты второй точки (low point2-1)
    x2 = array[point2-1][0] # timestamp предпоследней свечи
    y2 = array[point2-1][3] # low предпоследней свечи
    
    # Инициализация флага и начального предполагаемого значения второй точки
    point_found = False # Флаг, подтверждающий нахождение точки над исследуемой линией
    point2_possible = point2 - 1 # Начальное предполагаемое значение второй точки

    # Координаты верхней границы трендового интервала (low point2)
    x_int_border = array[point2][0] # timestamp последней свечи интервала
    y_int_border = array[point2][3] # low последней свечи интервала

    # Проверка свечей в обратном порядке
    for i in range(point2 - 2, point1, -1):
        x = array[i][0] # timestamp проверяемой свечи
        y = array[i][3] # low проверяемой свечи

        # Определение положения low проверяемой свечи относительно линии
        result = point_location(x1, x2, y1, y2, x, y)
        
        if result > 0: # Если low свечи находится выше линии
            # Проверка, не пересекает ли линия последнюю свечу интервала
            result_border_check = point_location(x1, x2, y1, y2, x_int_border, y_int_border)
            
            if result_border_check > 0: # Если верхняя граница трендового интервала выше линии
                point_found = True
                point2_possible = i + 1 # Установка индекса следующей свечи
                break
        else: # Если свеча пересекает линию
            x2 = array[i][0] # Обновление второй точки линии
            y2 = array[i][3]

    return point2_possible

def is_uptrendline_true(array, point1, point2_possible):
    """
    Проверяет, истинна ли потенциальная трендовая линия, построенная по точкам point1 и point2_possible.
    
    Потенциальная трендовая линия строится от Low первой свечи (point1) до Low второй свечи (point2_possible).
    Функция проверяет, что все свечи внутри интервала между point1 и point2_possible не пересекают эту линию.

    Аргументы:
    array -- numpy массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс начала трендового интервала, который представляет первую точку трендовой линии.
    point2_possible -- индекс второй точки трендовой линии, потенциально являющейся частью трендовой линии.

    Возвращаемое значение:
    True, если все свечи на интервале (point1, point2_possible) - не включая сами точки - находятся выше потенциальной трендовой линии.
    False, если хотя бы одна свеча внутри интервала пересекает или касается трендовой линии.

    Функция использует вспомогательную функцию point_location для проверки положения точки относительно линии.
    """

    x1 = array[point1][0]  # координата X первой точки трендовой линии (timestamp первой свечи)
    x2 = array[point2_possible][0]  # координата X второй точки трендовой линии (timestamp второй свечи)
    y1 = array[point1][3]  # координата Y первой точки трендовой линии (Low первой свечи)
    y2 = array[point2_possible][3]  # координата Y второй точки трендовой линии (Low второй свечи)

    trendline_check = True

    # Проверяем все свечи на интервале (point2_possible-1, point1)
    for i in range(point2_possible - 1, point1, -1):
        x_test = array[i][0]  # координата X проверяемой свечи
        y_test = array[i][3]  # координата Y проверяемой свечи
        result = point_location(x1, x2, y1, y2, x_test, y_test)

        if result <= 0:  # Если свеча пересекает или касается потенциальной трендовой линии
            trendline_check = False
            break  # Прерываем цикл проверки

    return trendline_check


def find_uptrend_line(array, point1, point2):
    """
    Находит подтвержденную трендовую линию на интервале (point1, point2).

    Функция ищет вторую точку трендовой линии, которая начинается с точки point1, и проверяет ее на истинность.
    Если находит действительную трендовую линию, возвращает список индексов обеих точек трендовой линии.

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс первой точки трендовой линии.
    point2 -- индекс последней точки интервала, где ищем трендовую линию.

    Возвращаемое значение:
    Список индексов [point1, point2_fact], где point2_fact - фактическая вторая точка трендовой линии.
    Если действительная трендовая не найдена, функция может вернуть пустой список.
    """

    for i in range(point2, point1, -1):
        point2_possible = find_first_candle_above_line(array, point1, i)  # Ищем потенциальную вторую точку трендовой
        trend_line_check = is_uptrendline_true(array, point1, point2_possible)  # Проверяем потенциальную трендовую

        if trend_line_check:  # Если трендовая линия подтверждена
            point2_fact = point2_possible  # Устанавливаем фактическую (подтвержденную) вторую точку трендовой линии
            trend_line = [point1, point2_fact]  # Формируем список индексов первой и второй свечи трендовой линии
            break

    return trend_line


def find_uptrend_lines_on_interval(array, point1, point2):
    """
    Находит все подтвержденные трендовые линии на интервале (point1, point2).

    Функция ищет все возможные трендовые линии, начинающиеся с точки point1, и заканчивающиеся до точки point2.
    Возвращает список всех найденных трендовых линий.

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс начала интервала поиска.
    point2 -- индекс конца интервала поиска.

    Возвращает:
    Список глубиной вложенности = 2, где каждый вложенный список содержит индексы двух точек трендовой линии.
    """

    uptrend_lines_list = []  # Список для хранения найденных трендовых линий
    line = []
    delta = point2 - point1  # Расстояние между начальной и конечной точкой интервала поиска

    while delta >= 3:  # Продолжаем искать трендовые линии, пока интервал достаточно велик
        line = find_uptrend_line(array, point1, point2)  # Находим трендовую линию на интервале

        # Обновляем точку начала поиска
        point1 = line[1]  # Новая начальная точка интервала поиска
        delta = point2 - point1  # Обновляем длину интервала поиска

        x_current = array[-1][0]  # Время последней свечи (текущая цена)
        point1_check = line[0]  # Индекс первой точки трендовой линии
        point2_check = line[1]  # Индекс второй точки трендовой линии

        x1 = array[point1_check][0]  # Время первой точки трендовой линии
        x2 = array[point2_check][0]  # Время второй точки трендовой линии
        y1 = array[point1_check][3]  # Low первой точки трендовой линии
        y2 = array[point2_check][3]  # Low второй точки трендовой линии

        k = (y2 - y1) / (x2 - x1)  # Коэффициент наклона трендовой линии
        b = y1 - x1 * k  # Пересечение с осью Y

        y = k * x_current + b  # Значение Y на текущем времени

        uptrend_lines_list.append(line)  # Добавляем найденную трендовую линию в список

    return uptrend_lines_list


def find_first_candle_under_line(array, point1, point2):
    """
    Находит первую свечу, находящуюся под потенциальной нисходящей трендовой линией.

    Функция строит потенциальную нисходящую трендовую линию от High свечи в точке point1 до High предпоследней свечи в интервале (point1, point2).
    Затем проверяет свечи в интервале (point2-2, point1) на соответствие двум условиям:
    1) High свечи должен быть выше линии.
    2) Линия не должна пересекаться с High последней свечи интервала (point2).

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс первой точки интервала, где находится локальный максимум (High).
    point2 -- индекс последней точки интервала, где находится локальный минимум (Low).

    Возвращаемое значение:
    point2_possible - Индекс свечи, следующей за найденной, которая считается потенциальной второй точкой трендовой линии.
    """

    # Строим линию по двум точкам
    x1 = array[point1][0]  # timestamp первой свечи
    x2 = array[point2-1][0]  # timestamp предпоследней свечи
    y1 = array[point1][2]  # High первой свечи
    y2 = array[point2-1][2]  # High предпоследней свечи

    point_found = False  # Флаг нахождения подходящей свечи
    point2_possible = point2 - 1  # Начальное предполагаемое значение второй точки

    x_int_border = array[point2][0]  # timestamp последней свечи
    y_int_border = array[point2][2]  # High последней свечи

    # Проверяем свечи в интервале (point2-2, point1) в обратном порядке
    for i in range(point2-2, point1, -1):
        x = array[i][0]  # timestamp текущей свечи
        y = array[i][2]  # High текущей свечи

        result = point_location(x1, x2, y1, y2, x, y)  # Проверка положения High текущей свечи относительно линии

        if result < 0:  # Если свеча находится под линией
            result_border_check = point_location(x1, x2, y1, y2, x_int_border, y_int_border)
            if result_border_check < 0:  # Если линия не пересекает High последней свечи
                point_found = True
                point2_possible = i + 1  # Устанавливаем потенциальную вторую точку трендовой линии
                break

        else:  # Если свеча пересекает линию
            x2 = x  # Обновляем вторую точку линии
            y2 = y

    return point2_possible


def is_downtrendline_true(array, point1, point2):
    """
    Проверяет истинность потенциальной нисходящей трендовой линии.

    Функция строит линию от High свечи в точке point1 до High свечи в точке point2 и проверяет,
    что все свечи в интервале (point1, point2) находятся ниже этой линии.

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс первой точки трендовой линии (локальный максимум).
    point2 -- индекс второй точки трендовой линии (локальный максимум).

    Возвращаемое значение:
    True, если все свечи между точками находятся ниже линии.
    False, если какая-либо свеча пересекает линию.
    """

    # Строим линию по двум точкам
    x1 = array[point1][0]  # timestamp первой точки
    x2 = array[point2][0]  # timestamp второй точки
    y1 = array[point1][2]  # High первой точки
    y2 = array[point2][2]  # High второй точки

    trendline_check = True  # Флаг проверки трендовой линии

    # Проверяем свечи в интервале (point2-1, point1)
    for i in range(point2-1, point1, -1):
        x_test = array[i][0]  # timestamp тестируемой свечи
        y_test = array[i][2]  # High тестируемой свечи
        result = point_location(x1, x2, y1, y2, x_test, y_test)

        if result >= 0:  # Если свеча пересекает линию
            trendline_check = False
            break  # Прерываем цикл проверки

    return trendline_check


def find_downtrend_line(array, point1, point2):
    """
    Находит подтвержденную нисходящую трендовую линию на интервале (point1, point2).

    Функция ищет вторую точку трендовой линии, которая начинается с точки point1, и проверяет ее на истинность.
    Если находит действительную трендовую линию, возвращает список индексов обеих точек трендовой линии.

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс первой точки трендовой линии.
    point2 -- индекс последней точки интервала, где ищем трендовую линию.

    Возвращаемое значение:
    Список индексов [point1, point2_fact], где point2_fact - фактическая вторая точка трендовой линии.
    Если действительная трендовая не найдена, функция может вернуть пустой список.
    """

    trend_line = []
    for i in range(point2, point1, -1):
        point2_possible = find_first_candle_under_line(array, point1, i)  # Находим потенциальную вторую точку трендовой
        trend_line_check = is_downtrendline_true(array, point1, point2_possible)  # Проверяем потенциальную трендовую

        if trend_line_check:  # Если трендовая линия подтверждена
            point2_fact = point2_possible
            trend_line = [point1, point2_fact]  # Формируем список трендовой линии
            break

    return trend_line


def find_downtrend_lines_on_interval(array, point1, point2):
    """
    Находит все подтвержденные нисходящие трендовые линии на интервале (point1, point2).

    Функция ищет все возможные нисходящие трендовые линии, начинающиеся с точки point1, и заканчивающиеся до точки point2.
    Возвращает список всех найденных нисходящих трендовых линий.

    Аргументы:
    array -- массив данных, где:
        0 - timestamp
        1 - open
        2 - high
        3 - low
        4 - close
        5 - volume
        6 - candle type
        7 - candle index
    point1 -- индекс начала интервала поиска.
    point2 -- индекс конца интервала поиска.

    Возвращаемое значение:
    Список списков, где каждый подсписок содержит индексы двух точек трендовой линии.
    """

    downtrend_lines_list = []  # Список для хранения найденных нисходящих трендовых линий
    line = []
    delta = point2 - point1  # Расстояние между начальной и конечной точкой интервала поиска

    while delta >= 3:  # Продолжаем искать трендовые линии, пока интервал достаточно велик
        line = find_downtrend_line(array, point1, point2)  # Находим нисходящую трендовую линию на интервале

        # Обновляем точку начала поиска
        point1 = line[1]  # Новая начальная точка интервала поиска
        delta = point2 - point1  # Обновляем длину интервала поиска

        x_current = array[-1][0]  # Время последней свечи (текущая цена)
        point1_check = line[0]  # Индекс первой точки трендовой линии
        point2_check = line[1]  # Индекс второй точки трендовой линии

        x1 = array[point1_check][0]  # Время первой точки трендовой линии
        x2 = array[point2_check][0]  # Время второй точки трендовой линии
        y1 = array[point1_check][2]  # High первой точки трендовой линии
        y2 = array[point2_check][2]  # High второй точки трендовой линии

        k = (y2 - y1) / (x2 - x1)  # Расчет коэффициента наклона трендовой линии
        b = y1 - x1 * k  # Расчет свободного члена трендовой линии

        y = k * x_current + b  # Расчет значения Y текущей цены по трендовой линии

        #if y > array[point2][3]:  # Проверка актуальности трендовой линии (закомментирована)
        downtrend_lines_list.append(line)  # Добавляем трендовую линию в список

    return downtrend_lines_list


def find_all_trendlines(df):
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
        'time_base', 'is_actual'
    ]

    result_df_down = pd.DataFrame(columns=trend_lines_df_columns)

    # Получение датафрейма трендовых периодов
    trend_periods_df = find_trend_periods(df)

    # Определение списков восходящих и нисходящих трендовых интервалов
    up_periods_list = trend_periods_df[trend_periods_df['up/down'] == 'up']['index_range'].tolist()
    down_periods_list = trend_periods_df[trend_periods_df['up/down'] == 'down']['index_range'].tolist()

    def find_all_uptrend_lines(array, uptrend_intervals_list):
        """
        Нахождение всех восходящих трендовых линий на основе предоставленных интервалов.
        """
        uptrendlines_df = pd.DataFrame(columns=['up/down', 'trend_interval', 'first_point_id', 'second_point_id'])
        for uptrend_interval in uptrend_intervals_list:
            uptrendlines_list = find_uptrend_lines_on_interval(array, uptrend_interval[0], uptrend_interval[1])
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
            downtrendlines_list = find_downtrend_lines_on_interval(array, downtrend_interval[0], downtrend_interval[1])
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
    all_uptrend_lines_df = find_all_uptrend_lines(array, up_periods_list)

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
    all_downtrend_lines_df = find_all_downtrend_lines(array, down_periods_list)

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
        'a_timestamp', 'b_timestamp'
    ]
    actual_trendlines_df = actual_trendlines_df[columns_list]

    return actual_trendlines_df