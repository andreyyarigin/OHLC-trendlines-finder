import numpy as np
import ast
import pandas as pd
import datetime
import os
from pathlib import Path


# ПРЕОБРАЗОВАНИЕ OTOHLCVCC-ДАТАФРЕЙМА В NP.ARRAY
def convert_df_to_array(otohlcvcc_df):
    tohlcvcc_df = otohlcvcc_df.drop(otohlcvcc_df.columns[0], axis=1) # удаление столбца 'open_time'
    array = tohlcvcc_df.values
    return array


# ПРЕОБРАЗОВАНИЕ NP.ARRAY в OTOHLCVCC-ДАТАФРЕЙМ
def array_to_df (array):
    result_df = pd.DataFrame(array[0:,0:8])
    result_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_type', 'candle_id']
    result_df = result_df.astype({'timestamp':'int', 'candle_type':'int', 'candle_id':'int'})
    result_df['open_time'] = pd.to_datetime(result_df['timestamp']/1000, unit ='s')
    cols = result_df.select_dtypes(exclude=['float','datetime64','int64']).columns
    result_df[cols] = result_df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    cols = result_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    result_df = result_df[cols]
    return(result_df)


# НАХОЖДЕНИЕ ТРЕНДОВЫХ ИНТЕРВАЛОВ
def trend_periods_finder(df):
    df_result = pd.DataFrame(columns=['open_time',
                                        'timestamp',
                                        'open',
                                        'high',
                                        'low',
                                        'close',
                                        'volume',
                                        'candle_type',
                                        'candle_id',
                                        'turning_point'])  # заготовка результирующего датафрейма с добавлением столбца 'turning_point' (экстремумы)

    lendata = df.shape[0]  # длина датафрейма
    extremum = int()

    df_vs_turn_points = df.copy()  # копируем получаемый функцией датафрейм с тем, чтобды работать дальше с ним

    # 01 # сначала находим какой экстремум появляется первым - максимум ('max') или минимум ('min')

    high_id = df_vs_turn_points.high.idxmax()  # индекс первого самого-максимума (максимум на всем интервале изначального df)
    low_id = df_vs_turn_points.low.idxmin()  # индекс первого самого-минимума (минимум на всем интервале изначального df)

    max_min_tuple = (high_id, low_id)  # список из индексов найденных самого-максимума и самого-минимума
    extremum = min(max_min_tuple)  # первый самый-экстремум тот - чей индекс меньше

    entry = df_vs_turn_points.loc[extremum].copy()  # получаем строку исходного датафрейма по найденному индексу первого экстремума

    if extremum == high_id:  # если первый экстремум - 'max'
        entry.at['turning_point'] = 'max'
        df_result = pd.concat([df_result, entry.to_frame().T], ignore_index=True)  # добавляем строку в результирующий dataframe
        first_extremum = 'min'  # вводим новую переменную 'first_extremum'
    else:  # если первый экстремум - 'min'
        entry.at['turning_point'] = 'min'
        df_result = pd.concat([df_result, entry.to_frame().T], ignore_index=True)  # добавляем строку в результирующий dataframe
        first_extremum = 'max'  # вводим новую переменную 'first_extremum'

    # 02 # составляем список индексов всех экстремумов

    # далее вводим новую переменную 'delta' - длина оставшегося диапазона индексов от предыдущего экстремума до конца исследуемого интервала (критерий окончания поисков: delta<=2)

    delta = lendata - extremum

    trend_periods_up = []
    trend_periods_down = []

    while delta > 2:  # до тех пор пока длина остаточного интервала будет больше 2х свечей - ищем остальные экстреммы
        # на этм этапе мы НЕ ПРОВЕРЯЕМ длину трендовых интервалов (трендовая образуется при перехае второй свечи - третьей. То есть, для появления трендовой необходим интервал, как минимум, из 3х свечей)

        if extremum == high_id:  # если текущий экстремум - 'max'
            extremum = df_vs_turn_points.low.iloc[extremum+1:lendata].idxmin()  # по лоям свечей оставшегося диапазона находим индекс следующего экстремума (в данном случае - 'min')
            entry = df_vs_turn_points.loc[extremum].copy()  # получаем строку из исходного датафрейма по найденному индексу экстремума
            entry.at['turning_point'] = 'min'
            down_period = (high_id, extremum)
            if len(range(high_id, extremum)) >= 3:  # проверяем трендовый интервал на длину - нас добавляем только те, длина который >=3
                trend_periods_down.append(down_period)
            low_id = extremum
            delta = lendata - extremum
        else:  # если текущий экстремум - 'min'
            extremum = df_vs_turn_points.high.iloc[extremum+1:lendata].idxmax()  # по лоям свечей оставшегося диапазона находим индекс следующего экстремума (в данном случае - 'max')
            entry = df_vs_turn_points.loc[extremum].copy()
            entry.at['turning_point'] = 'max'
            up_period = (low_id, extremum)
            if len(range(low_id, extremum)) >= 3:  # проверяем трендовый интервал на длину - нас добавляем только те, длина который >=3
                trend_periods_up.append(up_period)
            delta = lendata - extremum
            high_id = extremum

    # 03 Создаем итоговый список всех трендовых интервалов

    trend_periods = []

    trend_periods.append(trend_periods_up)
    trend_periods.append(trend_periods_down)

    #trend_periods_df = pd.DataFrame(columns=['up/down','index_range'])
    trend_periods_df = pd.DataFrame(columns=['up/down', 'index_range', 'time_range', 'price_range'])

    # Заполнение DataFrame данными из trend_periods
    for id, periods_list in enumerate(trend_periods):
        for period in periods_list:
            start_period = period[0]
            end_period = period[1]
            start_time = df.loc[start_period, 'open_time'].date()
            end_time = df.loc[end_period, 'open_time'].date()
            time_range = (start_time, end_time)
            price_range = (df.loc[start_period, 'low'], df.loc[end_period, 'high']) if id == 0 else (df.loc[start_period, 'high'], df.loc[end_period, 'low'])
            trend_periods_df.loc[len(trend_periods_df)] = [
                'up' if id == 0 else 'down',
                period,
                time_range,
                price_range
            ]

    return trend_periods_df


# ПРОВЕРКА ПОЛОЖЕНИЯ ТОЧКИ ОТНОСИТЕЛЬНО ПРЯМОЙ
def point_location(x1, x2, y1, y2, x, y): # определяет положение точки с координатами (x, y) относительно прямой, проведенной через точки А (x1, y1) и B (x2, y2), где x - значение времени, y - значение цены
    delta = (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)
    if delta > 0:
        return -1  # Point is below the line
    elif delta < 0:
        return 1   # Point is above the line
    else:
        return 0   # Point is on the line


# далее представлена последовательность функций, которые находят восходящие трендовые (на восходящем трендовом интервале)

# ФУНКЦИЯ 1 - first_candle_above_line (array, point1, point2)

# функция строит линию (потенциальную трендовую) от Low point1 (первой свечи тр.интервала) до Low point2-1 (предпоследней свечи восходящего тр.интервала)
# проверяет в обратной порядке по всему интервалу (point2-2 -> point1(не включительно)) Low свечей и находит первую свечу, которая соответствует обоим условиям:
# 1) Low проверяемой свечи выше построенной линии
# 2) Линия не пересекает последнюю свечу трендового интервала (Low point2 выше линии)


# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1, point2 - индексы первой и последней точек восходящего трендового интервала (индексы свечей, на которых образованы локальный минимум (point1, Low которого - MIN) и point2, High которого - MAX)

# НА ВЫХОД:
# Возвращает индекс свечи, следующей за найденной - (point2_possible) - как потенциальную вторую точку потенциальной трендовой


def first_candle_above_line (array, point1, point2):

    # строим линию по двум точкам

    x1 = array [point1][0] # координата X 1й точки - Time первой свечи трендового интервала point1 (по индексу свечи в массиве array)
    x2 = array [point2-1][0] # координата X 2й точки - Time предпоследней свечи трендового интервала point2-1 (по индексу свечи в массиве array)
    y1 = array [point1][3] # координата Y 1й точки - Low первой свечи трендового интервала point1 (по индексу свечи в массиве array)
    y2 = array [point2-1][3] # координата Y 2ой точки - Low предпоследней свечи интервала (по индексу свечи в массиве array)

    # устанавливаем флаг подтверждения и начальное предполагаемое значение второй точки

    point_found = False # флаг подтверждения нахождения точки над исследуемой линией (потенциальной трендовой)
    point2_possible = point2-1 # начальное предполагаемое значение второй точки

    # устанавливаем точку проверки (Low свечи - верхней границы восходящего трендового интервала)

    x_int_border = array [point2][0] # координата X - Time последней свечи трендового интервала point2 (по индексу свечи в массиве array)
    y_int_border = array [point2][3] # координата Y - Low последней свечи трендового интервала point2 (по индексу свечи в массиве array)


    for i in range (point2-2, point1, -1): # проверяет в обратном порядке в диапазоне индексов свечей, начиная с предпоследней свечи интервала до первой свечи интервала (не включая ее)

        x = array [i][0] # х проверяемой точки - Time проверяемой свечи
        y = array [i][3] # y проверяемой точки - Low проверяемой свечи

        result = point_location (x1, x2, y1, y2, x, y) # функция определяет - где располагается Low проверяемой свечи относительно прямой (в данном случае - лой свечи относительно трендовой)

        if result > 0: # если находим свечу, лой которой над "потенциальной трендовой"

            result_border_check = point_location (x1, x2, y1, y2, x_int_border, y_int_border)

            if result_border_check > 0: # если верхняя граница трендового интервала (point_2 с координатами (x_int_border, y_int_border)) находится выше линии (x1,x2, y1, y2)

                point_found = True
                point2_possible = i+1 # лой свечи с этим индексом лежит выше линии (x1,x2, y1, y2) - "потенциальной трендовой", а сама линия не пересекает свечу - верхнюю границу трендового интервала

                break

        else: # если свеча пересекает трендовую
            x2 = array [i][0] # то 2я точка потенциальной трендовой меняется на текущую точку проверки
            y2 = array [i][3]

    return (point2_possible)


# ФУНКЦИЯ 2 - is_uptrendline_true (array, point1, point2_possible)
# У нас есть потенциальная трендовая, построенная по точкам point1, point2_possible (найдена в результате работы функции first_candle_above_line (array, point1, point2)).
# Необходимо убедиться, что внутри интервала (point1, point2_possible) нет свеч, пересекающих линию, построенную по свечам (point1, point2_possible)
# функция строит линию (потенциальную трендовую) от Low point_1 (первой свечи тр.интервала) до Low Point_2_possible
# проверяет в обратной порядке по всему интервалу (point2_possible-1 -> point1(не включительно)) Low свечей и в случае, если все свечи выше линии - возвращает True

# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1 - индекс начала ттендового интервала, на котором мы ищем трендовую. Первая точка трендовой.
# point2_possible - Потенциальная вторая точка трендовой. Найдена функцией first_candle_above_line (array, point1, point2). Требует проверки.

# НА ВЫХОД:
# True - если все свечи между точками Low point_1 и Low point_2_possible выше этой линии. Т.Е. истинность трендовой подтверждается.
# False - если какаялибо свеча интервала (Low point_1+1 и Low point_2_possible-1) касается линии. Т.Е. истинность трендовой не подтверждается.

def is_uptrendline_true (array, point1, point2_possible): #

    x1 = array [point1][0] # координата X 1й точки - Time первой свечи трендового интервала point_1 (по индексу свечи в массиве array)
    x2 = array [point2_possible][0] # # координата X 2й точки - Time первой свечи трендового интервала point_2_possible (по индексу свечи в массиве array)
    y1 = array [point1][3] # координата Y 1й точки - Low первой свечи трендового интервала point_1 (по индексу свечи в массиве array)
    y2 = array [point2_possible][3] # координата Y 2й точки - Low первой свечи трендового интервала point_2_possible (по индексу свечи в массиве array)

    trendline_cheсk = True

    for i in range (point2_possible-1, point1, -1): # проверяем все точки до этой свечи
        x_test = array [i][0] # х точки, которую мы проверяем
        y_test = array [i][3] # y точки, которую мы проверяем
        result = point_location (x1, x2, y1, y2, x_test, y_test)

        if result <= 0 : # если свеча пересекает потенциальную трендовую

            trendline_cheсk = False
            break # прерываем цикл проверки

        else: # если свеча выше потенциальной трендовой
            continue # проверяем остальные точки
    return (trendline_cheсk)

# ФУНКЦИЯ 3 - uptrend_line (array, point1, point2) - которая ищет вторую точку трендовой на интервале (point1, point2) и возвращает трендовую в виде списка индексов первой и второй точки трендовой
# Функция в своей работе опирается на результаты работы функций first_candle_above_line (array, point1, point2) и is_uptrendline_true (array, point1, point2_possible)

# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1, point2 - индексы первой и последней точек трендового интервала

# НА ВЫХОД:
# Подтвержденная трендовая в виде списка индексов [point1, point2_fact]

def uptrend_line (array, point1, point2): # Функция, которая ищет вторую точку трендовой на интервале (point1, point2) и возвращает трендовую в виде списка индексов первой и второй точки трендовой
    for i in range (point2, point1, -1):
        point2_possible = first_candle_above_line (array, point1, i) # находим потенциальную 2ю точку трендовой
        trend_line_check = is_uptrendline_true (array, point1, point2_possible) # проверяем потенциальную трендовую

        if trend_line_check == True: # если это настоящая трендовая
            point2_fact = point2_possible # устанавливаем фактическую вторую точку
            trend_line = [point1, point2_fact] # список, определяющий трендовую через начальную и конечную точки
            break

    return (trend_line)

# функция uptrend_lines (array, point1, point2) - находит все трендовые линии на интервале индексов (point1, point2) и возвращает их в виде списка списков

def uptrend_lines (array, point1, point2):
    uptrend_lines_list = []
    line=[]
    delta = point2-point1 # delta - длина расстояния междй первой и последней точкой интервала поиска. Уменьшающийся в процессе нахождения трендовых параметр (критерий окончания поиска трендовых на интервале)

    while delta >=3 :
        line = uptrend_line (array, point1, point2) # находит трендовую линию на интервале

        # Следующий код добавляет трендовую линию в список, предварительно проверив ее на актуальность.
        # Но в нашем случает проверка на актуальность не производится (соответствующий код помечен "#" - комментарием)
        # Это сделано в связи с тем, что проверку всех трендовых на актуальность будем проводить потом, уже в результирующей функции trend_lines_df_v3

        point1 = line[1] # первая точка трендовой - индекс - перывй элемент списка (где второй элемент - вторая точка трендовой)
        delta = point2-point1

        x_current = array [-1][0] # х (время) текущей цены инструмента на данном тайм-фрейме (время последней свечи данного ТФ инструмента)
        point1_check = line[0] # индекс первой точки проверяемой трендовой
        point2_check = line[1] # индекс второй точки проверяемой трендовой

        x1 = array [point1_check][0] # х 1й точки - время свечи по индексу
        x2 = array [point2_check][0] # х 2й точки - время свечи по индексу
        y1 = array [point1_check][3] # y 1й точки - лой свечи по индексу
        y2 = array [point2_check][3] # y 2ой точки - лой свечи по индексу

        k=(y2-y1)/(x2-x1) # строим трендовую по двум точкам
        b=y1-x1*((y2-y1)/(x2-x1))

        y = k * x_current + b # подставляем значение Х текущей цены в уравнение трендовой - чтобы узнать y-значениие точки пересечения перпендикуляра

        #if y < array [point2][2]: # если пересечение перпендикуляра ниже хая текущего трендового интервала (это код проверки трендовой на актуальность - не используется здесь, впоследствии все трендовые на актуальность проверятся вместе)

        uptrend_lines_list. append(line) # добавляем трендовую в список
    return (uptrend_lines_list)


# далее представлена последовательность функций, которые находят нисходящие трендовые (на нисходящем трендовом интервале)

# ФУНКЦИЯ 1 - first_candle_under_line (array, point1, point2)

# функция строит линию (потенциальную трендовую) от High point1 (первой свечи тр.интервала) до High point2-1 (предпоследней свечи нисходящего тр.интервала)
# проверяет в обратной порядке по всему интервалу (point2-2 -> point1(не включительно)) High свечей и находит первую свечу, которая соответствует обоим условиям:
# 1) High проверяемой свечи выше построенной линии
# 2) Линия не пересекает последнюю свечу трендового интервала (High point2 ниже линии)


# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1, point2 - индексы первой и последней точек нисходящего трендового интервала (индексы свечей, на которых образованы локальный максимум (point1, High которого - MAX) и point2, Low которого - MIN)

# НА ВЫХОД:
# Возвращает индекс свечи, следующей за найденной - (point2_possible) - как потенциальную вторую точку потенциальной трендовой


def first_candle_under_line (array, point1, point2):

    # строим линию по двум точкам

    x1 = array [point1][0] # координата X 1й точки - Time первой свечи трендового интервала point1 (по индексу свечи в массиве array)
    x2 = array [point2-1][0] # координата X 2й точки - Time предпоследней свечи трендового интервала point2-1 (по индексу свечи в массиве array)
    y1 = array [point1][2] # координата Y 1й точки - High первой свечи трендового интервала point1 (по индексу свечи в массиве array)
    y2 = array [point2-1][2] # координата Y 2й точки - High предпоследней свечи трендового интервала point2-1 (по индексу свечи в массиве array)

    # устанавливаем флаг подтверждения и начальное предполагаемое значение второй точки

    point_found = False # флаг подтверждения нахождения точки под исследуемой линией (потенциальной трендовой)
    point2_possible = point2-1 # начальное предполагаемое значение второй точки

    # устанавливаем точку проверки (High свечи - верхней границы восходящего трендового интервала)

    x_int_border = array [point2][0] # координата X - Time последней свечи трендового интервала point2 (по индексу свечи в массиве array)
    y_int_border = array [point2][2] # координата Y - High последней свечи трендового интервала point2 (по индексу свечи в массиве array)

    for i in range (point2-2, point1, -1): # проверяет в обратном порядке в диапазоне индексов свечей, начиная с предпоследней свечи интервала до первой свечи интервала (не включая ее)

        x = array [i][0] # х проверяемой точки - Time проверяемой свечи
        y = array [i][2] # y проверяемой точки - High проверяемой свечи

        result = point_location (x1, x2, y1, y2, x, y) # функция определяет - где располагается Low проверяемой свечи относительно прямой (в данном случае - High свечи относительно трендовой)

        if result < 0: # если находим свечу, High которой под "потенциальной трендовой"

            result_border_check = point_location (x1, x2, y1, y2, x_int_border, y_int_border)

            if result_border_check < 0 : # если High свечи, соответствующей нижней границе трендового интервала (point2 с координатами (x_int_border, y_int_border)) находится ниже линии (x1,x2, y1, y2)

                point_found = True
                point2_possible = i+1 # High свечи с этим индексом лежит ниже линии (x1,x2, y1, y2) - "потенциальной трендовой", а сама линия не пересекает свечу - нижнюю границу трендового интервала

                break

        else: # если свеча пересекает трендовую
            x2 = array [i][0] # то 2я точка потенциальной трендовой меняется на текущую точку проверки
            y2 = array [i][2]

    return (point2_possible)

# ФУНКЦИЯ 2 - is_downtrendline_true (array, point1, point2_possible)
# У нас есть потенциальная нисходящая трендовая, построенная по точкам point1, point2_possible (найдена в результате работы функции first_candle_under_line (array, point1, point2)).
# Необходимо убедиться, что внутри интервала (point1, point2_possible) нет свеч, пересекающих линию, построенную по свечам (point1, point2_possible)
# функция строит линию (потенциальную трендовую) от High point_1 (первой свечи тр.интервала) до High Point_2_possible
# проверяет в обратной порядке по всему интервалу (point2_possible-1 -> point1(не включительно)) High свечей и в случае, если все свечи ниже линии - возвращает True

# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1 - индекс начала нисходящего трендового интервала, на котором мы ищем трендовую. Первая точка трендовой.
# point2_possible - Потенциальная вторая точка трендовой. Найдена функцией first_candle_under_line (array, point1, point2). Требует проверки.

# НА ВЫХОД:
# True - если все свечи между точками High point_1 и High point_2_possible ниже этой линии. Т.Е. истинность трендовой подтверждается.
# False - если какаялибо свеча интервала (High point_1+1 и High point_2_possible-1) касается линии. Т.Е. истинность трендовой не подтверждается.

def is_downtrendline_true (array, point1, point2): # у нас есть потенциальная трендовая, построенная по High точек point1, point2_possible. Необходимо убедиться, что внутри этого интервала нет свеч, пересекающих ее.

    x1 = array [point1][0] # х 1й точки - Time свечи по индексу
    x2 = array [point2][0] # х 2й точки - Time свечи по индексу
    y1 = array [point1][2] # y 1й точки - High свечи по индексу
    y2 = array [point2][2] # y 2ой точки - High свечи по индексу

    trendline_cheсk = True

    for i in range (point2-1, point1, -1): # проверяем все точки до этой свечи
        x_test = array [i][0] # х точки, которую мы проверяем
        y_test = array [i][2] # y точки, которую мы проверяем
        result = point_location (x1, x2, y1, y2, x_test, y_test)

        if result >= 0 : # если свеча пересекает потенциальную трендовую

            trendline_cheсk = False
            break # прерываем цикл проверки

        else: # если свеча выше потенциальной трендовой
            continue # проверяем остальные точки
    return (trendline_cheсk)

# ФУНКЦИЯ 3 - downtrend_line (array, point1, point2) - которая ищет вторую точку трендовой на интервале (point1, point2) и возвращает трендовую в виде списка индексов первой и второй точки трендовой
# Функция в своей работе опирается на результаты работы функций first_candle_under_line (array, point1, point2) и is_downtrendline_true (array, point1, point2_possible)

# НА ВХОД:
# array - массив данных, где (0-time, 1- open, 2 - high, 3 - low, 4 - close, 5 - volume, 6 - candle type, 7 - candle index)
# point1, point2 - индексы первой и последней точек трендового интервала

# НА ВЫХОД:
# Подтвержденная трендовая в виде списка индексов [point1, point2_fact]

def downtrend_line (array, point1, point2): # Функция, которая ищет вторую точку трендовой на интервале и возвращает трендовую в виде списка индексов первой и второй точки трендовой
    trend_line=[]
    for i in range (point2, point1, -1):
        point2_possible = first_candle_under_line (array, point1, i) # находим потенциальную 2ю точку трендовой
        trend_line_check = is_downtrendline_true (array, point1, point2_possible) # проверяем потенциальную трендовую

        if trend_line_check == True:
            point2_fact = point2_possible
            trend_line.append(point1)
            trend_line.append(point2_fact)
            break
    return (trend_line)

# функция downtrend_lines (array, point1, point2) - находит все трендовые линии на интервале индексов (point1, point2) и возвращает их в виде списка списков

def downtrend_lines (array, point1, point2):
    downtrend_lines_list = []
    line=[]
    delta = point2-point1

    while delta >= 3 :
        line = downtrend_line (array, point1, point2) # находит трендовую линию

        # Следующий код добавляет трендовую линию в список, предварительно проверив ее на актуальность.
        # Но в нашем случает проверка на актуальность не производится (соответствующий код помечен "#" - комментарием)
        # Это сделано в связи с тем, что проверку всех трендовых на актуальность будем проводить потом, уже в результирующей функции trend_lines_df_v3

        point1 = line[1]
        delta = point2-point1

        x_current = array [-1][0] # х текущей цены
        point1_check = line[0]
        point2_check = line[1]
        delta = point2-point1

        x1 = array [point1_check][0] # х 1й точки - Time свечи по индексу
        x2 = array [point2_check][0] # х 2й точки - Time свечи по индексу
        y1 = array [point1_check][2] # y 1й точки - High свечи по индексу
        y2 = array [point2_check][2] # y 2ой точки - High свечи по индексу

        k=(y2-y1)/(x2-x1) # строим трендовую по двум точкам
        b=y1-x1*((y2-y1)/(x2-x1))

        y = k * x_current + b # подставляем значение Х текущей цены в уравнение трендовой - чтобы узнать Y-значениие точки пересечения перпендикуляра

        #if y > array [point2][3]: (это код проверки трендовой на актуальность - не используется здесь, впоследствии все трендовые на актуальность проверятся вместе)
        downtrend_lines_list. append(line)
    return (downtrend_lines_list)

# Функция def trend_lines_df_v7 (array, df, timeframe) составляет датафрейм из всех трендовых линий, действительных на момент закрытия последней свечи входного датафрейма.

# trend_lines_df_v7 - четвертая версия функции, которая идентична второй, но в отличие от нее проверяет трендовые на актуальность уже после формирования их полного списка

# 'Up/Down' /str/ - восходящая или нисходящая (up / down)
# 'A_time' /time/ - время точки A
# 'B_time' /time/ - время точки B
# 'Trnd_base' /int/ (Trend_Base) - "база" - на сколько свечей "опирается" трендовая (количество свечей между точками A и B)
# 'Tg_Alpha' /float/ - тангенс угла наклона трендовой. Вычисляется как частное: (% изменения цены)/(количество свечей между точками A и B трендовой)
# 'A_id' /int/ - индекс точки A
# 'B_id' /int/ - индекс точки B
# 'A_price' /float/ - цена точки A
# 'B_price' /float/ - цена точки B
# 'Encl.' /int/ (Enclosured)- является ли трендовая вложенной на этом интервале (0 - не является, 1 - является 1й вложенной, 2 - 2й и т.д.)
# 'Brk_price' /float/ Breaking price level - цена, при достижении которой ломается трендовая (цена точки A)
# 'Trnd_int' /list/ (Trend_interval) - трендовый интервал
# 'Irr_price' /float/ - (Irrelevant price level) цена хая/лоя трендового интервала. На основании этой цены рассчитывается уровень цены, при достижении которой трендовая становится неактуальной.
# Восходящая трендовая становится неактуальной, если точка пересечения перпендикуляра текущей цены и трендовой - выше хая трендового интервала данной трендовой.
# Нисходящая трендовая становится неактуальной, если точка пересечения перпендикуляра текущей цены и трендовой - ниже лоя трендового интервала данной трендовой.

def trend_lines_df_v7(array, df):

    x_current = df['timestamp'].iloc[-1] # значение по оси Х (timestamp) текущей (последней) свечи ohlc-среза. Актуальность трендовой линии определяется относительно перпендикуляра к оси X из этой точки

    trend_lines_df_columns = ['up/down',
                              'a_timestamp',
                              'b_timestamp',
                              'first_point_time',
                              'second_point_time',
                              'candles_base',
                              'tg_alpha',
                              'first_point_id',
                              'second_point_id',
                              'first_point_price',
                              'second_point_price',
                              'enclosured',
                              'break_price',
                              'irrelevant_price',
                              'trend_interval',
                              'time_base',
                              'is_actual']

    result_df_down = pd.DataFrame(columns=trend_lines_df_columns)

    trend_periods_df = trend_periods_finder(df) # получаем датайрейм трендовых периодов из ohlc_df

    up_periods_list = trend_periods_df[trend_periods_df['up/down'] == 'up']['index_range'].tolist() # датафрейм восходящих трендовых интервалов
    down_periods_list = trend_periods_df[trend_periods_df['up/down'] == 'down']['index_range'].tolist() # датафрейм нисходящих трендовых интервалов

    def all_uptrend_lines (array, uptrend_intervals_list):
        uptrendlines_df = pd.DataFrame(columns = ['up/down', 'trend_interval', 'first_point_id', 'second_point_id'])
        for uptrend_interval in uptrend_intervals_list:
            uptrendlines_list = uptrend_lines (array, uptrend_interval[0], uptrend_interval[1])
            if len(uptrendlines_list)>0:
                for uptrendline in uptrendlines_list:
                    uptrend_interval_str = str(uptrend_interval)
                    uptrendline_df = pd.DataFrame ({'up/down':'up', 'trend_interval': uptrend_interval_str, 'first_point_id': uptrendline[0], 'second_point_id':uptrendline[1]}, index=[0])
                    uptrendlines_df= pd.concat([uptrendlines_df, uptrendline_df ], ignore_index=True)
        return uptrendlines_df

    def is_uptrendline_actual (array, x1, x2, y1, y2, trend_int):
        trend_int_high_id = ast.literal_eval(trend_int)[1] # индекс свечи, на которой образован хай трендового интервала
        trend_int_high = array[trend_int_high_id][2] # хай трендового интервала
        y_intersection = (x_current*(y2-y1)/(x2-x1) + y1 - x1*((y2-y1)/(x2-x1)))
        if y_intersection < trend_int_high: # если пересечение ниже хая интервала:
            return 1 # трендовая актуальна
        else:
            return 0 # трендовая неактуальна

    def all_downtrend_lines (array, downtrend_intervals_list):
        downtrendlines_df = pd.DataFrame(columns = ['up/down', 'trend_interval', 'first_point_id', 'second_point_id'])
        for downtrend_interval in downtrend_intervals_list:
            downtrendlines_list = downtrend_lines (array, downtrend_interval[0], downtrend_interval[1])
            if len(downtrendlines_list)>0:
                for downtrendline in downtrendlines_list:
                    downtrend_interval_str = str(downtrend_interval)
                    downtrendline_df = pd.DataFrame ({'up/down':'down', 'trend_interval': downtrend_interval_str, 'first_point_id': downtrendline[0], 'second_point_id': downtrendline[1]}, index=[0])
                    downtrendlines_df= pd.concat([downtrendlines_df, downtrendline_df ], ignore_index=True)
        return downtrendlines_df

    def is_downtrendline_actual (array, x1, x2, y1, y2, trend_int):
        trend_int_low_id = ast.literal_eval(trend_int)[1] # индекс свечи, на которой образован low трендового интервала
        trend_int_low = array[trend_int_low_id][3] # low трендового интервала
        y_intersection = (x_current*(y2-y1)/(x2-x1) + y1 - x1*((y2-y1)/(x2-x1)))
        if y_intersection > trend_int_low: # если пересечение ниже хая интервала:
            return 1 # трендовая актуальна
        else:
            return 0 # трендовая неактуальна

    all_uptrend_lines_df = all_uptrend_lines(array, up_periods_list)

    if not all_uptrend_lines_df.empty:

        all_uptrend_lines_df['first_point_price'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['second_point_price'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['first_point_time'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_uptrend_lines_df['second_point_time'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id,'open_time'])
        all_uptrend_lines_df['a_timestamp'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_uptrend_lines_df['b_timestamp'] = all_uptrend_lines_df['second_point_id'].apply(lambda id: df.at[id,'timestamp'])
        all_uptrend_lines_df['break_price'] = all_uptrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'low'])
        all_uptrend_lines_df['candles_base'] = all_uptrend_lines_df.apply(lambda x: x['second_point_id']-x['first_point_id'], axis =1)
        all_uptrend_lines_df['enclosured'] = all_uptrend_lines_df.apply(lambda x: 0 if x['first_point_id'] == ast.literal_eval(x['trend_interval'])[0] else 1 , axis=1 )
        all_uptrend_lines_df['irrelevant_price'] = all_uptrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'low'])
        all_uptrend_lines_df['time_base'] = all_uptrend_lines_df.apply(lambda x: x['second_point_time'] - x['first_point_time'], axis=1 )
        all_uptrend_lines_df['tg_alpha'] = all_uptrend_lines_df.apply(lambda x: abs(x['second_point_price'] - x['first_point_price']/x['second_point_price'])/x['candles_base'],  axis=1)
        all_uptrend_lines_df['is_actual'] = all_uptrend_lines_df.apply(lambda x: is_uptrendline_actual (array, x['a_timestamp'], x['b_timestamp'], x['first_point_price'], x['second_point_price'],  x['trend_interval']),  axis=1)

        all_uptrend_lines_df.drop(all_uptrend_lines_df[all_uptrend_lines_df['is_actual'] == 0].index, inplace=True)
        all_uptrend_lines_df.drop('is_actual', axis=1, inplace=True)

    all_downtrend_lines_df = all_downtrend_lines(array, down_periods_list)

    if not all_downtrend_lines_df.empty:

        all_downtrend_lines_df['first_point_price'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['second_point_price'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['first_point_time'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'open_time'])
        all_downtrend_lines_df['second_point_time'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id,'open_time'])
        all_downtrend_lines_df['a_timestamp'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'timestamp'])
        all_downtrend_lines_df['b_timestamp'] = all_downtrend_lines_df['second_point_id'].apply(lambda id: df.at[id,'timestamp'])
        all_downtrend_lines_df['break_price'] = all_downtrend_lines_df['first_point_id'].apply(lambda id: df.at[id, 'high'])
        all_downtrend_lines_df['candles_base'] = all_downtrend_lines_df.apply(lambda x: x['second_point_id']-x['first_point_id'], axis =1)
        all_downtrend_lines_df['enclosured'] = all_downtrend_lines_df.apply(lambda x: 0 if x['first_point_id'] == ast.literal_eval(x['trend_interval'])[0] else 1 , axis=1 )
        all_downtrend_lines_df['irrelevant_price'] = all_downtrend_lines_df['trend_interval'].apply(lambda x: df.at[ast.literal_eval(x)[1], 'high'])
        all_downtrend_lines_df['time_base'] = all_downtrend_lines_df.apply(lambda x: x['second_point_time'] - x['first_point_time'], axis=1 )
        all_downtrend_lines_df['tg_alpha'] = all_downtrend_lines_df.apply(lambda x: abs(x['second_point_price'] - x['first_point_price']/x['second_point_price'])/x['candles_base'],  axis=1)
        all_downtrend_lines_df['is_actual'] = all_downtrend_lines_df.apply(lambda x: is_downtrendline_actual (array, x['a_timestamp'], x['b_timestamp'], x['first_point_price'], x['second_point_price'],  x['trend_interval']),  axis=1)

        all_downtrend_lines_df.drop(all_downtrend_lines_df[all_downtrend_lines_df['is_actual'] == 0].index, inplace=True)
        all_downtrend_lines_df.drop('is_actual', axis=1, inplace=True)

    actual_trendlines_df = pd.concat([all_uptrend_lines_df, all_downtrend_lines_df], ignore_index=True)
    columns_list = ['up/down', 'first_point_time',
'second_point_time', 'trend_interval', 'time_base', 'first_point_price',
 'second_point_price',
'first_point_id',
 'second_point_id',
 'break_price',
 'candles_base',
 'enclosured',
 'irrelevant_price',
 'tg_alpha', 'a_timestamp',
 'b_timestamp']
    actual_trendlines_df = actual_trendlines_df [columns_list]

    return actual_trendlines_df

script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sample_file_path = script_path + '/Trendlines-finder-for-OHLC/XRPUSDT_OHLC_1d.csv'

tohlcvcc_columns = ('timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_type', 'candle_id')

btcusdt_tohlcvcc_df = pd.read_csv (sample_file_path, header = None, names = tohlcvcc_columns)
btcusdt_tohlcvcc_df['open_time'] = pd.to_datetime(btcusdt_tohlcvcc_df['timestamp'], unit='ms')

otohlcvcc_columns = ['open_time', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'candle_type', 'candle_id']
btcusdt_otohlcvcc_df = btcusdt_tohlcvcc_df[otohlcvcc_columns]

btcusdt_tohlcvcc_array = convert_df_to_array (btcusdt_otohlcvcc_df)
wewe_4h = trend_lines_df_v7(btcusdt_tohlcvcc_array, btcusdt_otohlcvcc_df)
print (wewe_4h)