# 测试向上价差与向下价差的参数敏感性
import datetime as dt
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
import scipy as sp
import pyalgotrade
import jhtalib

# # 启动Wind并获取数据
# from WindPy import *
# w.start()
# w.start(waitTime=60)  # 命令超时时间设置成60秒
# w.isconnected()  # 即判断WindPy是否已经登陆成功
# wsd_data = w.wsd('000001.SH', 'open,high,low,OPEN,volume,amt',
#                  '2007-1-4', '2017-12-31', 'Fill=Previous')

# # 将API数据转换为DataFrame数据，并保存到本地文件
# wsd_data_ndarray = np.transpose(np.array(wsd_data.Data))
# data = pd.DataFrame(wsd_data_ndarray, index=wsd_data.Times,
#                     columns=wsd_data.Fields)
# data.to_csv('D:\Documents\GitHub\investment\DATA.csv',index=True,sep=",")

# 读取数据
import datetime
data = pd.read_csv('D:\Documents\GitHub\investment\DATA.CSV')
data.index = map(lambda x: datetime.datetime.strptime(
    x, '%Y-%m-%d'), list(data['Unnamed: 0']))
data = data.drop('Unnamed: 0', 1)

def get_sharp(time, strategy, gz_return):
    # period = time[-1] -time[0]
    period = len(np.array(time))
    return (log(strategy[-1] / strategy[0]) / (period / 365) - gz_return) / np.std(np.array(strategy))

BASE = 2728  # np.mean(data['OPEN']) + 1000
SPREAD_UP = 20
SPREAD_DOWN = 20  # np.std(data['OPEN']) / 40
amount = 1
# direction = []
# strike_price = []
sharp = []
beta = []
volatility_strategy = []
strategy_returns = []
max_drawdown_strategy = []
period = len(np.array(data.index))
benchmark = data['OPEN'] / np.array(data.loc[data.index[0], ['OPEN']])
max_drawdown_benchmark = (max(benchmark) - min(benchmark)) / max(benchmark)
volatility_benchmark = np.std(np.array(benchmark))
benchmark_returns = log(benchmark[-1] / benchmark[0]) / (period / 365)


position_initial = range(5, 300)
for initial in position_initial:
    position = [initial, ]
    buy = 0
    sell = 0
    delta = [0, ]
    base = BASE
    capital = [300000, ]
    for date in data.index:    # 净多头
        if delta[-1] > 0:
            if data.at[date, 'OPEN'] <= base - SPREAD_DOWN and data.at[date, 'OPEN'] <= capital[-1]:   # 买入条件
                buy = buy + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] + amount])
                capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
                # direction.extend([1])
                delta.extend([buy - sell])
                base = BASE - delta[-1] * SPREAD_DOWN
            elif data.at[date, 'OPEN'] >= base + SPREAD_UP and position[-1] > 0:                        # 卖出条件
                sell = sell + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] - amount])
                capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
                # direction.extend([-1])
                delta.extend([buy - sell])
                if delta[-1] == 0:
                    base = BASE
                else:
                    base = BASE - delta[-1] * SPREAD_DOWN
            else:                                                                                        # 不操作
                # strike_price.extend([0])
                position.extend([position[-1]])
                capital.extend([capital[-1]])
                # direction.extend([0])
                delta.extend([delta[-1]])
        elif delta[-1] < 0:
            if data.at[date, 'OPEN'] >= base + SPREAD_UP and position[-1] > 0:                          # 卖出条件
                sell = sell + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] - amount])
                capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
                # direction.extend([-1])
                delta.extend([buy - sell])
                base = BASE + abs(delta[-1]) * SPREAD_UP
            elif data.at[date, 'OPEN'] <= base - SPREAD_DOWN and data.at[date, 'OPEN'] <= capital[-1]:  # 买入条件
                buy = buy + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] + amount])
                capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
                # direction.extend([1])
                delta.extend([buy - sell])
                if delta[-1] == 0:
                    base = BASE
                else:
                    base = BASE + abs(delta[-1]) * SPREAD_UP
            else:                                                                                        # 无操作
                # strike_price.extend([0])
                position.extend([position[-1]])
                capital.extend([capital[-1]])
                # direction.extend([0])
                delta.extend([delta[-1]])
        else:                                                                                            # 零头寸
            if data.at[date, 'OPEN'] <= base - SPREAD_DOWN and data.at[date, 'OPEN'] <= capital[-1]:   # 买入条件
                buy = buy + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] + amount])
                capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
                # direction.extend([1])
                delta.extend([buy - sell])
                base = BASE - delta[-1] * SPREAD_DOWN
            elif data.at[date, 'OPEN'] >= base + SPREAD_UP and position[-1] > 0:                        # 卖出条件
                sell = sell + 1
                # strike_price.extend([data.at[date, 'OPEN']])
                position.extend([position[-1] - amount])
                capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
                # direction.extend([-1])
                delta.extend([buy - sell])
                base = BASE + abs(delta[-1]) * SPREAD_UP
            else:# 无操作
                # strike_price.extend([0])
                position.extend([position[-1]])
                capital.extend([capital[-1]])
                # direction.extend([0])
                delta.extend([delta[-1]])
    value = position[1:len(position)] * data['OPEN'] + capital[1:len(capital)]
    strategy = value / value[0]
    sharp.extend([get_sharp(data.index, strategy, 0.0390)])
    beta.extend([pd.Series(strategy).corr(pd.Series(benchmark))])
    strategy_returns.extend([log(strategy[-1] / strategy[0]) / (period / 365)])
    volatility_strategy.extend([np.std(np.array(strategy))])
    max_drawdown_strategy.extend([(max(strategy) - min(strategy)) / max(strategy)])

    j = 0
    for i in position:
        if i != 0:
            j = j + 1
    holding_time = j / len(position)

fig1 = plt.plot(position_initial,sharp)
plt.title('sharp')
plt.show(fig1)

fig2 = plt.plot(position_initial,beta)
plt.title('beta')
plt.show()

fig3 = plt.plot(position_initial,strategy_returns)
plt.title('returns')
plt.show()

fig3 = plt.plot(position_initial,max_drawdown_strategy)
plt.title('max drawdown')
plt.show()
