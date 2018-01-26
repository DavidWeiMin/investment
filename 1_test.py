import datetime as dt
from math import *
from mpl_toolkits.mplot3d import Axes3D
import jhtalib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyalgotrade
import scipy as sp

from WindPy import *

global data,capital,BASE,base,spread_up,spread_down,position,buy,sell,amount,direction,delta,strike_price,r

w.start()
wsd_data = w.wsd('000001.SH', 'open,high,low,close,volume,amt',
                 '1997-7-21', '1998-3-24', 'Fill=Previous')

# 将API数据转换为DataFrame数据，并保存到本地文件
wsd_data_ndarray = np.transpose(np.array(wsd_data.Data))
data = pd.DataFrame(wsd_data_ndarray, index=wsd_data.Times,
                    columns=wsd_data.Fields)
data.to_csv('D:\Documents\GitHub\investment\DATA.csv',index=True,sep=",")

data = pd.read_csv('D:\Documents\GitHub\investment\DATA.CSV')
data.index = map(lambda x :dt.datetime.strptime(x,'%Y-%m-%d'),list(data['Unnamed: 0']))
data = data.drop('Unnamed: 0',1)

BASE = data['OPEN'][0]
r = 0.03

period = len(np.array(data.index))
benchmark = data['OPEN'] / np.array(data.loc[data.index[0], ['OPEN']])
Open = data['OPEN']
total_return_benchmark = benchmark[-1] / benchmark[0] - 1
total_return_strategy = []
annual_return_benchmark = ((1 + total_return_benchmark) ** (252.0 / period) - 1)
annual_return_strategy = []
beta = []
alpha = []
daily_return_benchmark = np.array(Open[1:period]) / np.array(Open[0:-1]) - 1
daily_volitility_benchmark = np.std(daily_return_benchmark)
daily_volitility_strategy = []
annual_volitility_benchmark = daily_volitility_benchmark * sqrt(252)
annual_volitility_strategy = []
sharp = []
information_ratio = []
max_drawdown_benchmark = (min(benchmark) - max(benchmark)) / max(benchmark)
max_drawdown_strategy = []
holding_time = []
turnover = []

def buy_action():
    global capital,base,position,buy,sell,direction,delta,strike_price
    buy = buy + 1
    delta.extend([buy - sell])
    base = BASE - delta[-1] * spread_down
    if capital[-1] >= data.at[date, 'OPEN'] * amount:
        position.extend([position[-1] + amount])
        capital.extend([(capital[-1] - amount * data.at[date, 'OPEN']) * pow(1 + r,1 / 365)])
    else:
        position.extend([position[-1] + capital[-1] // data.at[date, 'OPEN']])
        capital.extend([(capital[-1] -  capital[-1] // data.at[date, 'OPEN'] * data.at[date, 'OPEN']) * pow(1 + r,1 / 365)])
    
def sell_action():
    global capital,base,position,buy,sell,direction,delta,strike_price
    sell = sell + 1
    delta.extend([buy - sell])
    if delta[-1] == 0:
        base = BASE
    else:
        base = BASE - delta[-1] * spread_down
    if position[-1] > amount:
        position.extend([position[-1] - amount])
        capital.extend([(capital[-1] + amount * data.at[date, 'OPEN']) * pow(1 + r,1 / 365)])
    else:
        position.extend([0])
        capital.extend([(capital[-1] + position[-1] * data.at[date, 'OPEN']) * pow(1 + r,1 / 365)])

def null_action():
    global capital,base,position,buy,sell,direction,delta,strike_price                                                           
    position.extend([position[-1]])
    capital.extend([(capital[-1]) * pow(1 + r,1 / 365)])
    delta.extend([delta[-1]])

for spread in range(5,300,2):
    base = BASE
    spread_down = spread
    spread_up = spread
    capital = [100000, ]
    position = [0.5 * capital[-1] // base, ]
    amount = 0.1 * 0.5 * capital[-1] // base
    buy = 0
    sell = 0
    delta = [0, ]
    for date in data.index:
        if delta[-1] > 0:                                                       # 净多头
            if data.at[date, 'OPEN'] <= base - spread_down :                    # 买入条件
                buy_action()                                                    # 买入操作
            elif data.at[date, 'OPEN'] >= base + spread_up:                     # 卖出条件
                sell_action()                                                   # 卖出操作
            else:                                                               
                null_action()                                                   # 不操作
        elif delta[-1] < 0:                                                     # 净空头
            if data.at[date, 'OPEN'] >= base + spread_up:                       # 卖出条件
                sell_action()                                                    # 卖出操作
            elif data.at[date, 'OPEN'] <= base - spread_down:                   # 买入条件
                buy_action()                                                    # 买入操作
            else:                                               
                null_action()                                                   # 无操作
        else:                                                                   # 零头寸
            if data.at[date, 'OPEN'] <= base - spread_down:                     # 买入条件
                buy_action()                                                    # 买入操作
            elif data.at[date, 'OPEN'] >= base + spread_up:                     # 卖出条件
                sell_action()                                                   # 卖出操作
            else:
                null_action()                                                   # 无操作
    value = position[1:len(position)] * data['OPEN'] + capital[1:len(capital)]
    strategy = value / value[0]

    total_return_strategy.extend([strategy[-1] / strategy[0] - 1])

    annual_return_strategy.extend([((1 + total_return_strategy[-1]) ** (252.0 / period) - 1)])

    daily_return_strategy = np.array(value[1:period]) / np.array(value[0:-1]) - 1
    beta.extend([pd.Series(daily_return_strategy).corr(pd.Series(daily_return_benchmark))])

    alpha.extend([annual_return_strategy[-1] - (r + beta[-1] * (annual_return_benchmark - r))])

    daily_volitility_strategy.extend([np.std(daily_return_strategy)])
    annual_volitility_strategy.extend([daily_volitility_strategy[-1] * sqrt(252)])
    sharp.extend([(annual_return_strategy[-1] - r) / annual_volitility_strategy[-1]])

    information_ratio.extend([(annual_return_strategy[-1] - annual_return_benchmark) / (np.std(daily_return_strategy - daily_return_benchmark) * sqrt(252))])

    max_drawdown_strategy.extend([(min(strategy) - max(strategy)) / max(strategy)])

    turnover.extend([buy + sell])

    j = 0
    for i in position:
        if i != 0:
            j = j + 1
    holding_time.extend([j / len(position)])

fig1 = plt.plot(range(5,300,2),total_return_strategy)
plt.title('total returns')
plt.show(fig1)

fig2 = plt.plot(range(5,300,2),annual_return_strategy)
plt.title('annual return')
plt.show(fig2)

fig3 = plt.plot(range(5,300,2),daily_volitility_strategy)
plt.title('daily volitility')
plt.show(fig3)

fig4 = plt.plot(range(5,300,2),annual_volitility_strategy)
plt.title('annual volitility')
plt.show(fig4)

fig5 = plt.plot(range(5,300,2),beta)
plt.title('beta')
plt.show(fig5)

fig6 = plt.plot(range(5,300,2),alpha)
plt.title('alpha')
plt.show(fig6)

fig7 = plt.plot(range(5,300,2),sharp)
plt.title('sharp')
plt.show(fig7)

fig8 = plt.plot(range(5,300,2),information_ratio)
plt.title('information_ratio')
plt.show(fig8)

fig9 = plt.plot(range(5,300,2),max_drawdown_strategy)
plt.title('max drawdown')
plt.show(fig9)
