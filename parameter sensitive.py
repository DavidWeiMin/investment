import datetime as dt
from math import *
from mpl_toolkits.mplot3d import Axes3D
import jhtalib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import pyalgotrade
import scipy as sp
from WindPy import *

global data,capital,BASE,base,spread_up,spread_down,position,buy,sell,amount,direction,delta,strike_price,r

w.start()
wsd_data = w.wsd('000001.SH', 'open,high,low,close,volume,amt',
                 '2016-2-1', '2018-1-2', 'Fill=Previous')

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
        capital.extend([(capital[-1] - amount * data.at[date, 'OPEN']) * pow(1 + r,1.0 / 365)])
    else:
        position.extend([position[-1] + round(capital[-1] / data.at[date, 'OPEN'])])
        capital.extend([(capital[-1] -  round(capital[-1] / data.at[date, 'OPEN'])* data.at[date, 'OPEN']) * pow(1 + r,1.0 / 365)])
    
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
        capital.extend([(capital[-1] + amount * data.at[date, 'OPEN']) * pow(1 + r,1.0 / 365)])
    else:
        position.extend([0])
        capital.extend([(capital[-1] + position[-1] * data.at[date, 'OPEN']) * pow(1 + r,1.0 / 365)])

def null_action():
    global capital,base,position,buy,sell,direction,delta,strike_price                                                           
    position.extend([position[-1]])
    capital.extend([(capital[-1]) * pow(1 + r,1.0 / 365)])
    delta.extend([delta[-1]])

for spread in range(3,300,3):
    for k in [i / 250 for i in range(1,100)]:
        base = BASE
        spread_down = spread
        spread_up = spread
        capital = [100000, ]
        position = [round(0.5 * capital[-1] / base), ]
        amount = round(k * 0.5 * capital[-1] / base)
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
                    sell_action()                                                   # 卖出操作
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

X = np.arange(2,200,2)
Y = np.array([i / 250 for i in range(1,100)])
X, Y = np.meshgrid(X, Y)

fig1 = plt.figure()
ax1 = Axes3D(fig1)
annual_return_strategy = np.array(annual_return_strategy).reshape((99,99))
Z = np.array(annual_return_strategy)
ax1.plot_surface(X, Y, Z,cmap=cm.gist_earth)
plt.title('annual return of strategy')
plt.show()

fig2 = plt.figure()
ax2 = Axes3D(fig2)
sharp = np.array(sharp).reshape((99,99))
Z = np.array(sharp)
ax2.plot_surface(X, Y, Z, cmap=cm.gist_earth)
plt.title('sharp')
plt.show()

fig3 = plt.figure()
ax3 = Axes3D(fig3)
beta = np.array(beta).reshape((99,99))
Z = np.array(beta)
ax3.plot_surface(X, Y, Z, cmap=cm.gist_earth)
plt.title('beta')
plt.show()

fig4 = plt.figure()
ax4 = Axes3D(fig4)
alpha = np.array(alpha).reshape((99,99))
Z = np.array(alpha)
ax4.plot_surface(X, Y, Z, cmap=cm.gist_earth)
plt.title('alpha')
plt.show()

fig5 = plt.figure()
ax5 = Axes3D(fig5)
max_drawdown_strategy = np.array(max_drawdown_strategy).reshape((99,99))
Z = np.array(max_drawdown_strategy)
ax5.plot_surface(X, Y, Z, cmap=cm.gist_earth)
plt.title('max drawdown')
plt.show()

fig6 = plt.figure()
ax6 = Axes3D(fig6)
information_ratio = np.array(information_ratio).reshape((99,99))
Z = np.array(information_ratio)
ax6.plot_surface(X, Y, Z, cmap=cm.gist_earth)
plt.title('information ratio')
plt.show()