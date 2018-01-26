# 测试初始仓位的参数敏感性
import datetime as dt
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
import scipy as sp
import pyalgotrade
import jhtalib

# 启动Wind并获取数据
from WindPy import *
w.start()
w.start(waitTime=60)  # 命令超时时间设置成60秒
w.isconnected()  # 即判断WindPy是否已经登陆成功
wsd_data = w.wsd('000001.SH', 'open,high,low,close,volume,amt',
                 '2012-12-11', '2013-12-31', 'Fill=Previous')

# 将API数据转换为DataFrame数据，并保存到本地文件
wsd_data_ndarray = np.transpose(np.array(wsd_data.Data))
data = pd.DataFrame(wsd_data_ndarray, index=wsd_data.Times,
                    columns=wsd_data.Fields)
data.to_csv('D:\Documents\GitHub\investment\DATA.csv',index=True,sep=",")

# 读取数据
import datetime
data = pd.read_csv('D:\Documents\GitHub\investment\DATA.CSV')
data.index = map(lambda x: dt.datetime.strptime(
    x, '%Y-%m-%d'), list(data['Unnamed: 0']))
data = data.drop('Unnamed: 0', 1)

# 设置静态参数
BASE = 2077  # np.mean(data['OPEN']) + 1000
SPREAD_UP = 20
SPREAD_DOWN = 20  # np.std(data['OPEN']) / 40
amount = 1

period = len(np.array(data.index))
benchmark = data['OPEN'] / np.array(data.loc[data.index[0], ['OPEN']])
Open = data['OPEN']
total_return_benchmark = benchmark[-1] / benchmark[0] - 1
total_return_strategy = []
annual_return_benchmark = ((1 + total_return_benchmark) ** (252 / period) - 1)
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
max_drawdown_benchmark = (max(benchmark) - min(benchmark)) / max(benchmark)
max_drawdown_strategy = []
holding_time = []

position_initial = range(1, 100)
for initial in position_initial:
    position = [initial, ]
    buy = 0
    sell = 0
    delta = [0, ]
    base = BASE
    capital = [1000000, ]
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
    # 策略表现
    value = position[1:len(position)] * data['OPEN'] + capital[1:len(capital)]
    strategy = value / value[0]

    total_return_strategy.extend([strategy[-1] / strategy[0] - 1])

    annual_return_strategy.extend([((1 + total_return_strategy[-1]) ** (252 / period) - 1)])

    daily_return_strategy = np.array(value[1:period]) / np.array(value[0:-1]) - 1
    beta.extend([pd.Series(daily_return_strategy).corr(pd.Series(daily_return_benchmark))])

    alpha.extend([annual_return_strategy[-1] - (0.03 + beta[-1] * (annual_return_benchmark - 0.03))])

    daily_volitility_strategy.extend([np.std(daily_return_strategy)])
    annual_volitility_strategy.extend([daily_volitility_strategy[-1] * sqrt(252)])
    sharp.extend([(annual_return_strategy[-1] - 0.03) / annual_volitility_strategy[-1]])

    information_ratio.extend([(annual_return_strategy[-1] - annual_return_benchmark) / (np.std(daily_return_strategy - daily_return_benchmark) * sqrt(252))])

    max_drawdown_strategy.extend([(max(strategy) - min(strategy)) / max(strategy)])

    j = 0
    for i in position:
        if i != 0:
            j = j + 1
    holding_time.extend([j / len(position)])

fig1 = plt.plot(position_initial,total_return_strategy)
plt.title('total returns')
plt.show(fig1)

fig2 = plt.plot(position_initial,annual_return_strategy)
plt.title('annual return')
plt.show(fig2)

fig3 = plt.plot(position_initial,daily_volitility_strategy)
plt.title('daily volitility')
plt.show(fig3)

fig4 = plt.plot(position_initial,annual_volitility_strategy)
plt.title('annual volitility')
plt.show(fig4)

fig5 = plt.plot(position_initial,beta)
plt.title('beta')
plt.show(fig5)

fig6 = plt.plot(position_initial,alpha)
plt.title('alpha')
plt.show(fig6)

fig7 = plt.plot(position_initial,sharp)
plt.title('sharp')
plt.show(fig7)

fig8 = plt.plot(position_initial,information_ratio)
plt.title('information_ratio')
plt.show(fig8)

fig9 = plt.plot(position_initial,max_drawdown_strategy)
plt.title('max drawdown')
plt.show(fig9)

# print('Benchmark Returns        \t:\t %.4f%%' % (benchmark_returns * 100))
# print('Strategy Returns         \t:\t %.4f%%' % (strategy_returns * 100))
# print('Sharp                    \t:\t %.2f' % (sharp))
# print('Beta                     \t:\t %.4f' % (beta))
# print('Volatility of strategy   \t:\t %.2f%%' % (volatility_strategy * 100))
# print('Volatility of benchmark  \t:\t %.2f%%' % (volatility_benchmark * 100))
# print('Max Drawdown of strategy \t:\t %.2f%%' % (max_drawdown_strategy * 100))
# print('Max Drawdown of benchmark\t:\t %.2f%%' % (max_drawdown_benchmark * 100))
# print('Buy                      \t:\t %d' % (buy))
# print('Sell                     \t:\t %d' % (sell))
# print('Holding Time             \t:\t %.2f%%' % (holding_time * 100))


# # 保存交易日志
# print(delta)
# trader = [direction,strike_price,position[1:len(position)],capital[1:len(capital)],delta[1:len(delta)]]
# print(trader)
# # np.savetxt('D:\Documents\GitHub\investment\TRADER.csv', trader, delimiter = ',')
# header = ['Direction','Strike Price','Position','Capital','Delta']
# Trader = pd.DataFrame(data=trader,index=data.index,columns=header)
# Trader.to_csv('D:\Documents\GitHub\investment\TRADER.csv',index=True,sep=",")
# print(data.index)


# 问题
# 1. 符合买入条件，但是钱不够买入预期股数，是不买还是把钱用光了去买，若是后者，后续仓位如何管理（后面怎么卖）
# 2. 行情不连续或者跳跃(开盘价)
# 3. 冲击成本
# 4. 参数A B 初始仓位
# 5. 静态、动态比较
# 6. 手续费
