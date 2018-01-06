import datetime as dt
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
import scipy as sp
import pyalgotrade
import jhtalib
from WindPy import *

# 启动Wind并获取数据
w.start()
w.start(waitTime=60)  # 命令超时时间设置成60秒
w.isconnected()  # 即判断WindPy是否已经登陆成功
wsd_data = w.wsd('000001.SH', 'open,high,low,close,volume,amt',
                 '2012-1-1', '2014-1-1', 'Fill=Previous')

# 将API数据转换为DataFrame数据，并保存到本地文件
wsd_data_ndarray = np.transpose(np.array(wsd_data.Data))
data = pd.DataFrame(wsd_data_ndarray, index=wsd_data.Times,
                    columns=wsd_data.Fields)
# print(data)
# data.to_csv('D:\Documents\GitHub\investment\DATA.csv',index=True,sep=",")

# # 读取数据
# data = pd.read_csv('D:\Documents\GitHub\investment\DATA.CSV')
# array = data[['OPEN','HIGH','LOW','CLOSE','VOLUME','AMT']]
# print(data['Unnamed: 0'])
# data = pd.DataFrame(array,index=data['Unnamed: 0'],columns=['Open','High','Low','Close','Volume','Amt'])
# print(data)


capital = [300000, ]
base = 2212
print(base)
spread = 50
print(spread)
position = [15, ]
max_trader = position[0]  # 最大买卖股票数
b = 1
s = 1
N = 3  # 网格数
direction = [0, ]
for date in data.index:
    # 如果某天收盘价小于基准价 - b 倍价差
    if data.at[date, 'CLOSE'] <= base - max(b - s + 1, 1) * spread:
        if data.at[date, 'OPEN'] <= capital[-1]:                   # 且剩余资金大于等于第二天的开盘价
            b = b + 1
            position.extend([position[-1] + max_trader / N])
            # 买入股票
            capital.extend([capital[-1] - max_trader /
                            N * data.at[date, 'OPEN']])
            direction.extend([1])
        else:
            position.extend([position[-1]])
            capital.extend([capital[-1]])
            direction.extend([0])
    # 如果某天收盘价大于基准价 + s 倍价差 且持仓大于 0
    elif data.at[date, 'CLOSE'] >= base + max(s - b + 1, 1) * spread:
        if position[-1] > 0:
            s = s + 1
            position.extend([position[-1] - max_trader / N])
            # 卖出股票，
            capital.extend([capital[-1] + max_trader /
                            N * data.at[date, 'CLOSE']])
            direction.extend([-1])
        else:
            position.extend([position[-1]])
            capital.extend([capital[-1]])
            direction.extend([0])
    else:
        position.extend([position[-1]])
        capital.extend([capital[-1]])
        direction.extend([0])

value = position[1:len(position)] * data['CLOSE'] + capital[1:len(capital)]
strategy = value / value[0]
benchmark = data['CLOSE'] / np.array(data.loc[data.index[0], ['CLOSE']])

# 绘图
fig1, ax1_1 = plt.subplots()
plt.plot(data.index, strategy, 'b', lw=1.5, label='strategy')
plt.legend(loc=1)
plt.plot(data.index, benchmark, 'g', lw=1.5, label='benchmark')
plt.ylabel('cummulative value')
plt.legend(loc=1,fancybox=True,shadow=True)
ax1_2 = ax1_1.twinx()
plt.plot(data.index, position[1:len(position)], 'k', lw=1.5, label='position')
plt.ylabel('position')
plt.legend(loc=2,numpoints=1,fancybox=True,shadow=True)
plt.grid(True)
plt.axis('tight')
plt.xlabel('time')
plt.show()

def get_sharp(time,strategy,gz_return):
    period = time[-1] -time[0]
    return (log(strategy[-1] / strategy[0]) / (period.days / 365) - gz_return) / np.std(np.array(strategy))

def get_beta(time,strategy,benchmark,gz_return):
    period = time[-1] -time[0]
    strategy_returns = log(strategy[-1] / strategy[0]) / (period.days / 365)
    benchmark_returns = log(benchmark[-1] / benchmark[0]) / (period.days / 365)
    return [(strategy_returns - gz_return) / (benchmark_returns - gz_return),benchmark_returns,strategy_returns]


# 交易时间与对应的交易量，成交价格，交易方向
# 仓位变化

# 策略表现
# 收益指标：年化收益率，基准年化收益率，α，夏普比率，信息比率，最大回撤
# 风险指标：β，收益波动率，最大回撤
sharp = get_sharp(data.index,strategy,0.0390)
[beta,benchmark_returns,strategy_returns] = get_beta(data.index,strategy,benchmark,0.0390)
volatility = np.std(np.array(strategy))
max_drawdown = (max(strategy) - min(strategy)) / max(strategy)
print('Benchmark Returns\t:\t %.2f' % (benchmark_returns))
print('Strategy Returns \t:\t %.2f' % (strategy_returns))
print('Sharp            \t:\t %.2f' % (sharp))
print('Beta             \t:\t %.2f' % (beta))
print('Volatility       \t:\t %.2f%%' % (volatility * 100))
print('Max Drawdown     \t:\t %.2f%%' % (max_drawdown * 100))
print('Buy              \t:\t %d' % (b))
print('Sell             \t:\t %d' % (s))
