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
data.index = map(lambda x :datetime.datetime.strptime(x,'%Y-%m-%d'),list(data['Unnamed: 0']))
data = data.drop('Unnamed: 0',1)

capital = [300000, ]
BASE = 2728 # np.mean(data['OPEN']) + 1000
base = BASE
spread_up = 20
spread_down = 20 # np.std(data['OPEN']) / 40
position = [50, ]
buy = 0
sell = 0
amount = 2
direction = []
delta = [0,]
strike_price = []
for date in data.index:
    if delta[-1] > 0:                                                                                # 净多头
        if data.at[date, 'OPEN'] <= base - spread_down and data.at[date, 'OPEN'] <= capital[-1]:   # 买入条件
            buy = buy + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] + amount])
            capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
            direction.extend([1])
            delta.extend([buy - sell])
            base = BASE - delta[-1] * spread_down
        elif data.at[date, 'OPEN'] >= base + spread_up and position[-1] > 0:                        # 卖出条件
            sell = sell + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] - amount])
            capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
            direction.extend([-1])
            delta.extend([buy - sell])
            if delta[-1] == 0:
                base = BASE
            else:
                base = BASE - delta[-1] * spread_down
        else:                                                                                        # 不操作
            strike_price.extend([0])                                                                 
            position.extend([position[-1]])
            capital.extend([capital[-1]])
            direction.extend([0])
            delta.extend([delta[-1]])
    elif delta[-1] < 0:                                                                              # 净空头
        if data.at[date, 'OPEN'] >= base + spread_up and position[-1] > 0:                          # 卖出条件
            sell = sell + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] - amount])
            capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
            direction.extend([-1])
            delta.extend([buy - sell])
            base = BASE + abs(delta[-1]) * spread_up
        elif data.at[date, 'OPEN'] <= base - spread_down and data.at[date, 'OPEN'] <= capital[-1]: # 买入条件
            buy = buy + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] + amount])
            capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
            direction.extend([1])
            delta.extend([buy - sell])
            if delta[-1] == 0:
                base = BASE
            else:
                base = BASE + abs(delta[-1]) * spread_up
        else:                                                                                        # 无操作
            strike_price.extend([0])                                                                                   
            position.extend([position[-1]])
            capital.extend([capital[-1]])
            direction.extend([0])
            delta.extend([delta[-1]])
    else:                                                                                            # 零头寸
        if data.at[date, 'OPEN'] <= base - spread_down and data.at[date, 'OPEN'] <= capital[-1]:   # 买入条件
            buy = buy + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] + amount])
            capital.extend([capital[-1] - amount * data.at[date, 'OPEN']])
            direction.extend([1])
            delta.extend([buy - sell])
            base = BASE - delta[-1] * spread_down
        elif data.at[date, 'OPEN'] >= base + spread_up and position[-1] > 0:                        # 卖出条件
            sell = sell + 1
            strike_price.extend([data.at[date, 'OPEN']])
            position.extend([position[-1] - amount])
            capital.extend([capital[-1] + amount * data.at[date, 'OPEN']])
            direction.extend([-1])
            delta.extend([buy - sell])
            base = BASE + abs(delta[-1]) * spread_up
        else:
            strike_price.extend([0])                                                                                        # 无操作
            position.extend([position[-1]])
            capital.extend([capital[-1]])
            direction.extend([0])
            delta.extend([delta[-1]])

value = position[1:len(position)] * data['OPEN'] + capital[1:len(capital)]
strategy = value / value[0]
benchmark = data['OPEN'] / np.array(data.loc[data.index[0], ['OPEN']])

# 绘图1
fig1 = plt.plot(data.index, strategy, 'b', lw=1.5, label='strategy')
plt.plot(data.index, benchmark, 'g', lw=1.5, label='benchmark')
plt.legend(loc=0,numpoints=1,fancybox=True,shadow=True)
plt.grid(True)
plt.axis('tight')
plt.xlabel('time')
plt.ylabel('cummulative value')
plt.show()

# 绘图2
fig2,ax2_1= plt.subplots()
plt.plot(data.index,data['OPEN'],'g', lw=1.5,label='benchmark')
plt.legend(loc=1,fancybox=True,shadow=True)
ax2_2 = ax2_1.twinx()
plt.plot(data.index,position[1:len(position)],'b', lw=1.5,label='position')
plt.legend(loc=9,fancybox=True,shadow=True)
plt.grid(True)
plt.axis('tight')
plt.xlabel('time')
plt.show()

def get_sharp(time,strategy,gz_return):
    # period = time[-1] -time[0]
    period = len(np.array(time))
    return (log(strategy[-1] / strategy[0]) / (period / 365) - gz_return) / np.std(np.array(strategy))

# 策略表现
sharp = get_sharp(data.index,strategy,0.0390)
beta = pd.Series(strategy).corr(pd.Series(benchmark))
period = len(np.array(data.index))
strategy_returns = log(strategy[-1] / strategy[0]) / (period / 365)
benchmark_returns = log(benchmark[-1] / benchmark[0]) / (period / 365)
volatility_strategy = np.std(np.array(strategy))
volatility_benchmark = np.std(np.array(benchmark))
max_drawdown_strategy = (max(strategy) - min(strategy)) / max(strategy)
max_drawdown_benchmark = (max(benchmark) - min(benchmark)) / max(benchmark)
j = 0
for i in position:
    if i != 0:
        j = j + 1
holding_time = j / len(position)
print('Benchmark Returns        \t:\t %.4f%%' % (benchmark_returns * 100))
print('Strategy Returns         \t:\t %.4f%%' % (strategy_returns * 100))
print('Sharp                    \t:\t %.2f'   % (sharp))
print('Beta                     \t:\t %.4f'   % (beta))
print('Volatility of strategy   \t:\t %.2f%%' % (volatility_strategy * 100))
print('Volatility of benchmark  \t:\t %.2f%%' % (volatility_benchmark * 100))
print('Max Drawdown of strategy \t:\t %.2f%%' % (max_drawdown_strategy * 100))
print('Max Drawdown of benchmark\t:\t %.2f%%' % (max_drawdown_benchmark * 100))
print('Buy                      \t:\t %d'     % (buy))
print('Sell                     \t:\t %d'     % (sell))
print('Holding Time             \t:\t %.2f%%' % (holding_time * 100))


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
# 7. 策略表现指标究竟如何计算
    # https://www.zhihu.com/question/27264526/answer/147672695?utm_source=com.tencent.tim&utm_medium=social
    # https://community.bigquant.com/t/量化学堂-策略开发策略回测结果指标详解/257?utm_source=com.tencent.tim&utm_medium=social


# 定投策略
# 交易规则：当天开盘价相对于 平均持仓成本 ± x%,买入金额 ∓ 5 * x%
# 该规则既包含买入规则，又包含止盈规则
# 注意：卖出一部分后，持仓成本不变