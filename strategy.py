import datetime as dt
import matplotlib.pyplot as plt
from math import *
import numpy as np
import pandas as pd
import scipy as sp
import pyalgotrade
import jhtalib

global data,capital,BASE,base,spread_up,spread_down,position,buy,sell,amount,direction,delta,strike_price,r

# 启动Wind并获取数据
from WindPy import *
w.start()
w.start(waitTime=60)  # 命令超时时间设置成60秒
w.isconnected()  # 即判断WindPy是否已经登陆成功
wsd_data = w.wsd('000001.SH', 'open,high,low,close,volume,amt',
                 '2012-1-1', '2014-3-24', 'Fill=Previous')

# 将API数据转换为DataFrame数据，并保存到本地文件
wsd_data_ndarray = np.transpose(np.array(wsd_data.Data))
data = pd.DataFrame(wsd_data_ndarray, index=wsd_data.Times,
                    columns=wsd_data.Fields)
data.to_csv('D:\Documents\GitHub\investment\DATA.csv',index=True,sep=",")

# 读取数据
import datetime
data = pd.read_csv('D:\Documents\GitHub\investment\DATA.CSV')
data.index = map(lambda x :dt.datetime.strptime(x,'%Y-%m-%d'),list(data['Unnamed: 0']))
data = data.drop('Unnamed: 0',1)

capital = [100000, ] # 本金
BASE = data['OPEN'][0] # 初始基准价 np.mean(data['OPEN']) + 1000 
base = BASE # 变动基准价，初始值设为初始基准价
spread_up = 20 # 向上变动价差
spread_down = 20 # 向下变动价差 np.std(data['OPEN']) / 40
position = [0.5 * capital[-1] // base, ] # 初始头寸
buy = 0 # 买入笔数
sell = 0 # 卖出笔数
amount = 0.05 * capital[-1] // base # 每次策略的执行交易量
direction = [] # 每次策略执行的交易方向
delta = [0,] # 净头寸（仅指买入量与卖出量的差，不包含初始头寸）
strike_price = [] # 每次策略执行的交易价格
r = 0.03 # 无风险利率

def buy_action():
    global capital,base,position,buy,sell,direction,delta,strike_price
    buy = buy + 1
    strike_price.extend([data.at[date, 'OPEN']])
    direction.extend([1])
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
    strike_price.extend([data.at[date, 'OPEN']])
    direction.extend([-1])
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
    strike_price.extend([0])                                                                 
    position.extend([position[-1]])
    capital.extend([(capital[-1]) * pow(1 + r,1 / 365)])
    direction.extend([0])
    delta.extend([delta[-1]])
    
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

# 策略表现
period = len(np.array(data.index))   # 回测天数
value = position[1:len(position)] * data['OPEN'] + capital[1:len(capital)] # 股票+现金的价值
strategy = value / value[0] # 以序列第一天为1
benchmark = data['OPEN'] / np.array(data.loc[data.index[0], ['OPEN']]) # 取回测期间开盘价并设置第一天为1

total_return_strategy = strategy[-1] / strategy[0] - 1    # 策略总回报
total_return_benchmark = benchmark[-1] / benchmark[0] - 1 # 基准总回报

annual_return_strategy = ((1 + total_return_strategy) ** (252.0 / period) - 1) # 策略年化回报
annual_return_benchmark = ((1 + total_return_benchmark) ** (252.0 / period) - 1) # 基准年化回报

Open = data['OPEN']  # 取回测期间开盘价
daily_return_benchmark = np.array(Open[1:period]) / np.array(Open[0:-1]) - 1 # 基准每日回报
daily_return_strategy = np.array(value[1:period]) / np.array(value[0:-1]) - 1 # 策略每日回报
beta = pd.Series(daily_return_strategy).corr(pd.Series(daily_return_benchmark)) # 贝塔值

alpha = annual_return_strategy - (0.03 + beta * (annual_return_benchmark - 0.03)) # 阿尔法值

daily_volitility_strategy = np.std(daily_return_strategy) # 策略日波动率
annual_volitility_strategy = daily_volitility_strategy * sqrt(252) # 策略年化波动率
sharp = (annual_return_strategy - 0.03) / annual_volitility_strategy # 夏普比率

daily_volitility_benchmark = np.std(daily_return_benchmark) # 基准日波动率
annual_volitility_benchmark = np.std(daily_return_benchmark) * sqrt(252) # 基准年化波动率
information_ratio = (annual_return_strategy - annual_return_benchmark) / (np.std(daily_return_strategy - daily_return_benchmark) * sqrt(252)) # 信息比率

max_drawdown_strategy = (min(strategy) - max(strategy)) / max(strategy) # 策略最大回撤
max_drawdown_benchmark = (min(benchmark) - max(benchmark)) / max(benchmark) # 基准最大回撤

j = 0
for i in position:
    if i != 0:
        j = j + 1
holding_time = j / len(position) # 持仓时间

print('基准总回报        \t:\t  %.4f%%' % (total_return_benchmark* 100))
print('策略总回报        \t:\t  %.4f%%' % (total_return_strategy * 100))
print('基准年化回报      \t:\t  %.4f%%' %(annual_return_benchmark * 100))
print('策略年化回报      \t:\t  %.4f%%' %(annual_return_strategy * 100))
print('基准日波动率      \t:\t  %.4f%%' % (daily_volitility_benchmark * 100))
print('策略日波动率      \t:\t  %.4f%%' % (daily_volitility_strategy * 100))
print('基准年化波动率    \t:\t  %.4f%%' % (annual_volitility_benchmark * 100))
print('策略年化波动率    \t:\t  %.4f%%' % (annual_volitility_strategy * 100))
print('夏普比率         \t:\t  %.2f'   % (sharp))
print('贝塔             \t:\t  %.4f'   % (beta))
print('阿尔法           \t:\t  %.2f%%' % (alpha * 100))
print('信息比率         \t:\t  %.4f'   % (information_ratio))
print('基准最大回撤     \t:\t  %.2f%%' % (max_drawdown_benchmark * 100))
print('策略最大回撤     \t:\t  %.2f%%' % (max_drawdown_strategy * 100))
print('买入笔数         \t:\t  %d'     % (buy))
print('卖出笔数         \t:\t  %d'     % (sell))
print('持仓时间         \t:\t  %.2f%%' % (holding_time * 100))

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