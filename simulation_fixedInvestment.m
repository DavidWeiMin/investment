%% 清空内存
clear;clc;close all;

% 设置定投期数
n = 100;

% 创建定投点
time = linspace(1,n,n);

% 精确绘图
m = 20; % 放大倍数
x = linspace(1,n,m * n);

%% 选择行情走势
type = input('请输入行情走势选择（1:线性上涨；2：指数上涨；3：对数上涨；4:震荡上升；默认正弦震荡):\n');
switch type
    case 1
        cost = time / 25;
        y = x / 25;
    case 2
        cost = exp(3 * time / n) / 5;
        y = exp(3 * x / n) / 5;
    case 3
        cost = 1 + log(time);
        y = 1 + log(x);
    case 4
        cost =(1 + time .* abs(sin(0.3*time))) / 100;
        y =(1 + x .* abs(sin(0.3*x))) / 100;
    case 5
        cost = 2 * rand(1,n) .* abs(sin(time));
        y = 2 * rand(1,n * m) .* abs(sin(x));
    otherwise
        cost = 2 + sin(time);
        y = 2 + sin(x);
end

%% 计算平均成本
share = 1 ./ cost;
totalShare = sum(share);
averageCost = n / totalShare;

%% 计算定投优于一次买入的概率
switch type
    case 1
        p = (n - 25 * averageCost) / (n - 1);
    case 2
        p = (n - 100 * log(5 * averageCost) / 3) / (n - 1);
    case 3
        p = (n - exp(averageCost - 1)) / (n - 1);
    case 4
        p = 0;
    case 5
        p=0;
    otherwise
        p(1,1) = (pi - 2 * asin(averageCost-2)) * round((n-1) / (2 * pi)) / (n-1);
        p(1,2) = (pi - 2 * asin(averageCost-2)) * round((n-1) / (2 * pi)+1) / (n-1);
end

%% 输出
fprintf('%d期定投的平均成本是:',n);
averageCost
fprintf('%d期定投超越一次买入的概率：',n);
p
f1 = plot(x,y);
title('定投曲线图');
xlabel('时间');
ylabel('价格');
hold on
for i = 1 : n
    f2 = plot(time(i),cost(i),'r.','markersize',8);
    pause(0.1)
    hold on
end
f3 = plot(x,averageCost * ones(1,n * m),'g');
legend([f1,f2,f3],'行情走势','定投点','平均成本',2);
