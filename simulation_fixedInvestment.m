%% ����ڴ�
clear;clc;close all;

% ���ö�Ͷ����
n = 100;

% ������Ͷ��
time = linspace(1,n,n);

% ��ȷ��ͼ
m = 20; % �Ŵ���
x = linspace(1,n,m * n);

%% ѡ����������
type = input('��������������ѡ��1:�������ǣ�2��ָ�����ǣ�3���������ǣ�4:��������Ĭ��������):\n');
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

%% ����ƽ���ɱ�
share = 1 ./ cost;
totalShare = sum(share);
averageCost = n / totalShare;

%% ���㶨Ͷ����һ������ĸ���
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

%% ���
fprintf('%d�ڶ�Ͷ��ƽ���ɱ���:',n);
averageCost
fprintf('%d�ڶ�Ͷ��Խһ������ĸ��ʣ�',n);
p
f1 = plot(x,y);
title('��Ͷ����ͼ');
xlabel('ʱ��');
ylabel('�۸�');
hold on
for i = 1 : n
    f2 = plot(time(i),cost(i),'r.','markersize',8);
    pause(0.1)
    hold on
end
f3 = plot(x,averageCost * ones(1,n * m),'g');
legend([f1,f2,f3],'��������','��Ͷ��','ƽ���ɱ�',2);
