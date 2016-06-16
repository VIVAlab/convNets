close all
clear all
%F1=importdata('12net_3channels_12ct_calibTh0.log');
%F1=importdata('12net_3channels_12ct_calib2Th40.log');

ich=1;
F3=importdata(sprintf('%s%d%s','20net_',ich,'channels_12ct_calib2Th35.log'));

precision=F3.data(:,8);
recall=F3.data(:,4);
threshold=F3.data(:,3);
windows=F3.data(:,6);
tp=F3.data(:,5);
Fscore=F3.data(:,1);
fp=F3.data(:,2);
percSwind=(tp+fp)./windows;

ich=3;
F1=importdata(sprintf('%s%d%s','20net_',ich,'channels_12ct_calib2Th35.log'));
precision1=F1.data(:,8);
recall1=F1.data(:,4);
threshold1=F1.data(:,3);
windows1=F1.data(:,6);
tp1=F1.data(:,5);
Fscore1=F1.data(:,1);
fp1=F1.data(:,2);
percSwind1=(tp1+fp1)./windows1;

figure
h1=plot(recall,precision);
hold on 
scatter(recall,precision,'b')
xlabel('recall'),ylabel('precision')
title('1 and 3-channel precision/recall curve')

N=length(threshold);
b = num2str(threshold); c = cellstr(b);
dx = 0.001; dy = 0.001;
t0=text(recall+dx, precision+dy, c);

clear t0
h2=plot(recall1,precision1,'r')
hold on 
scatter(recall1,precision1,'r')
xlabel('recall'),ylabel('precision')
title('precision/recall curve')
legend([h1 h2],'gray','RGB')
ylim([0 1])

%{
N=length(threshold);
b = num2str(threshold); c = cellstr(b);
dx = 0.01; dy = 0.01;
t0=text(recall1+dx, precision1+dy, c);
clear t0
%}
%{
figure
plot(1-recall,fp)
hold on
scatter(1-recall,fp,'r')
xlabel('miss rate'),ylabel('false positives')

N=length(threshold);
b = num2str(threshold); c = cellstr(b);
dx = 0.02; dy = 0.02;
t0=text((1-recall)+dx, fp+dy, c);

figure
plot(threshold,Fscore)
hold on
scatter(threshold,Fscore,'r')
xlabel('threshold'),ylabel('F-score')
%}
figure
h1=plot(threshold,recall);
hold on
scatter(threshold,recall,'b')
h2=plot(threshold1,recall1,'r');
scatter(threshold1,recall1,'r')
xlabel('threshold'),ylabel('recall')
title('recall wrt threshold')
legend([h1 h2],'gray','RGB')
ylim([0 1])

%{
figure
plot(threshold,precision)
hold on
scatter(threshold,precision,'r')
xlabel('threshold'),ylabel('precision')

normfp=fp/max(fp);
figure
plot(1-recall,normfp)
hold on
scatter(1-recall,normfp,'r')
xlabel('miss rate'),ylabel('normalized false positives')

N=length(threshold);
b = num2str(threshold); c = cellstr(b);
dx = 0.02; dy = 0.02;
t0=text((1-recall)+dx, normfp+dy, c);
%}
figure
h1=plot(recall,percSwind);
hold on
scatter(recall,percSwind,'b')
xlabel('recall'),ylabel('percent survived windows')
title('recall and survived windows')
h2=plot(recall1,percSwind1,'r');
scatter(recall1,percSwind1,'r')
N=length(threshold);
b = num2str(threshold); c = cellstr(b);
dx = 0.002; dy = 0.002;
t0=text((recall)+dx, percSwind+dy, c);
legend([h1 h2],'gray','RGB')
ylim([0 1])
N=length(threshold1);
b = num2str(threshold1); c = cellstr(b);
dx = 0.002; dy = 0.002;
t0=text((recall1)+dx, percSwind1+dy, c);
%{
figure
plot(threshold,sqrt(percSwind.*recall))
hold on
scatter(threshold,sqrt(percSwind.*recall),'r')
xlabel('threshold'),ylabel('sqrt(percSwind.*recall)')
%}