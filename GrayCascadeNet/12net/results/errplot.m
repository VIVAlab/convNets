close all
clear all
Tr=importdata('train.log');
Te=importdata('test.log');
%Tr=importdata('train_fixedLpatch.log');
%Te=importdata('test_fixedLpatch.log');
Lte=length(Te.data);
Ltr=length(Tr.data);
Eopt=zeros(size(Te.data));
Epir=zeros(size(Te.data));
%GL=zeros(size(Te.data));
GL2=zeros(size(Te.data));
GL3=zeros(size(Te.data));
GL4=zeros(size(Te.data));
idxmin=zeros(size(Te.data));
T=100;
for t=1:Lte;
    [Eopt(t),idxmin(t)]=min(1-Te.data(1:t)/100);
    Epir(t)=max(1-Te.data(max(1,t-T+1):t)/100);
    %GL(t)=((1-Te.data(t)/100)/Eopt(t)-1);
    GL2(t)=((1-Te.data(t)/100)/Epir(t)-1);
    %GL3(t)=GL(t)-GL2(t);
    GL4(t)=min(GL3(max(1,t-T+1):t));
end
subplot(2,1,1)
plot((1:Lte),1-Te.data/100,(1:Ltr),1-Tr.data/100,(1:Lte),Eopt), legend('test','train','Eopt')
subplot(2,1,2)
plot((1:Lte),idxmin), legend('idxmin')
Trv=importdata('train_var.log');
Tev=importdata('test_var.log');

Ltev=length(Tev.data);
Ltrv=length(Trv.data);
Eoptv=zeros(size(Tev.data));
GLv=zeros(size(Tev.data));
for t=1:Ltev;
    Eoptv(t)=min(1-Tev.data(1:t)/100);
    GLv(t)=((1-Tev.data(t)/100)/Eoptv(t)-1);
    
end
Trva=importdata('train(variable_augmented).log');
Teva=importdata('test(variable_augmented).log');

Lteva=length(Teva.data);
Ltrva=length(Trva.data);
Eoptva=zeros(size(Teva.data));
GLva=zeros(size(Teva.data));
for t=1:Lteva;
    Eoptva(t)=min(1-Teva.data(1:t)/100);
    GLva(t)=((1-Teva.data(t)/100)/Eoptva(t)-1);
    
end
figure
plot((1:Ltev),1-Tev.data/100,(1:Ltrv),1-Trv.data/100,(1:Ltev),GLv,(1:Ltev),Eoptv), legend('test','train','GL','Eopt')

TrP=importdata('train(P).log');
TeP=importdata('test(P).log');
LteP=length(TeP.data);
LtrP=length(TrP.data);
EoptP=zeros(size(TeP.data));
for t=1:LteP;
    EoptP(t)=min(1-TeP.data(1:t)/100);    
end

figure
plot((1:Lte),Eopt,(1:Ltev),Eoptv,(1:Lteva),Eoptva,(1:LteP),EoptP), legend('Eopt_{Pnoa}','Eopt_{var}','Eopt_{vara}','EoptP')
figure
plot((1:Lteva),1-Teva.data/100,(1:Ltrva),1-Trva.data/100,(1:LtrP),1-TrP.data/100), legend('test_{vara}','train_{vara}')
%}
%{
alph=.9;
A=[1 alph];
B=[1-alph];
trfilt=filter(B,A,Tr.data);
tefilt=filter(B,A,Te.data);
figure
plot((1:Lte),tefilt,(1:Ltr),trfilt), legend('test','train')
%}