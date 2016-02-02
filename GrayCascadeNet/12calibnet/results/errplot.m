
clear all
Tr=importdata('train.log');
Te=importdata('test.log');
Lte=length(Te.data);
Ltr=length(Tr.data);
Eopt=zeros(size(Te.data));
Epir=zeros(size(Te.data));
GL=zeros(size(Te.data));
GL2=zeros(size(Te.data));
GL3=zeros(size(Te.data));
GL4=zeros(size(Te.data));
idxmin=zeros(size(Te.data));
idxgapmin=zeros(size(Te.data));
T=100;
for t=1:Lte;
    [Eopt(t),idxmin(t)]=min(1-Te.data(1:t)/100);
    if 1-Te.data(t)/100==Eopt(t)
       idxgapmin(t)=0;
    else
       idxgapmin(t)=idxgapmin(max(t-1,1))+1;
    end
    Epir(t)=max(1-Te.data(max(1,t-T+1):t)/100);
    GL(t)=((1-Te.data(t)/100)/Eopt(t)-1);
    GL2(t)=((1-Te.data(t)/100)/Epir(t)-1);
    GL3(t)=GL(t)-GL2(t);
    GL4(t)=min(GL3(max(1,t-T+1):t));
end
subplot(3,1,1)
plot((1:Lte),1-Te.data/100,(1:Ltr),1-Tr.data/100,(1:Lte),Eopt), legend('test','train','Eopt')
subplot(3,1,2)
plot((1:Lte),idxmin), legend('idxmin')
subplot(3,1,3)
plot((1:Lte),idxgapmin), legend('idexgap')

TrF=importdata('train(flip_noDropout).log');
TeF=importdata('test(flip_noDropout).log');

LteF=length(TeF.data);
LtrF=length(TrF.data);
EoptF=zeros(size(TeF.data));

for t=1:LteF;
    EoptF(t)=min(1-TeF.data(1:t)/100);
end
TrD=importdata('train(FullDropout).log');
TeD=importdata('test(FullDropout).log');

LteD=length(TeD.data);
LtrD=length(TrD.data);
EoptD=zeros(size(TeD.data));

for t=1:LteD;
    EoptD(t)=min(1-TeD.data(1:t)/100);
end

TrpD=importdata('train(CNND).log');
TepD=importdata('test(CNND).log');

LtepD=length(TepD.data);
LtrpD=length(TrpD.data);
EoptpD=zeros(size(TepD.data));

for t=1:LtepD;
    EoptpD(t)=min(1-TepD.data(1:t)/100);
end

Trn=importdata('train().log');
Ten=importdata('test().log');

Lten=length(Ten.data);
Ltrn=length(Trn.data);
Eoptn=zeros(size(Ten.data));

for t=1:Lten;
    Eoptn(t)=min(1-Ten.data(1:t)/100);
end
figure
plot((1:LtepD),EoptpD,(1:LteD),EoptD,(1:Lten),Eoptn,(1:Lte),Eopt), legend('Eopt_{PartialDropout}','Eopt_{Dropout}','Eopt_{noDropout|noFlip}','Eopt_{F}')
%{
Trva=importdata('train.log');
Teva=importdata('test.log');

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


figure
plot((1:Lte),Eopt,(1:Ltev),Eoptv,(1:Lteva),Eoptva), legend('Eopt_{fixed}','Eopt_{var}','Eopt_{vara}')
figure
plot((1:Lteva),1-Teva.data/100,(1:Ltrva),1-Trva.data/100), legend('test_{vara}','train_{vara}')

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