%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for Modulated Wideband Converter %
%             Version 1.0               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear,
close all 
clc
coset_begin = 1;
tic
for signal_N = 1:8;
for coset_N=1:9
    Detection(signal_N,coset_N)=0;
    FalseAlarm(signal_N,coset_N)=0;
for Simul_N=1:10000
%% Signal model
fprintf(1,'%d,%d,%d\n',signal_N,coset_N,Simul_N);
quant = 16;
SNR = 1000;                                      % Input SNR
N = signal_N;                                    % Number of bands (when counting  a band and its conjugate version separately)
B = 8e6;                                         % Maximal width of each band
Bi = ones(1,N)*B;
fnyq = 320e6;                                    % Nyquist rate
Ei = ones(1,N)*10;                               % Energy of the i'th band
Tnyq = 1/fnyq;
K = 20;
K0 = 10;                                         % R*K0*L is reserved for padding zeros
L = 10;
TimeResolution = Tnyq;
%% Sampling parameters
ChannelNum = coset_N + coset_begin-1;           % Number of channels
fs = fnyq/L;
m = ChannelNum;                                 % Number of channels
% sign alternating  mixing
Patterns = randperm(L-1,m);                     % Draw a random +-1 for mixing sequences
% calucations
Ts = 1/fs;
%% Signal Representation
Nbps=1; M=2^Nbps;                              % Modulation order=2/4/6 for QPSK/16QAM/64QAM
X= randi([M-M,M-1],N,K);                       % bit: integer vector
Xmod= qammod(X,M,'gray')/sqrt(1);
Soring = randperm(L,N);
xf = zeros(1,K*L);
for i = 1:N
    xf(1,((Soring(i)-1)*K+1):(Soring(i)*K)) = Xmod(i,:);
end
x = ifft(xf);
x = [x, zeros(1,K0*L)];
%% Noise Generation
noise = randn(1,(K+K0)*L)+j*randn(1,(K+K0)*L);              % Generate white Gaussian nosie within [-fnyq,fnyq]
% Calculate energies
NoiseEnergy = norm(noise)^2;
SignalEnergy = norm(x)^2;
CurrentSNR = SignalEnergy/NoiseEnergy;
SNR_val = 10^(SNR/10);          % not dB
sig = x + noise*sqrt(CurrentSNR/SNR_val);

sig_r = real(sig);
sig_r_m = max(abs(sig_r));
sig_i = imag(sig);
sig_i_m = max(abs(sig_i));
maxvalue = max(sig_r_m,sig_i_m);
quant_level = maxvalue/quant;
sig = round(fix(sig./quant_level)/2);
%% Mixing
SignalSampleSequences = zeros(m,K+K0);
MixedSigSequences = zeros(m,length(sig));
%low-pass filtering
temp = zeros(1,(K+K0)*L);
temp(1:(K+K0)) = 1;
for channel=1:m
    SignalSampleSequences(channel,:) = downsample(sig,L,Patterns(channel));
    Nonzeroindeces = (0:K+K0-1)*L+Patterns(channel);
    MixedSigSequences(channel,Nonzeroindeces) = SignalSampleSequences(channel,:);
    MixedSigSequences(channel,:) = fft(MixedSigSequences(channel,:)).*temp;
end
%% CTF block
% define matrices for fs=fp
theta = exp(-j*2*pi/L);
F = theta.^(Patterns'*[0:L-1]);
% This is for digital input only. Note that when R -> infinity,
% D then coincides with that of the paper

D = 1/(Tnyq*L);
A = F*D;
A = conj(A);
% Frame construction
Q =  MixedSigSequences * MixedSigSequences';
% decompose Q to find frame V
[V,d] = eig(Q);
d = diag(d);
v = V*diag(sqrt(d));
lambda = zeros(1,m);
lambda(1) = d(1);
for i = 1:m-1
    a(i) = sqrt(0.5*(15/((i+1).^2+2)-sqrt(225/((i+1).^2+2).^2-180*(i+1)/(K*((i+1).^2-1)*((i+1).^2+2)))));
    r(i) = exp(-2*a(i));
    J(i) = (1-r(i))/(1-r(i).^(i+1));
    sigma(i) = sum(d(1:(i+1)))/(i+1);
    lambda(i+1) = (i+1)*J(i)*sigma(i);
end
thre = abs(lambda - d')./lambda;
aaa = 0.02;
aaaa = 0.01;
aa = find(thre>(0.4-aaa*signal_N+aaaa*coset_N));
sparse = size(aa,2);
% N iterations at most, since we force symmetry in the support...
[ RecSupp] = RunOMP_Unnormalized(v(:,aa), A,sparse, 0, 0.2, 0);
% RecSuppSorted = sort(unique(RecSupp));
RecSuppSorted = RecSupp;
% Decide on success
Detection1 = intersect(RecSuppSorted,Soring);
Detection(signal_N,coset_N) = Detection(signal_N,coset_N) + numel(Detection1);
False_alarm = setdiff(RecSuppSorted,Soring);
FalseAlarm(signal_N,coset_N) = FalseAlarm(signal_N,coset_N) + numel(False_alarm);
end
end
toc
TIME = toc;
Detection(signal_N,:)=Detection(signal_N,:)/Simul_N/N;
FalseAlarm(signal_N,:)=FalseAlarm(signal_N,:)/Simul_N/(L-N);
end
%% Analysis & Plots
figure;plot(((1:coset_N)+coset_begin-1)*1,Detection,'-<');
title('Detection probability'), xlabel('number of coset'),ylabel('Probability'),grid on;
legend('Signal: 1','Signal: 2','Signal: 3','Signal: 4','Signal: 5','Signal: 6','Signal: 7','Signal: 8')
figure;plot(((1:coset_N)+coset_begin-1)*1,FalseAlarm,'-<');
title('False Alarm probability'), xlabel('number of coset'),ylabel('Probability'),grid on;
legend('Signal: 1','Signal: 2','Signal: 3','Signal: 4','Signal: 5','Signal: 6','Signal: 7','Signal: 8')