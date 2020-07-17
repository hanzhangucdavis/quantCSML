%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Demo for Modulated Wideband Converter %
%             Version 1.0               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear,
close all 
clc
Chnanel_N_beg = 4;
tic
for signal_N = 1:8;
for coset_N=1:21
    Success(signal_N,coset_N)=0;
    FalseAlarm(signal_N,coset_N)=0;
for Simul_N=1:1000
%% Signal model
fprintf(1,'%d,%d\n',coset_N,Simul_N);
quant = 4;
SNR = 1000;                                   % Input SNR
N = signal_N;                                      % Number of bands (when counting  a band and its conjugate version separately)
B = 8e6;                                   % Maximal width of each band
Bi = ones(1,N)*B;
fnyq = 320e6;                                % Nyquist rate
Ei = ones(1,N)*10;                        % Energy of the i'th band
Tnyq = 1/fnyq;
K = 205;
K0 = 10;                                    % R*K0*L is reserved for padding zeros
L = 40;
TimeResolution = Tnyq;
TimeWin = [0  L*K-1 L*(K+K0)-1]*TimeResolution; % Time interval in which signal is observed
Taui = [0.8 0.6 0.4 0.2]*max(TimeWin);          % Time offest of the i'th band

%% Sampling parameters
ChannelNum = coset_N + Chnanel_N_beg;                            % Number of channels
fs = fnyq/L;
m = ChannelNum;                             % Number of channels
% sign alternating  mixing
Patterns = randperm(L-1,m);                 % Draw a random +-1 for mixing sequences
% calucations
Ts = 1/fs;
%% Signal Representation
t_axis = TimeWin(1)  : TimeResolution : TimeWin(end);     % Time axis
t_axis_sig  = TimeWin(1)  : TimeResolution : TimeWin(2);
% Signal Generation
% x = zeros(size(t_axis_sig));
% Soring = randperm(L,N);
% fi = (Soring-0.5)*B;      % Draw random carrier within [0, fnyq/2]
% for n=1:N
%     x = x+sqrt(Ei(n)) * sqrt(Bi(n))*sinc(Bi(n)*(t_axis_sig-Taui(n))) .* exp(j*2*pi*fi(n)*(t_axis_sig-Taui(n)));
% end
% x = [x, zeros(1,K0*L)];               % Zero padding
Nbps=1; M=2^Nbps;  % Modulation order=2/4/6 for QPSK/16QAM/64QAM
X= randi([M-M,M-1],N,K); % bit: integer vector
Xmod= qammod(X,M,'gray')/sqrt(1);
Soring = randperm(L-2,N);
Soring = Soring +1;
xf = zeros(1,K*L);
for i = 1:N
    xf(1,(Soring(i)-1)*K:(Soring(i)*K-1)) = Xmod(i,:);
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
% sig_r = sign(real(sig));
% sig_i = sign(imag(sig));
% sig = sig_r + j*sig_i;
%% Mixing
SignalSampleSequences = zeros(m,K+K0);
MixedSigSequences = zeros(m,length(t_axis));
%low-pass filtering
temp = zeros(1,(K+K0)*L);
temp(1:(K+K0)) = 1;
for channel=1:m
    SignalSampleSequences(channel,:) = downsample(sig,L,Patterns(channel));
    Nonzeroindeces = (0:K+K0-1)*L+Patterns(channel);
    MixedSigSequences(channel,Nonzeroindeces) = SignalSampleSequences(channel,:);
    MixedSigSequences(channel,:) = fft(MixedSigSequences(channel,:)).*temp;
end
decfactor = L;
Digital_time_axis = downsample(t_axis,decfactor);

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
aa = find(thre>0.25);
sparse = size(aa,2);
% N iterations at most, since we force symmetry in the support...
[ RecSupp] = RunOMP_Unnormalized(v(:,aa), A,sparse, 0, 0.2, 0);
% RecSuppSorted = sort(unique(RecSupp));
RecSuppSorted = RecSupp;
% Decide on success
Detection = intersect(RecSuppSorted,Soring);
Success(signal_N,coset_N) = Success(signal_N,coset_N) + numel(Detection);
False_alarm = setdiff(RecSuppSorted,Soring);
FalseAlarm(signal_N,coset_N) = FalseAlarm(signal_N,coset_N) + numel(False_alarm);
end
end
toc
TIME = toc;
Success(signal_N,:)=Success(signal_N,:)/Simul_N/N;
FalseAlarm(signal_N,:)=FalseAlarm(signal_N,:)/Simul_N/(L-N);
end
%% Recover the singal
A_S = A(:,RecSuppSorted);
hat_zn = pinv(A_S)*SignalSampleSequences;  % inverting A_S
hat_zt = zeros(size(hat_zn,1),length(x));
for ii = 1:size(hat_zt,1)                     % interpolate (by sinc)
    hat_zt(ii,:) = interpft(hat_zn(ii,:),L*length(hat_zn(ii,:)));
end

x_rec = zeros(1,length(x));
for ii = 1:size(hat_zt,1)                      % modulate each band to their corresponding carriers
    x_rec = x_rec+hat_zt(ii,:).*exp(j*2*pi*(RecSuppSorted(ii)-1)*fs.*t_axis);
end
x_rec = real(x_rec);
%% Analysis & Plots
% figure,
% plot(t_axis,x,'r'), axis([t_axis(1) t_axis(end) 1.1*min(x) 1.1*max(x) ])
% title('Original signal'), xlabel('t')
% grid on
% figure,plot(t_axis,sig,'r'), axis([t_axis(1) t_axis(end) 1.1*min(x) 1.1*max(x) ])
% title('Original noised signal'), xlabel('t')
% grid on
% figure, plot(t_axis,x_rec), axis([t_axis(1) t_axis(end) 1.1*min(x) 1.1*max(x) ]),
% grid on,
% title('Reconstructed signal'), xlabel('t')
% figure, plot(linspace(-2,2,length(x)),abs(fftshift(fft(x)))),hold on 
% figure, plot(linspace(-2,2,length(x_rec)),abs(fftshift(fft(x_rec)))),grid on 
figure;plot(((1:coset_N)+4)*1,Success,'-<');
title('detection probability'), xlabel('number of coset'),ylabel('Probability'),grid on;
legend('Signal: 1','Signal: 2','Signal: 3','Signal: 4','Signal: 5','Signal: 6','Signal: 7','Signal: 8')
figure;plot(((1:coset_N)+4)*1,FalseAlarm,'-<');
title('False Alarm probability'), xlabel('number of coset'),ylabel('Probability'),grid on;
legend('Signal: 1','Signal: 2','Signal: 3','Signal: 4','Signal: 5','Signal: 6','Signal: 7','Signal: 8')