clear all 
close all
clc
tic
signal_N = 400;
B = 8e6;                                   % width of each band
fnyq = 320e6;                              % Nyquist rate
Tnyq = 1/fnyq;
K = 20;
L = 10;
TimeResolution = Tnyq;
support_all = 1:10;
% Signal Generation


for N = 1:10
    x_r = [];
    x_i = [];
    Soring_N_t = [];
    for iii = 1:100;
        Nbps=1; M=2^Nbps;                              % Modulation order=2/4/6 for QPSK/16QAM/64QAM
        X= randi([M-M,M-1],N,K);                       % bit: integer vector
        Xmod= qammod(X,M,'gray')/sqrt(1);
        Soring_all = combntns(support_all,N);
        Soring_Num = size(Soring_all,1);
        for ii = 1:Soring_Num
            Soring = Soring_all(ii,:);
            Soring_N = zeros(1,length(support_all));
            Soring_N(1,Soring) = 1;
        
            xf = zeros(1,K*L);
            for i = 1:N
                xf(1,((Soring(i)-1)*K+1):(Soring(i)*K)) = Xmod(i,:);
            end
            x = ifft(xf);
            x1_r = real(x);
            x1_i = imag(x);
            x_r = [x_r;x1_r];
            x_i = [x_i;x1_i];
            Soring_N_t = [Soring_N_t;Soring_N];
        end
    end
    csvwrite('xr.csv',x_r);
    csvwrite('xt.csv',x_i);
    csvwrite('support.csv',Soring_N_t);
    str0 = 'signal_';
    str1 = num2str(N);
    str3 = 'C:\Users\yangj\Desktop\data_signal_N\';
    folder_name = [str0 str1];
    mkdir(str3,folder_name);
    DST_PATH_t = [str3 folder_name];
    movefile('xr.csv',DST_PATH_t);
    movefile('xt.csv',DST_PATH_t);
    movefile('support.csv',DST_PATH_t);
end

toc