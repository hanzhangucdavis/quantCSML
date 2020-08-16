#The given demo.m is probably WRONG




import numpy as np
Channel_N_beg = 4;
Channel_N=np.array(range(1,21))
Success=np.zeros_like(Channel_N)
FalseAlarm=np.zeros_like(Channel_N)
SNR=5
N=4
B=8e6
Bi = np.ones([1,N])*B
fnyq=320e6
Ei = np.ones([1,N])*10
Tnyq = 1/fnyq
K = 205
K0 = 10
L = 40
TimeResolution = Tnyq
TimeWin =np.array( [0,L*K-1,L*(K+K0)-1])*TimeResolution
Taui = np.array([0.8,0.6,0.4,0.2])*max(TimeWin);

ChannelNum = Channel_N + Channel_N_beg
fs = fnyq/L
m = ChannelNum
Patterns = np.random.choice(L-1,m,False)
Ts = 1/fs

t_axis = np.array(range(TimeWin[0],TimeWin[-1], TimeResolution))
t_axis_sig=np.array(range(TimeWin[0],TimeWin[1], TimeResolution))

x =np.zeros_like(t_axis_sig)
Soring = np.random.choice(L,N,False)



print(z)