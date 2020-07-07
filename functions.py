import numpy as np
def supportGen(Num,Len,ratio):
    #generating support vector as label
    tmp=np.zeros([Num,Len])
    for i in range(Num):
        tmp[i,np.random.choice(Len,ratio,False)]=1
    return tmp

def signalGen(sup,width):
    ret=np.zeros([sup.shape[0],sup.shape[1]*width])
    #generating the signal on frequency domain
    for i in range(sup.shape[1]):
        tmp=np.repeat(np.expand_dims(sup[:,i],axis=1),width,axis=1)
        ret[:,i*width:i*width+width]=tmp
    #using square wave as a placeholder
    return ret
def freqToTime(freq):
    #padded a 0 for the DC
    freqPad=np.zeros([freq.shape[0],1])
    freqPad=np.concatenate([freqPad,freq],axis=1)

    return np.fft.irfft(freqPad)
def quantify(input,quant):
    # use average energy to quantify signal
    if quant<0:
        return input,0
    stair=2*np.power(np.average(np.power(input,2)),0.5)/(quant)
    tmp=(input/stair).astype(np.int32)
    tmp = np.clip(tmp,-quant,quant)
    return tmp,stair
def quantifyMax(input,quant):
    #use peak energy to quantify signal
    if quant<0:
        return input,0
    stair=np.max(np.abs(input))/quant
    tmp=(input/stair).astype(np.int32)
    return tmp,stair
def generateCosets(Len,ratio,width):
    tmp=np.random.choice(Len, ratio, False)
    tmp=np.sort(tmp)
    tmp2=[]
    for i in range(width):
        tmp2.append(tmp+i*Len)
    return np.concatenate(tmp2)
def mulCoset(orig,cosets):
    return orig[:,cosets]
def genData(a,cosets=None):
    support = supportGen(a.trainingNum, a.subBandNum, a.occupyNum)
    sendFreq = signalGen(support, a.subBandWidth)
    sendTime = freqToTime(sendFreq)
    if cosets is None:
        cosets = generateCosets(int(sendTime.shape[1] / a.subBandWidth), a.cosetNum, a.subBandWidth)
    recTime = mulCoset(sendTime, cosets)
    quantTime, quantStair = quantify(recTime, a.quantifyLevel)
    return support, sendTime, cosets,quantTime,quantStair
def threshold(inp,thres):
    inp = 1 * (inp > thres)
    return inp
def errorRate(inp1,inp2):
    #all those 3 only works with binary matrices in array form
    return np.sum(1*(abs(inp1-inp2)>0))/inp2.size
def falseAlarm(inppred,inpori):
    totalNum=np.sum(inppred)
    tmp=inpori * inppred
    return 1-np.sum(tmp)/totalNum
def misdetection(inppred,inpori):
    totalNum=np.sum(inpori)
    tmp=inpori * inppred
    return 1-np.sum(tmp)/totalNum