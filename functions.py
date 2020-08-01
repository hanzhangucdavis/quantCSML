import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def conductTraining(model,opt,quantTimeTrain,supportTrain,quantTimeTest,supportTest):
    perep = int(opt.epochs / opt.divideNum)
    historytr = []
    historyva = []
    #test_loss, test_acc = model.evaluate(quantTimeTrain, supportTrain, verbose=2)
    #historytr.append([test_loss, test_acc])
    #test_loss, test_acc = model.evaluate(quantTimeTest, supportTest, verbose=2)
    #historyva.append([test_loss, test_acc])
    for i in range(opt.divideNum):
        model.fit(quantTimeTrain, supportTrain, epochs=perep, verbose=1,shuffle=1)
        test_loss, test_acc,test_mis,test_fal = model.evaluate(quantTimeTrain, supportTrain, verbose=2)
        historytr.append([test_loss, test_acc,test_mis,test_fal])
        test_loss, test_acc,test_mis,test_fal = model.evaluate(quantTimeTest, supportTest, verbose=2)
        historyva.append([test_loss, test_acc,test_mis,test_fal])
    return historyva,historytr,model
def conductTrainingCrossentropy(model,opt,quantTimeTrain,supportTrain,quantTimeTest,supportTest):
    perep = int(opt.epochs / opt.divideNum)
    historytr = []
    historyva = []
    for i in range(opt.divideNum):
        model.fit(quantTimeTrain, supportTrain, epochs=perep, verbose=1,batch_size=1024)
        #test_loss, test_acc= model.evaluate(quantTimeTrain, supportTrain, verbose=2)
        #historytr.append([test_loss, test_acc])
        #test_loss, test_acc = model.evaluate(quantTimeTest, supportTest, verbose=2)
        #historyva.append([test_loss, test_acc])
    #pred=model.predict(quantTimeTest)
    #pred=np.argmax(pred,axis=1)
    return historyva,historytr,model
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
def loadData(a,supbase,cosets=None):
    if cosets is None:
        siz=a.trainingNum
    else:
        siz=a.testingNum
    tmplst=[]
    indlst=[]
    while siz>=supbase.shape[0]:
        tmplst.append(supbase)
        indlst.append(np.array(range(supbase.shape[0])))
        siz-=supbase.shape[0]

    index=np.random.randint(0,supbase.shape[0],siz)
    tmplst.append(supbase[index,:])
    indlst.append(index)
    support=np.concatenate(tmplst,0)
    index=np.concatenate(indlst)
    sendFreq = signalGen(support, a.subBandWidth)
    sendTime = freqToTime(sendFreq)
    if cosets is None:
        cosets = generateCosets(int(sendTime.shape[1] / a.subBandWidth), a.cosetNum, a.subBandWidth)
    recTime = mulCoset(sendTime, cosets)
    quantTime, quantStair = quantify(recTime, a.quantifyLevel)
    return support, sendTime, cosets,quantTime,quantStair,index
def threshold(inp,thres):
    inp = 1 * (inp > thres)
    return inp
def errorRate(inp1,inp2):
    #all those 3 only works with binary matrices in array form
    return np.sum(1*(abs(inp1-inp2)>0))/inp2.size
def falseAlarm(inppred,inpori):
    #how much is wrong in the predicted positives
    totalNum=np.sum(inppred)
    tmp=inpori * inppred
    return 1-np.sum(tmp)/totalNum
def misdetection(inppred,inpori):
    #how much is not detected in real positives
    #this need to be kept low
    totalNum=np.sum(inpori)
    tmp=inpori * inppred
    return 1-np.sum(tmp)/totalNum
def pltFigureRand(ind,data,drawNum,trainingNum,givenindex=None):
    z=plt.figure(ind)
    if givenindex is None:
        givenindex=np.random.randint(0,trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(data[givenindex[i],:])
    return ind+1,z,givenindex
def detectionRate(inppred,inpori):
    tmp=inpori * inppred
    tmp=np.sum(tmp,axis=-1)
    tmp=tmp<np.sum(inpori[0,:])
    tmp=tmp*1
    tmp=1-tmp
    return np.sum(tmp)/tmp.size
def detectionMetric(yt,yp):
    ypt=tf.keras.backend.round(tf.keras.backend.clip(yp, 0, 1))
    tmp=yt * ypt
    tmp=tf.keras.backend.sum(tmp,axis=-1)
    tmp=tmp<tf.keras.backend.sum(yt[0,:])
    tmp=tf.keras.backend.cast(tmp,dtype=tf.int32)
    return 1-tf.keras.backend.sum(tmp)/tf.size(tmp)
def numLoss(yt,yp):
    siz=tf.keras.backend.sum(yt[0,:])
    tmp=yp
    tmp=tf.keras.backend.sum(tmp,axis=-1)
    tmp=tf.keras.backend.pow(tmp-siz,2)
    return tmp
def misdetectionMetric(yt,yp):
    ypt=tf.keras.backend.round(tf.keras.backend.clip(yp, 0, 1))
    tmp=yt * ypt
    return 1-tf.keras.backend.sum(tmp)/tf.keras.backend.sum(yt)
def falseAlarmMetric(yt,yp):
    ypt=tf.keras.backend.round(tf.keras.backend.clip(yp, 0, 1))
    tmp=yt * ypt
    return 1-tf.keras.backend.sum(tmp)/tf.keras.backend.sum(ypt)
def falseAlarmLoss(yt,yp):
    ypt=tf.keras.backend.round(tf.keras.backend.clip(yp, 0, 1))
    squared_difference=tf.keras.backend.square(yp*ypt-yt*ypt)
    return tf.reduce_mean(squared_difference, axis=-1)
def misDetectionLoss(yt,yp):
    ytt=yt
    ypt=ytt*yp
    squared_difference=tf.keras.backend.square(ypt-ytt)
    return tf.reduce_mean(squared_difference, axis=-1)

def combinedLoss(yt,yp,weightlst):
    return weightlst[0]*falseAlarmLoss(yt,yp)+weightlst[1]*misDetectionLoss(yt,yp)+weightlst[2]*numLoss(yt,yp)

def combinedLossWrap(weightlst):
    def clhelper(yt,yp):
        return combinedLoss(yt,yp,weightlst)
    return clhelper