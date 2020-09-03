import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *
import time
#from itertools import combinations #method for generating combination




class option:
    trainingNum = 102400
    testingNum = 1000
    subBandNum = 10
    occupyNum = 2
    #freqToTimeRatio = 5
    subBandWidth = 20
    quantifyLevel = -1
    cosetNum = 5
    showPlot = False
    epochs = 5 #NN training epoch
    drawNum=3  #draw random curves of generated data
    divideNum=1 #how many times does verification happens
    repeatNum=1 #number of nn repetation, more rounds means less variation.
    pathToData='C10_2vector.txt'
    pathToTrainingi='x_i.csv'
    pathToTrainingr='x_r.csv'
    pathToSupport='support.csv'
    pathToIndex='indexlist.txt'
    pathToTrainingite='x_ite.csv'
    pathToTrainingrte='x_rte.csv'
    pathToSupportte='supportte.csv'
    pathToIndexte='indexlistte.csv'
    pathToStageList='stagelist.csv'
    supflag = True
    batch_size=128

opt=option()

if tf.test.is_gpu_available():
    opt.batch_size=10240

xaxis=[]

SNRList=[-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
sigNumList=[1,2,3,4,5,6,7,8,9,10]

parameterPack=np.meshgrid([1,2,3],[SNRList])
hla=[]
for SNR in SNRList:
    #opt.cosetNum=cosetNum

    addressPrefix = 'trainingnoisy/snr'
    addressTails = '/support.csv'
    addressTailxi = '/xt.csv'
    addressTailxr = '/xr.csv'
    supportlst = []
    sendTimelsti = []
    sendTimelstr = []
    for signalNum in sigNumList:
        address = addressPrefix + str(SNR) + 'dB/signal_' + str(signalNum) + addressTails
        supportlst.append(np.loadtxt(address, delimiter=','))
        address = addressPrefix + str(SNR) + 'dB/signal_' + str(signalNum) + addressTailxi
        sendTimelsti.append(np.loadtxt(address, delimiter=','))
        address = addressPrefix + str(SNR) + 'dB/signal_' + str(signalNum) + addressTailxr
        sendTimelstr.append(np.loadtxt(address, delimiter=','))
    sendTimelstr = np.concatenate(sendTimelstr, axis=0)
    sendTimelsti = np.concatenate(sendTimelsti, axis=0)
    supportlst = np.concatenate(supportlst, axis=0)
    supbase = np.loadtxt('C10_avector.txt')
    basedict = {}
    for i in range(supbase.shape[0]):
        basedict[str(supbase[i, :])] = i
    indexlist = np.zeros([supportlst.shape[0], 1])
    for i in range(supportlst.shape[0]):
        indexlist[i] = basedict[str(supportlst[i])]
    support=supportlst
    sendTimei=sendTimelsti
    sendTimer = sendTimelstr
    hlb=[]
    for cosetNum in range(1,10):
        opt.cosetNum=cosetNum
        cosets = generateCosets(opt.subBandNum, opt.cosetNum, opt.subBandWidth)
        recTimei = mulCoset(sendTimei, cosets)
        recTimer = mulCoset(sendTimer, cosets)
        recTime=np.concatenate([recTimer,recTimei],axis=1)
        quantTime, quantStair = quantify(recTime, opt.quantifyLevel)
        indextr=indexlist
        recTime2=recTime
        quantTime2, quantStair = quantify(recTime2, opt.quantifyLevel)
        indexte= indexlist





        hl2=[]
        for i in range(opt.repeatNum):
            t1=time.time()
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64,activation='relu'),
                tf.keras.layers.Dense(supbase.shape[0])
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        # loss="MSE",
                          metrics=['accuracy'])
            historyva,historytr,model=conductTrainingCrossentropy(model,opt,quantTime,indextr,quantTime2,indexte)
            model.save_weights('modelSNR'+str(SNR)+'coset'+str(cosetNum)+'iter'+str(i)+'.h5')
            pred=model.predict(quantTime2)
            pred=np.argmax(pred,-1)
            predres=supbase[pred,:]
            realres=support
            err = errorRate(predres, realres)
            fal = falseAlarm(predres, realres)
            mis = misdetection(predres, realres)
            det = detectionRate(predres, realres)
            #need to add evaluation for different num of signals
            hl2.append([err,fal,mis,det])
            t2=time.time()
            print(t2-t1)
        hlb.append(hl2)
    hla.append(hlb)
    hlbt = np.array(hla)
    np.save('testresult'+time.strftime('%Y%m%d%H'),hlbt)



print(hlbt)






plt.show()



# test_res= model.predict(quantTime2)
# test_res=threshold(test_res,0.5)
# print("error Rate:",errorRate(test_res,support2))
# print("False Alarm:",falseAlarm(test_res,support2))
# print("Mis Detection:",misdetection(test_res,support2))
# print("Detection:",detectionRate(test_res,support2))


#========
#Generate support
#========
#Generate frequency domain signaltest_res= model.predict(quantTime)
#========
#IFFT to time domain
#========
#Sample
#========
#Quantify



