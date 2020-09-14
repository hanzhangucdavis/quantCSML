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
testroot=[]

addressPrefix = 'Noise_free_testing_small/'
addressTails = '/support.csv'
addressTailxi = '/xt.csv'
addressTailxr = '/xr.csv'

for signalNum in sigNumList[3:4]:
    #opt.cosetNum=cosetNum

    stageList = [0]
    supportlst = []
    sendTimelsti = []
    sendTimelstr = []
    address = addressPrefix +  'signal_' + str(signalNum) + addressTails
    supportlst.append(np.loadtxt(address, delimiter=','))
    address = addressPrefix + 'signal_' + str(signalNum) + addressTailxi
    sendTimelsti.append(np.loadtxt(address, delimiter=','))
    address = addressPrefix + 'signal_' + str(signalNum) + addressTailxr
    sendTimelstr.append(np.loadtxt(address, delimiter=','))
    stageList.append(stageList[-1]+sendTimelsti[-1].shape[0])
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
    test=[]
    for cosetNum in range(1,10):
        opt.cosetNum=cosetNum


        hl2=[]
        test2=[]
        for trainSNR in SNRList[15:]:
            bestindex=np.loadtxt('modelSNR'+str(trainSNR)+'coset'+str(cosetNum)+'bestiter'+'.txt')
            bestindex=bestindex.astype(np.int64)
            cosets = np.loadtxt('modelSNR' + str(trainSNR) + 'coset' + str(cosetNum) + 'iter'+str(bestindex) + '.txt')
            cosets=cosets.astype(np.int64)
            recTimei = mulCoset(sendTimei, cosets)
            recTimer = mulCoset(sendTimer, cosets)
            recTime = np.concatenate([recTimer, recTimei], axis=1)
            quantTime, quantStair = quantify(recTime, opt.quantifyLevel)
            indextr = indexlist
            t1=time.time()
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64,activation='relu'),
                tf.keras.layers.Dense(supbase.shape[0])
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            pred = model.predict(quantTime[0:2,:])
            model.load_weights('modelSNR' + str(trainSNR) + 'coset' + str(cosetNum) + 'iter'+str(bestindex) + '.h5')
            pred=model.predict(quantTime)
            pred=np.argmax(pred,-1)
            predres=supbase[pred,:]
            realres=support
            err = []
            fal = []
            det = []
            test3=[]
            for testSNR in range(0, len(stageList)-1):
                ran = range(stageList[testSNR], stageList[testSNR + 1])
                errt = errorRate(predres[ran, :], realres[ran, :])
                falt = falseAlarm(predres[ran, :], realres[ran, :])
                dett = detectionRate(predres[ran, :], realres[ran, :])
                err.append(errt)
                fal.append(falt)
                det.append(dett)
                test3.append(['signalNum',signalNum,'cosetNum',cosetNum,"trainSNR",trainSNR,'testSNR',testSNR])
            errt = errorRate(predres, realres)
            falt = falseAlarm(predres, realres)
            dett = detectionRate(predres, realres)
            err.append(errt)
            fal.append(falt)
            det.append(dett)
            #need to add evaluation for different num of signals
            hl2.append([err,fal,det])
            test2.append(test3)
            t2=time.time()
            print(t2-t1)
        hlb.append(hl2)
        test.append(test2)
    hla.append(hlb)
    testroot.append(test)
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



