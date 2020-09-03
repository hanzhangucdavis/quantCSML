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
    epochs = 50 #NN training epoch
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
hla=[]
hlb=[]
xaxis=[]


#===================================
#=Script of generating training data

# addressPrefix = 'training_set/signal_'
# addressTails = '/support.csv'
# addressTailxi = '/xt.csv'
# addressTailxr = '/xr.csv'
# supportlst = np.zeros([100, opt.subBandNum])
# sendTimelsti = np.zeros([100, opt.subBandNum * opt.subBandWidth])
# sendTimelstr = np.zeros([100, opt.subBandNum * opt.subBandWidth])
# for signalNum in range(1, 11):
#     address = addressPrefix + str(signalNum) + addressTails
#     tmp = np.loadtxt(address, delimiter=',')
#     supportlst = np.concatenate([supportlst, tmp], axis=0)
#     address = addressPrefix + str(signalNum) + addressTailxi
#     tmp = np.loadtxt(address, delimiter=',')
#     sendTimelsti = np.concatenate([sendTimelsti, tmp], axis=0)
#     address = addressPrefix + str(signalNum) + addressTailxr
#     tmp = np.loadtxt(address, delimiter=',')
#     sendTimelstr = np.concatenate([sendTimelstr, tmp], axis=0)
#
# supbase = np.loadtxt('C10_avector.txt')
# basedict = {}
# for i in range(supbase.shape[0]):
#     basedict[str(supbase[i, :])] = i
#
# indexlist = np.zeros([supportlst.shape[0], 1])
# for i in range(supportlst.shape[0]):
#     indexlist[i] = basedict[str(supportlst[i])]




if opt.supflag:

    supportlst=np.loadtxt(opt.pathToSupport)
    sendTimelsti=np.loadtxt(opt.pathToTrainingi)
    sendTimelstr=np.loadtxt(opt.pathToTrainingr)
    supbase = np.loadtxt('C10_avector.txt')
    #basedict = {}
    #for i in range(supbase.shape[0]):
        #basedict[str(supbase[i, :])] = i
    indexlist=np.loadtxt(opt.pathToIndex)

    supportlstte=np.loadtxt(opt.pathToSupportte)
    sendTimelstite=np.loadtxt(opt.pathToTrainingite)
    sendTimelstrte=np.loadtxt(opt.pathToTrainingrte)
    indexlistte=np.loadtxt(opt.pathToIndexte)
    stageList=np.loadtxt(opt.pathToStageList)
    stageList=np.array([0]+list(range(100,10101,1000))) #override


for cosetNum in range(1,10):
    opt.cosetNum=cosetNum
    support=supportlst
    sendTimei=sendTimelsti
    sendTimer = sendTimelstr
    cosets = generateCosets(opt.subBandNum, opt.cosetNum, opt.subBandWidth)
    recTimei = mulCoset(sendTimei, cosets)
    recTimer = mulCoset(sendTimer, cosets)
    recTime=np.concatenate([recTimer,recTimei],axis=1)
    quantTime, quantStair = quantify(recTime, opt.quantifyLevel)
    indextr= indexlist


    support2=supportlstte
    sendTimei2=sendTimelstite
    sendTimer2 = sendTimelstrte
    recTimei2 = mulCoset(sendTimei2, cosets)
    recTimer2 = mulCoset(sendTimer2, cosets)
    recTime2=np.concatenate([recTimer2,recTimei2],axis=1)
    quantTime2, quantStair = quantify(recTime2, opt.quantifyLevel)
    indexte= indexlist





    findex=0
    if opt.showPlot:
        findex,fhandle,drwo=pltFigureRand(findex, support2, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, sendTimei, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, quantTime, opt.drawNum, opt.trainingNum)


    hl1=[]
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
        pred=model.predict(quantTime2)
        pred=np.argmax(pred,-1)
        predres=supbase[pred,:]
        realres=support2
        err=[]
        fal=[]
        mis=[]
        det=[]
        for k in range(0,11):
            ran=range(stageList[k],stageList[k+1])
            errt = errorRate(predres[ran,:],realres[ran,:])
            falt = falseAlarm(predres[ran,:], realres[ran,:])
            mist = misdetection(predres[ran,:], realres[ran,:])
            dett = detectionRate(predres[ran,:], realres[ran,:])
            err.append(errt)
            fal.append(falt)
            mis.append(mist)
            det.append(dett)
        errt = errorRate(predres, realres)
        falt = falseAlarm(predres, realres)
        mist = misdetection(predres, realres)
        dett = detectionRate(predres, realres)
        err.append(errt)
        fal.append(falt)
        mis.append(mist)
        det.append(dett)
        #need to add evaluation for different num of signals
        hl2.append([err,fal,mis,det])
        t2=time.time()
        print(t2-t1)
    hlb.append(hl2)
    hlbt=np.array(hlb)
    np.save('testresult'+time.strftime('%Y%m%d%H'),hlbt)


        #test

if opt.showPlot:
    findex,dum1,dum2=pltFigureRand(findex, support2, opt.drawNum, opt.trainingNum,drwo)
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



