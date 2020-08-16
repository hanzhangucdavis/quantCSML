import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *
import time
#from itertools import combinations #method for generating combination




class option:
    trainingNum = 1024
    testingNum = 1000
    subBandNum = 10
    occupyNum = 2
    #freqToTimeRatio = 5
    subBandWidth = 3
    quantifyLevel = 0
    cosetNum = 1
    showPlot = False
    epochs = 80 #NN training epoch
    drawNum=3  #draw random curves of generated data
    divideNum=1 #how many times does verification happens
    repeatNum=1 #number of nn repetation, more rounds means less variation.
    pathToData='C10_2vector.txt'
    pathToTrainingi='x_i.csv'
    pathToTrainingr='x_r.csv'
    pathToSupport='support.csv'
    supflag = True
    batch_size=128

opt=option()

if tf.test.is_gpu_available():
    opt.batch_size=10240
hla=[]
hlb=[]
xaxis=[]

if opt.supflag:
    supbase=genTrainingList(opt)
    basedict = {}
    for i in range(supbase.shape[0]):
        basedict[str(supbase[i, :])] = i
    supportlst=np.loadtxt(opt.pathToSupport,delimiter=',')
    sendTimelsti=np.loadtxt(opt.pathToTrainingi,delimiter=',')
    sendTimelstr=np.loadtxt(opt.pathToTrainingr,delimiter=',')
    indexlist=np.zeros([supportlst.shape[0],1])
    for i in range(supportlst.shape[0]):
        indexlist[i]=basedict[str(supportlst[i])]
for j in range(1):
    xaxis.append(0.5)

    tmp=np.random.randint(0,sendTimelsti.shape[0],opt.trainingNum)
    support=supportlst[tmp,:]
    sendTimei=sendTimelsti[tmp,:]
    sendTimer = sendTimelstr[tmp, :]
    cosets = generateCosets(int(sendTimei.shape[1] / 3), opt.cosetNum, 3)
    recTimei = mulCoset(sendTimei, cosets)
    recTimer = mulCoset(sendTimer, cosets)
    recTime=np.concatenate([recTimer,recTimei],axis=1)
    quantTime, quantStair = quantify(recTime, opt.quantifyLevel)
    indextr= indexlist[tmp]


    tmp=np.random.randint(0,sendTimelsti.shape[0],opt.testingNum)
    support2=supportlst[tmp,:]
    sendTimei2=sendTimelsti[tmp,:]
    sendTimer2 = sendTimelstr[tmp, :]
    recTimei2 = mulCoset(sendTimei2, cosets)
    recTimer2 = mulCoset(sendTimer2, cosets)
    recTime2=np.concatenate([recTimer2,recTimei2],axis=1)
    quantTime2, quantStair = quantify(recTime2, opt.quantifyLevel)
    indexte= indexlist[tmp]





    findex=0
    if opt.showPlot:
        findex,fhandle,drwo=pltFigureRand(findex, support2, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, sendTime, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, quantTime, opt.drawNum, opt.trainingNum)


    hl1=[]
    hl2=[]
    for i in range(opt.repeatNum):
        t1=time.time()
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(91930)
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
        err=errorRate(predres,realres)
        fal = falseAlarm(predres, realres)
        mis = misdetection(predres, realres)
        det = detectionRate(predres, realres)
        hl1.append(historytr)
        hl2.append([err,fal,mis,det])
        t2=time.time()
        print(t2-t1)
    hla.append(hl1)
    hlb.append(hl2)
    hlat=np.array(hla)
    hlbt=np.array(hlb)
    np.save('trainresult'+time.strftime('%Y%m%d%H'),hlat)
    np.save('testresult'+time.strftime('%Y%m%d%H'),hlbt)

if opt.showPlot:
    findex,dum1,dum2=pltFigureRand(findex, support2, opt.drawNum, opt.trainingNum,drwo)
print(hlat)
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



