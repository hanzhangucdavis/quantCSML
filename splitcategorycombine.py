import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *
import time
#from itertools import combinations #method for generating combination




class option:
    trainingNum = 102400
    testingNum = 10000
    subBandNum = 40
    occupyNum = 4
    #freqToTimeRatio = 5
    subBandWidth = 20
    quantifyLevel = 16
    cosetNum = 8
    showPlot = False
    epochs = 10 #NN training epoch
    drawNum=3  #draw random curves of generated data
    divideNum=1 #how many times does verification happens
    repeatNum=1 #number of nn repetation, more rounds means less variation.
    pathToData='C40_4vector.txt'
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


for j in range(1):
    rat = [0.5,0.5]
    xaxis.append(0.5)
    if opt.supflag:
        support, sendTime, cosets, quantTime, quantStair,indextr=loadData(opt,supbase)
        support2, d1, d2, quantTime2, d3,indexte = loadData(opt,supbase, cosets)
    else:
        support, sendTime, cosets, quantTime, quantStair = genData(opt)
        support2, d1, d2, quantTime2, d3 = genData(opt, cosets)



    findex=0
    if opt.showPlot:
        findex,fhandle,drwo=pltFigureRand(findex, support2, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, sendTime, opt.drawNum, opt.trainingNum)
        findex,dum1,dum2=pltFigureRand(findex, quantTime, opt.drawNum, opt.trainingNum)


    hl1=[]
    hl2=[]
    for i in range(opt.repeatNum):
        t1=time.time()
        predlist=[]
        for k in range(opt.subBandNum):
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64,activation='relu'),
                tf.keras.layers.Dense(2,activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        # loss="MSE",
                          metrics=['accuracy'])
            historyva,historytr,model=conductTrainingCrossentropy(model,opt,quantTime,support[:,k],quantTime2,support2[:,k])
            pred=model.predict(quantTime2)
            #pred=np.argmax(pred,-1)
            predlist.append(pred)
        predres=np.stack(predlist,axis=-1)
        predres=np.argpartition(predres[:,1,:].squeeze(),-opt.occupyNum)
        predres=predres[:,-4:]
        dum=np.zeros_like(support2)
        for t in range(support2.shape[0]):
            dum[t,predres[t,:]]+=1
        predres=dum
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



