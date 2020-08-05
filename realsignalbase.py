import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *
import time





class option:
    trainingNum = 102400
    subBandNum = 40
    occupyNum = 4
    #freqToTimeRatio = 5
    subBandWidth = 5
    quantifyLevel = 0
    cosetNum = 8
    showPlot = False
    epochs = 20 #NN training epoch
    drawNum=3  #draw random curves of generated data
    divideNum=1 #how many times does verification happens
    repeatNum=1 #number of nn repetation, more rounds means less variation.
    batch_size=1024
opt=option()
hla=[]
hlb=[]
xaxis=[]









for j in range(3):
    #xaxis.append(0.5)


    support, sendTime, cosets,quantTime,quantStair=genData(opt)
    support2,d1,d2,quantTime2,d3=genData(opt,cosets)


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
            tf.keras.layers.Dense(opt.subBandNum)
        ])

        model.compile(optimizer='adam',
                      loss=combinedLossWrap([1,1,0]),
                      #loss="MSE",
                      metrics=[detectionMetric,misdetectionMetric,falseAlarmMetric])
        historyva,historytr,model=conductTraining(model,opt,quantTime,support,quantTime2,support2)
        hl1.append(historytr)
        hl2.append(historyva)
        t2=time.time()
        print(t2-t1)
    hla.append(hl1)
    hlb.append(hl2)
    hlat=np.array(hla)
    hlbt=np.array(hlb)
    np.save('trainresult'+time.strftime('%Y%m%d%H'),hlat)
    np.save('testresult'+time.strftime('%Y%m%d%H'),hlbt)









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



