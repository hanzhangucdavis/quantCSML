import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *

#======
#problems
#1. NEED false alarm and mis detection calculation
#when the signal is sparse, the NN might choose always output 0
#2. when the signal becomes LONG, the peak energy becomes extreme, thus making quantify deleating most of the signal




class option:
    trainingNum = 10000
    subBandNum = 40
    occupyNum = 4
    freqToTimeRatio = 5
    subBandWidth = 20
    quantifyLevel = 16
    cosetNum = 8
    showPlot = True




opt=option()
support, sendTime, cosets,quantTime,quantStair=genData(opt)







support2,d1,d2,quantTime2,d3=genData(opt,cosets)


if opt.showPlot:
    plt.figure(1)
    drawNum=3
    drwo=np.random.randint(0,opt.trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(support2[drwo[i],:])

    plt.figure(2)
    drawNum=3
    drw=np.random.randint(0,opt.trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(sendTime[drw[i],:])

    plt.figure(3)
    drawNum=3
    drw=np.random.randint(0,opt.trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(quantTime[drw[i],:])






model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(opt.subBandNum)
])


model.compile(optimizer='adam',
              loss=combinedLossWrap([1,1]),
              #loss="MSE",
              metrics=[detectionMetric])


# test_res= model.predict(quantTime)
# test_res=1*(test_res>0.5)
# z=np.sum(1*(abs(test_res-support)>0))
# print(z/support.size)

# test_res=np.zeros_like(support)
# z=np.sum(1*(abs(test_res-support)>0))
# print(z/support.size)
#
# test_res=np.zeros_like(support)+1
# z=np.sum(1*(abs(test_res-support)>0))
# print(z/support.size)

model.fit(quantTime, support, epochs=50,verbose=1)

test_loss, test_acc = model.evaluate(quantTime,  support, verbose=2)
test_loss, test_acc = model.evaluate(quantTime2,  support2, verbose=2)

test_res= model.predict(quantTime2)
test_res=threshold(test_res,0.5)
print("error Rate:",errorRate(test_res,support2))
print("False Alarm:",falseAlarm(test_res,support2))
print("Mis Detection:",misdetection(test_res,support2))
print("Detection:",detectionRate(test_res,support2))
if opt.showPlot:
    plt.figure(4)
    for i in range(0,drawNum):
        plt.plot(test_res[drwo[i],:])

    plt.show()

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