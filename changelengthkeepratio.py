import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *

#======
#problems
#1. NEED false alarm and mis detection calculation
#when the signal is sparse, the NN might choose always output 0
#2. when the signal becomes LONG, the peak energy becomes extreme, thus making quantify deleating most of the signal

#testing keeping the occupy ratio steady



class option:
    trainingNum = 1000
    subBandNum = 10
    occupyNum = 2
    freqToTimeRatio = 5
    subBandWidth = 3
    quantifyLevel = 16
    cosetNum = 4
    showPlot = False


err=[]
fal=[]
mis=[]
xaxis=[]
for i in range(0,100):
    opt=option()
    opt.occupyNum+=1*i
    opt.subBandNum+=5*i
    opt.cosetNum+=2*i
    xaxis.append(opt.occupyNum)
    print(opt.occupyNum,opt.subBandNum,opt.cosetNum)
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
                  loss="MSE",
                  metrics=['accuracy'])


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

    model.fit(quantTime, support, epochs=50,verbose=0)

    test_loss, test_acc = model.evaluate(quantTime,  support, verbose=2)


    test_res= model.predict(quantTime2)
    test_res=threshold(test_res,0.5)
    err.append(errorRate(test_res,support2))
    fal.append(falseAlarm(test_res,support2))
    mis.append(misdetection(test_res,support2))
    print("error Rate:",err[-1])
    print("False Alarm:",fal[-1])
    print("Mis Detection:",mis[-1])
    if mis[-1]>0.3:
        break
    if opt.showPlot:
        plt.figure(4)
        for i in range(0,drawNum):
            plt.plot(test_res[drwo[i],:])

        plt.show()

plt.figure()
print(err)
print(fal)
print(mis)
l1=plt.plot(xaxis,err,label='err')
l2=plt.plot(xaxis,fal,label='fal')
l3=plt.plot(xaxis,mis,label='mis')
plt.figlegend()
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