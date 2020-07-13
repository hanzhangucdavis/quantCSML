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
    subBandNum = 40
    occupyNum = 4
    freqToTimeRatio = 5
    subBandWidth = 3
    quantifyLevel = 16
    cosetNum = 8
    showPlot = False


err=[]
fal=[]
mis=[]
xaxis=[]
opt=option()
for i in range(11):
    rat=[0.1*i,1-0.1*i]
    xaxis.append(0.1*i)
    print(xaxis[-1])
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

    errt = []
    mist = []
    falt = []
    for j in range(10):

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(opt.subBandNum)
        ])


        def falseAlarmLoss(yt, yp):
            ypt = tf.keras.backend.round(tf.keras.backend.clip(yp, 0, 1))
            squared_difference = tf.keras.backend.square(yp * ypt - yt * ypt)
            return tf.reduce_mean(squared_difference, axis=-1)


        def misDetectionLoss(yt, yp):
            ytt = yt
            ypt = ytt * yp
            squared_difference = tf.keras.backend.square(ypt - ytt)
            return tf.reduce_mean(squared_difference, axis=-1)


        def combinedLoss(yt, yp, weightlst):
            return weightlst[0] * falseAlarmLoss(yt, yp) + weightlst[1] * misDetectionLoss(yt, yp)


        def combinedLossWrap(weightlst):
            def clhelper(yt, yp):
                return combinedLoss(yt, yp, weightlst)

            return clhelper


        model.compile(optimizer='adam',
                      loss=combinedLossWrap(rat),
                      # loss="MSE",
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

        model.fit(quantTime, support, epochs=50, verbose=1)

        test_loss, test_acc = model.evaluate(quantTime, support, verbose=2)

        test_res = model.predict(quantTime2)
        test_res = threshold(test_res, 0.5)
        print("error Rate:", errorRate(test_res, support2))
        print("False Alarm:", falseAlarm(test_res, support2))
        print("Mis Detection:", misdetection(test_res, support2))
        if opt.showPlot:
            plt.figure(4)
            for i in range(0, drawNum):
                plt.plot(test_res[drwo[i], :])

            plt.show()
        errt.append(errorRate(test_res,support2))
        falt.append(falseAlarm(test_res,support2))
        mist.append(misdetection(test_res,support2))
    err.append(np.mean(errt))
    mis.append(np.mean(mist))
    fal.append(np.mean(falt))
    if opt.quantifyLevel<2:
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
print()
l1=plt.plot(xaxis,err,label='err')
l2=plt.plot(xaxis,fal,label='fal')
l3=plt.plot(xaxis,mis,label='mis')
plt.legend()
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