import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions import *

class option:
    trainingNum = 1000
    subBandNum = 40
    occupyNum = 4
    freqToTimeRatio = 5
    subBandWidth = 20
    quantifyLevel = 16
    cosetNum = 8
    showPlot = True


opt=option()

support=np.loadtxt('support.csv',delimiter=',')
quantTimet1=np.loadtxt('x_real.csv',delimiter=',')
quantTimet2=np.loadtxt('x_imag.csv',delimiter=',')
cosets = generateCosets(int(quantTimet1.shape[1] / opt.subBandWidth), opt.cosetNum, opt.subBandWidth)
quantTimet1 = mulCoset(quantTimet1,cosets)
quantTimet1, quantStair = quantify(quantTimet1, opt.quantifyLevel)
quantTimet2 = mulCoset(quantTimet2, cosets)
quantTimet2, quantStair = quantify(quantTimet2,opt.quantifyLevel)





quantTime=np.concatenate([quantTimet1,quantTimet2],axis=1)


support2=np.loadtxt('support_t.csv',delimiter=',')
quantTimet3=np.loadtxt('x_real_t.csv',delimiter=',')
quantTimet4=np.loadtxt('x_imag_t.csv',delimiter=',')
quantTimet3 = mulCoset(quantTimet3,cosets)
quantTimet3, quantStair = quantify(quantTimet3, opt.quantifyLevel)
quantTimet4 = mulCoset(quantTimet4, cosets)
quantTimet4, quantStair = quantify(quantTimet4,opt.quantifyLevel)





quantTime2=np.concatenate([quantTimet3,quantTimet4],axis=1)





if opt.showPlot:
    plt.figure(1)
    drawNum=3
    drwo=np.random.randint(0,opt.trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(support2[drwo[i],:])

    plt.figure(3)
    drawNum=3
    drw=np.random.randint(0,opt.trainingNum,drawNum)
    for i in range(0,drawNum):
        plt.plot(quantTime[drw[i],:])


qt1=np.zeros_like(quantTimet1,dtype=complex)
qt1+=quantTimet1
qt1+=quantTimet2*1j
qt1mod=np.abs(qt1)
qt2=np.zeros_like(quantTimet3,dtype=complex)
qt2+=quantTimet3
qt2+=quantTimet4*1j
qt2mod=np.abs(qt2)
qt1f=np.abs(np.fft.fft(qt1))


plt.figure(2)
drawNum = 3
drw = np.random.randint(0, opt.trainingNum, drawNum)
for i in range(0, drawNum):
    plt.plot(qt1f[drwo[i], :])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(opt.subBandNum)
])


model.compile(optimizer='adam',
              loss="MSE",
              metrics=['accuracy'])

test_res= model.predict(quantTime)
test_res=threshold(test_res,0.5)
print("error Rate:",errorRate(test_res,support))
print("False Alarm:",falseAlarm(test_res,support))
print("Mis Detection:",misdetection(test_res,support))

model.fit(quantTime, support, epochs=1000,verbose=0)

test_loss, test_acc = model.evaluate(quantTime,  support, verbose=2)
test_res= model.predict(quantTime)
test_res=threshold(test_res,0.5)
print("error Rate:",errorRate(test_res,support))
print("False Alarm:",falseAlarm(test_res,support))
print("Mis Detection:",misdetection(test_res,support))




test_res= model.predict(quantTime2)
test_res=threshold(test_res,0.5)
print("error Rate:",errorRate(test_res,support2))
print("False Alarm:",falseAlarm(test_res,support2))
print("Mis Detection:",misdetection(test_res,support2))
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