import numpy as np
import matplotlib.pyplot as plt
import scipy.io

z=np.load('testresultnewmetric.npy')
#test_loss, test_det,test_mis,test_fal
z=np.average(z,axis=1)
xaxis=np.array(range(4,12))
plt.figure()
plt.plot(xaxis,z[:,0,0],label='err')
plt.plot(xaxis,z[:,0,1],label='detection')
plt.plot(xaxis,z[:,0,2],label='misdetection')
plt.plot(xaxis,z[:,0,3],label='falseAlarm')
plt.legend()
plt.show()