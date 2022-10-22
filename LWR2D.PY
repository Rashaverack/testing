import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#test plot scatter
#x = np.arange(-5,5,0.5)
#xx , yy = np.meshgrid(x,x)
#zz = xx**2 + yy**2

#fig = plt.figure()
#ax = plt.axes(projection ="3d")

#ax.scatter3D(xx, yy, zz)
#plt.show()

def pesosdata(centro,t,datos):
    centro = np.array(centro).reshape(-1,1)
    datos = datos.reshape(centro.size,-1)

    #w = np.exp(-(((centro-datos)*np.dot(np.linalg.inv(sigma),centro-datos)).sum(0))/2)
    w = np.exp(-(((centro-datos)*(centro-datos)).sum(0))/(2*t**2)) #using identity matrix
    return w

x = np.arange(-5,5,0.5)
xx , yy = np.meshgrid(x,x)
zz = xx**2 + yy**2 + np.random.normal(0,3,xx.shape)

datos = np.vstack((np.ones(xx.size),xx.ravel(), yy.ravel()))

zpred = []
for i in range(xx.size):
    centro = datos[:,i]
    w = pesosdata(centro,0.5,datos)
    theta = np.linalg.pinv(datos@(w.reshape(-1,1)*datos.T)) @ (datos@(w.reshape(-1,1)*zz.ravel().reshape(-1,1)))
    zpred.append(np.dot(centro,theta))

pred = np.array(zpred).reshape(xx.shape)


#zz = xx**2 + yy**2

fig = plt.figure()
ax = plt.axes(projection ="3d")

ax.scatter3D(xx, yy, pred)
plt.show()