import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def pesosall(centro,t,DATOS):
    centro = np.array([0,0]).reshape(-1,1)
    sigma = np.eye(np.size(centro))*t**2
    x = np.arange(-5,5,0.25)
    y = np.arange(-5,5,0.25)
    xx,yy = np.meshgrid(x,y)

    datos = np.vstack((xx.ravel(),yy.ravel()))

    #w = np.exp(-(((centro-datos)*np.dot(np.linalg.inv(sigma),centro-datos)).sum(0))/2)
    w = np.exp(-(((centro-datos)*(centro-datos)).sum(0))/(2*t**2)) #using identity matrix

    z = w.reshape(np.shape(xx))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def pesosdata(centro,t,datos):
    centro = np.array(centro).reshape(-1,1)
    #sigma = np.eye(np.size(centro))*t**2
    #x = np.arange(-5,5,0.25)
    #y = np.arange(-5,5,0.25)
    #xx,yy = np.meshgrid(x,y)

    datos = datos.reshape(centro.size,-1)

    #w = np.exp(-(((centro-datos)*np.dot(np.linalg.inv(sigma),centro-datos)).sum(0))/2)
    w = np.exp(-(((centro-datos)*(centro-datos)).sum(0))/(2*t**2)) #using identity matrix
    return w

x = np.arange(0,2*np.pi,0.1)    #datos
y = np.sin(x) + np.random.normal(0,0.2,x.size)  #etiquetas
X = np.vstack((np.ones(x.size),x)) #matriz de diseno 2xdatos

#w = pesosdata([1,1.7],1,X)
#theta = np.dot(np.linalg.pinv(np.dot(X,w.reshape(-1,1)*X.T)) , (np.dot(X,w.reshape(-1,1)*y.reshape(-1,1))))

ypred = []
for i in range(x.size):
    dw = pesosdata(X[:,i],0.25,X)
    theta = np.dot(np.linalg.pinv(np.dot(X,dw.reshape(-1,1)*X.T)) , (np.dot(X,dw.reshape(-1,1)*y.reshape(-1,1))))
    pred = np.dot(X[:,i],theta)[0]
    ypred.append(pred)

#print(ypred)
plt.plot(x,ypred,x,y)
plt.show()