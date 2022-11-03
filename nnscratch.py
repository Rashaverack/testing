import numpy as np
from sklearn import datasets
from scipy.special import expit
import matplotlib.pyplot as plt

D = datasets.load_iris()
X = D['data']
y = D['target']

#matriz de diseno 
X_d = np.c_[np.ones(X.shape[0]),X[:,0],X[:,2]]

neur_capa1 = 5
columasW1 = X_d.shape[1]
#inicializar la matriz W1
W1 = np.random.normal(size=(neur_capa1,columasW1))


for i in range(50000):

    #Salida de mi primera capa totalmente conectada,cada dato en cada fila
    K1_fully = X_d@W1.T
    #salida de capa de activacion
    Y1 = expit(K1_fully)

    #segunda capa
    #extender salida de la capa anterior con sesgo
    X2 = np.c_[np.ones(Y1.shape[0]), Y1] #salida de mi primera capa es la entrada de mi segunda capa

    if i == 0:
        columasW2 = X2.shape[1]
        salida_k2 = 3 #numero de clases
        #inicializar la matriz
        W2 = np.random.normal(size=(salida_k2,columasW2))

    #salida de mi segunda capa totalmente conectada
    K2_fully = X2@W2.T
    #salida de capa de activacion usando softmax cada dato en cada fila de tamano 3 (clases)
    Y2 = np.exp(K2_fully)
    suma = Y2.sum(1)
    Y2 = Y2/suma.reshape(-1,1)

    #error
    Y_onehot = np.eye(3)
    labels = Y_onehot[:,y].T
    #Error usando todos los datos
    J = np.power((Y2 - labels), 2).sum() / labels.shape[0]

    #RRETROPROPAGACION
    #grad de error(J) wrt Y predicho
    g_wrt_Y2 = 2*(Y2 - labels)/labels.shape[0]
    #grad de error J wrt entrada entrada softmax
    g_wrt_K2fully = Y2 * g_wrt_Y2 - (np.diag((Y2 * g_wrt_Y2) @ np.ones(Y2.shape[1]))) @ Y2
    #grad de error J wrt entradas segunda capa X2
    g_wrt_X2 = g_wrt_K2fully @ W2[:,1:]
    #print(g_wrt_X2.shape,X2.shape)
    #grad de error J wrt pesos segunda capa W2
    g_wrt_W2 = X2.T @ g_wrt_K2fully
    #print(g_wrt_W2.shape,W2.shape)

    #grad del la salida J wrt entrada K1_fully
    g_wrt_K1fully = g_wrt_X2 * (Y1*(1-Y1)) 
    #print(g_wrt_K1fully.shape)
    #grad de error J wrt pesos primera capa W1
    g_wrt_W1 = X_d.T @ g_wrt_K1fully
    #print(g_wrt_W1.shape, W1.shape)
    W2 = W2 - 0.001*g_wrt_W2.T
    W1 = W1 - 0.001*g_wrt_W1.T


XX,YY = np.meshgrid(np.arange(2,10,0.1),np.arange(0,10,0.1))
entr = np.c_[np.ones(XX.size),XX.ravel(),YY.ravel()]
#Salida de mi primera capa totalmente conectada,cada dato en cada fila
K1_fully = entr@W1.T
#salida de capa de activacion
Y1 = expit(K1_fully)

#segunda capa
#extender salida de la capa anterior con sesgo
X2 = np.c_[np.ones(Y1.shape[0]), Y1] #salida de mi primera capa es la entrada de mi segunda capa

#salida de mi segunda capa totalmente conectada
K2_fully = X2@W2.T
#salida de capa de activacion usando softmax cada dato en cada fila de tamano 3 (clases)
Y2 = np.exp(K2_fully)
suma = Y2.sum(1)
Y2 = Y2/suma.reshape(-1,1)

clases_ganadoras = Y2.argmax(1).reshape(XX.shape)

fig, ax = plt.subplots()
ax.pcolormesh(XX,YY,clases_ganadoras)
ax.scatter(X[y==0,0],X[y==0,2])
ax.scatter(X[y==1,0],X[y==1,2])
ax.scatter(X[y==2,0],X[y==2,2])
plt.show()