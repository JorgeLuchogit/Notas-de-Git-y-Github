import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint         
# odeint es un solver umérico para ecuaciones diferenciales


"""------------------------------------------------------
Programa para el análisis de un sistema de ecuaciones 
diferenciales del modelo Kermack-McKendrick (SIR). El 
programa entrega una gráfica de las soluciones s(t),  
I(t) y R(t), con la opción de computar la población 
inicial susceptible, así como la población inicial in-
fectada. También genera una gráfica del plano S-I de las 
obtenidas al variar los parámetros alfa y beta.
Se muestra las solución estacionaria 
------------------------------------------------------"""

def SIR(Y, t, N, beta, alfa):

    """--------------------------------------------------
    Función para calcular las derivadas del modelo SIR.
    Se toma la notación dS/dt=dS, dI/dt=dI y dR/dt=dR.
    --------------------------------------------------"""

    S,I,R = Y
    dS = -beta*S*I/N
    dI = beta*S*I/N-alfa*I
    dR = alfa*I

    return dS, dI, dR

# Condiciones iniciales:
N = 47000000
I_0 = 10000
R_0 = 0
S_0 = N-I_0-R_0
alfa = 0.02
beta = 0.1
Y_0 = S_0, I_0, R_0 

t = np.linspace(0, 365, 365)

# Soluciones del sistema:
s1 = odeint(SIR, Y_0, t, args=(N, beta, alfa))

# Gráfica de las soluciones:  
plt.figure()
plt.title('Soluciones del Modelo Kermack-McKendrick \n(SIR)', size=15)
for i in range (3):
    plt.plot(s1[:,i]/N, label='S(t)')
plt.ylabel('Población normalizada', size=13)
plt.xlabel('Tiempo [s]', size=13)
plt.legend()

# Gráfica del plano S-I:
alfa = np.arange(0.02, 0.1, 0.04)
beta = np.arange(0.1, 1, 0.2)

plt.figure()
for a in alfa:
    for b in beta:
        s2 = odeint(SIR, Y_0, t, args=(N, b, a))
        plt.plot(s2[:,0]/N, s2[:,1]/N, label = '$\\alpha=$%.2f y $\\beta=$%.2f' %(a,b))
plt.title('Plano S-I', size=20)
plt.ylabel('Población suceptible normalizada', size=15)
plt.xlabel('Población infectada  normalizada', size=15)
plt.legend()



# Se hace un array (.T sirve para sacar una traspuesta)
S,I,R = s1.T   

alfa = alfa[0]
beta = beta [0]

# Gráfica de la solución estacionaria I vs. N
plt.figure()
X = np.linspace(0,N,1000)
Z = -X**2+(N-1)*X-alfa/beta+N
plt.plot(X, Z/np.max(Z)*100, label = 'I(N)')
plt.title('$\dot I=\\alpha I(N-1)-\\frac{\\beta I}{1+I}$ \n Solución estacionaria', size=20)
plt.ylabel('Población infectada (%)', size=15)
plt.xlabel('Población total', size=15)
plt.legend()
plt.show()