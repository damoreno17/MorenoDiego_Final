import numpy as np 
import matplotlib.pyplot as plt

#ejercicio 15
N = 10
data = np.loadtxt('valores.txt')


def prob (sigma, data):
    return ((1/(sigma*np.sqrt(2*np.pi)))np.exp((-1/2)*(data[i]/sigma)**2))

#for i in range (100000): 
#    prob ()
    
#plt.plot()
#plt.title("Valor medio")
#plt.xlabel('x')
#plt.ylabel('probabilidad')
#plt.show()
#plt.savefig('sigma.png')



#ejercicio 16

sol = np.genfromtxt('monthrg.dat')
print (sol)

ano = sol[:,0]
spot = sol [:,3]
print (ano)

def FT(spot):
    
    X = np.zeros(len(spot), dtype=complex)

    for k in range(len(spot)):
        X[k] = 0.0j
        for n in range(len(spot)):
            X[k] += spot[n] * np.exp(-2.0 * np.pi * 1.0j / N ) ** (k * n) 
    return X

plt.plot(ano, FT(spot))
plt.title("periodo vs manchas solares")
plt.xlabel('años')
plt.ylabel('FT')
plt.show()
plt.savefig('solar.png')

#ejercicio 17

