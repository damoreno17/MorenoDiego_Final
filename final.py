import numpy as np 
import matplotlib.pyplot as plt
import random

#ejercicio 15
def MCMC_polynomial(filename, poly_degree=2, n_steps=50000):
    lineas = np.loadtxt(filename)
    x=lineas[:,0]
    y=lineas[:,1]
    sigma=lineas[:,0]
    n=len(x)
    N=poly_degree
    
    def polinomio(x,coeficientes):
        for i in range(N):
            resp=coeficientes[0]+(coeficientes[i]x*i)
        return resp
    #Modelo de Bayes
    def bayes(x,y,sigma,coeficientes):
        for i in range(n):
            like=(y[i]-polinomio(x[i],coeficientes))/sigma[i]
            total=-0.5*np.sum(like**2)
        return total
    def delimitacion():
        for i in range(N):
            if coeficiente[i] < 40 and coeficiente[i]> -40:
                p += 0.0
            else:
                p += np.inf
        return p
    
    #Metodo de Metropolis Hastings
    valor_inicial=np.random.random()
    mi_lista =np.empty(0)
    mi_lista =np.append(mi_lista, valor_inicial)
    
    sigma_coef=0.1
    
    for i in range(0, n_steps):
        xnuevo= random.gauss(mi_lista[i], sigma_coef)
        
        bayesCorr_antes=[bayes(x,y,sigma,mi_lista[i-1])+delimitacion(mi_lista[i-1])]
        bayesCorr_nuevo=[bayes(x,y,sigma,mi_lista[i-1]+xnuevo)+delimitacion(mi_lista[i-1]+xnuevo)]
        
        
        alpha=0.5
        parametro=min(1,np.exp(bayesCorr_nuevo-bayesCorr_antes))
        
        if alpha<parametro:
            mi_lista.append(mi_lista,xnuevo)
        else:
            mi_lista.append(mi_lista,mi_lista[i])
            
    a=mi_lista
    return a
        
        
 def MCMC_polynomial2(filename, poly_degree=2, n_steps=50000):
    
    lineas = np.loadtxt(filename)
    x=lineas[:,0]
    y=lineas[:,1]
    sigma=lineas[:,0]
    n=len(x)
    N=poly_degree
    
    def polinomio(x,coeficientes):
        print(x,coeficientes)
        for i in range(N):
            print(i)
            resp=coeficientes[0]+(coeficientes[i]x*i)
        return resp
    
    def bayes(x,y,sigma,coeficientes):
        print(coeficientes)
        for i in range(n):
            like=(y[i]-polinomio(x[i],coeficientes))/sigma[i]
            total=-0.5*np.sum(like**2)
        return total
    
    valor_inicial=np.random.random()
    mi_lista =np.empty(0)
    mi_lista =np.append(mi_lista, valor_inicial)
    sigma_coef=0.1
    
    for i in range(0, n_steps):
        xnuevo= random.gauss(mi_lista[i], sigma_coef)
        bayesCorr_antes=[bayes(x,y,sigma,mi_lista[i-1])]
        bayesCorr_nuevo=[bayes(x,y,sigma,mi_lista[i-1]+xnuevo)+delimitacion(mi_lista[i-1]+xnuevo)]
        
        alpha=0.5
        parametro=min(1,np.exp(bayesCorr_nuevo-bayesCorr_antes))

        if alpha<parametro:
            mi_lista.append(mi_lista,xnuevo)
        else:
            mi_lista.append(mi_lista,mi_lista[i])

    a = np.ones(1)
    return a
    
    algo = MCMC_polynomial2('valores.txt', poly_degree=2, n_steps=2)
    
space = np.linspace(0,100000)	
plt.plot(space, algo)
plt.title("Valor medio")
plt.xlabel('x')
plt.ylabel('probabilidad')
plt.show()
plt.savefig('sigma.png')



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

