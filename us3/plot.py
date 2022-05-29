import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray
from uncertainties import unumpy as unp 
import scipy.constants as const

# Daten:

delta_nu = pd.read_csv('tables/winkel.csv')
np.savetxt('tables/delta_nu.txt', delta_nu.values, header='f/Hz U/V', fmt='%.3f')
P, dnu15, dnu30, dnu60 = np.genfromtxt('tables/delta_nu.txt', unpack=True, skip_header=1)

alpha = [80.06, 70.53, 54.74]   # Prismenwinkel
alpha = np.multiply(alpha, (np.pi/180)) 
c = const.speed_of_light
nu0 = 2e6   # 2 MHz

def v(nu, a):
    return (nu * c)/(2 * nu0 * np.cos(a)) /1e5

v15 = np.zeros(5)
v30 = np.zeros(5)
v60 = np.zeros(5)

for j in range(5):
    v15[j] = v(dnu15[j], alpha[0])
    v30[j] = v(dnu30[j], alpha[1])
    v60[j] = v(dnu60[j], alpha[2])

# Plot 1:

def f(v):
    return (2 * nu0 * v)/c

plt.plot(v15, f(v15), 'xr', markersize=6 , label = 'Messdaten')

def g(x, a, b):
    return a*x + b

para, pcov = curve_fit(g, v15, f(v15))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 1, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
plt.xlim(0, 1)                  
plt.ylim(0, 0.014)

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 

# Plot 2:

v30 = abs(v30)
plt.plot(v30, f(v30), 'xr', markersize=6 , label = 'Messdaten')

para, pcov = curve_fit(g, v30, f(v30))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 1.4, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
plt.xlim(0, 1.3)                  
plt.ylim(0, 0.018)

plt.savefig('build/plot2.pdf', bbox_inches = "tight")
plt.clf() 

# Plot 3:

plt.plot(v60, f(v60), 'xr', markersize=6 , label = 'Messdaten')

para, pcov = curve_fit(g, v60, f(v60))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 1.4, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
plt.xlim(0, 1.1)                  
plt.ylim(0, 0.015)

plt.savefig('build/plot3.pdf', bbox_inches = "tight")
plt.clf() 


# Daten
p75 = pd.read_csv('tables/75p.csv')
np.savetxt('tables/p75.txt', p75.values, header='x_sec/micros delta_nu/Hz I_s/V^2/s', fmt='%.3f')
x_sec, v75, I75 = np.genfromtxt('tables/p75.txt', unpack=True, skip_header=1)
x_mm = (6/4) * x_sec - 18

p45 = pd.read_csv('tables/45p.csv')
np.savetxt('tables/p45.txt', p45.values, header='x_sec/micros delta_nu/Hz I_s/V^2/s', fmt='%.3f')
x_sec, v45, I45 = np.genfromtxt('tables/p45.txt', unpack=True, skip_header=1)

# Plot 4
plt.plot(x_mm, v75, 'xr', markersize=6 , label = 'Momentangeschwindigkeit für P = 75%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v_{mom} \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)

plt.savefig('build/plot4.pdf', bbox_inches = "tight")
plt.clf() 

# Plot 5
plt.plot(x_mm, I75, 'xr', markersize=6 , label = 'Streuintensitäten für P = 75%')
plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$I_S \, / \, 1000 \, \mathrm{V^2 s^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True) 

plt.savefig('build/plot5.pdf', bbox_inches = "tight")
plt.clf() 

# Plot 6
plt.plot(x_mm, v45, 'xr', markersize=6 , label = 'Momentangeschwindigkeit für P = 45%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v_{mom} \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True) 

plt.savefig('build/plot6.pdf', bbox_inches = "tight")
plt.clf() 

# Plot 7
plt.plot(x_mm, I45, 'xr', markersize=6 , label = 'Streuintensitäten für P = 45%')
plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$I_S \, / \, 1000 \, \mathrm{V^2 s^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)  

plt.savefig('build/plot7.pdf', bbox_inches = "tight")
plt.clf() 