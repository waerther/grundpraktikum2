import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

md = pd.read_csv('tables/md.csv')

# Plot 1

md1 = md.iloc[:, [0,1]]
np.savetxt('tables/a.txt', md1.values, header='U/V N', fmt='%.3f')
U, N = np.genfromtxt('tables/a.txt', unpack=True, skip_header=1)

# Plateaubereich 380 - 620 V
Up = U[5:30]
Np = N[5:30]
Np = Np/120

# Ausgleichsgerade
def g(u, a, b):
    return a * u + b

para, pcov = curve_fit(g, Up, Np)
a, b = para
pcov = sqrt(np.diag(pcov))
fa, fb = pcov 

xx = np.linspace(380, 620, 10000)   # Spannungen für das Plateau-Gebiet
fN = sqrt(N)                        # N Poisson-verteilt
uN = uarray(N, fN)
uN = uN/120                         # Impulsrate mit Fehler
plt.errorbar(U, noms(uN), yerr = stds(uN), fmt='r.', elinewidth = 1, capsize = 2, label = 'Messdaten')
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Plateaugerade')

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$N \, / \, \mathrm{s^{-1}}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 


# plot 2

md2 = md.iloc[:, [1, 2]]
np.savetxt('tables/b.txt', md2.values, header='N I/μA', fmt='%.3f')
N, I = np.genfromtxt('tables/b.txt', unpack=True, skip_header=1)
I *= 1e-6
t = 120      # Integrationszeit in s 

# einfallende Teilchen
e = const.elementary_charge
def z(i, n):
    e = const.elementary_charge
    return i/(e*n)

fI = 0.05*1e-6       # Fehler des Stroms in μA
uI = uarray(I, fI)

fN = sqrt(N)
uN = uarray(N, fN)
uN = uN/t

Z = z(uI, uN)

plt.errorbar(I, noms(Z), yerr = stds(Z), fmt='r.', elinewidth = 1, capsize = 2, label = 'Messdaten')

# Ausgleichsgerade
def g(u, a, b):
    return a * u + b

para, pcov = curve_fit(g, I, noms(Z)*1e-10)
a, b = para; print(para)
pcov = sqrt(np.diag(pcov))
fa, fb = pcov 
xx = np.linspace(0.1*1e-6, 1.3*1e-6, 10000)
a *= 1e10; b *= 1e10
fa *= 1e10; fb *= 1e10
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsgerade')

plt.xlabel(r'$I \, / \, \mu A$')
plt.ylabel(r'$Z \, / \, \mathrm{e}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(0.12*1e-6, 1.2*1e-6)
plt.ylim(0.5*1e10, 7e10)

plt.savefig('build/plot2.pdf', bbox_inches = "tight")
plt.clf() 