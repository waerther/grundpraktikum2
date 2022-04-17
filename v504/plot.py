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

md1 = pd.read_csv('tables/md1.csv')
np.savetxt('tables/md1.txt', md1.values, fmt='%.3f')
U, k1, k2, k3, k4, k5 = np.genfromtxt('tables/md1.txt', unpack=True)

# plot 1:
plt.plot(U, k1, 'xr', label = "Kennlinie 1", zorder=2)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)

plt.savefig('build/plot_k1.pdf', bbox_inches = "tight")
plt.clf() 

#plot 2:
plt.plot(U, k2, 'xr', label = "Kennlinie 2")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)

plt.savefig('build/plot_k2.pdf', bbox_inches = "tight")
plt.clf() 

# plot 3:
plt.plot(U, k3, 'xr', label = "Kennlinie 3")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)  

plt.savefig('build/plot_k3.pdf', bbox_inches = "tight")
plt.clf() 

# plot 4:
plt.plot(U, k4, 'xr', label = "Kennlinie 4")

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)  

plt.savefig('build/plot_k4.pdf', bbox_inches = "tight")
plt.clf() 

# plot 5:
plt.plot(U, k5, 'xr', label = "Kennlinie 5", zorder=2, alpha = 0.7)

def lsr(u, a, b):
    return (4/9) * const.epsilon_0 * np.sqrt(2 * const.elementary_charge/ const.electron_mass) * u**(b)/a**2

para, pcov = curve_fit(lsr, U[:15], k5[:15])
a, b = para
fa, fb = np.sqrt(np.diag(pcov))

ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 280, 100)
plt.plot(xx, lsr(xx, noms(ua), noms(ub)), '-b', label = "Ausgleichsfunktion", linewidth = 1, zorder=1)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)
plt.xlim(-10, 280) 
plt.ylim(-0.1, 2.3) 

plt.savefig('build/plot_k5.pdf', bbox_inches = "tight")
plt.clf() 

# plot 6: Aufgabe d)
md2 = pd.read_csv('tables/md2.csv')
np.savetxt('tables/md2.txt', md2.values, fmt='%.4f')
U, I = np.genfromtxt('tables/md2.txt', unpack=True)

plt.plot(U, I, 'xr', label = "Messdaten",  alpha = 0.7)

def f(u, a, b):
    return a * np.exp(-u / b)

para, pcov = curve_fit(f, U, I)
a, b = para
fa, fb = np.sqrt(np.diag(pcov))

xx = np.linspace(-0.5, 1.5, 100)
plt.plot(xx, f(xx, a, b), '-b', label = "Ausgleichsfunktion", linewidth = 1)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{nA}$')
plt.legend(loc="best")     
plt.grid(True)
plt.xlim(0, 1.05)
plt.ylim(-0.5, 9)

plt.savefig('build/plot6.pdf', bbox_inches = "tight")
plt.clf()

# Tabelle für Temperaturberechnung
md3 = pd.read_csv('tables/md3.csv')
np.savetxt('tables/md3.txt', md3.values, fmt='%.1f')