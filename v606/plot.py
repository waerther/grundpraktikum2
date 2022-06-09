import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

# Plot 1:

Ua = pd.read_csv('tables/spannung.csv')
np.savetxt('tables/Ua.txt', Ua.values, header='f/Hz U/V', fmt='%.3f')
nu, U = np.genfromtxt('tables/Ua.txt', unpack=True, skip_header=1)
U = U/(2*0.9892978920131985)

plt.plot(nu, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)

# für den initial guess bei curvefit()
n = len(nu)                              #the number of data
mean = sum(nu*U)/n                       #note this correction
# sigma = sum(U*(nu - mean)**2)/n        #note this correction
sigma = np.sqrt(sum(U*(nu - mean)**2))

# Ausgleichsrechung nach Gaußverteilung
def g(x,a,x0,b):
    return a*np.exp(-(x-x0)**2/(b)) # b = 2*sigma**2

para, pcov = curve_fit(g, nu, U, p0=[1,mean,sigma])
a, nu0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fnu0, fb = pcov

# Fehler der Parameter
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(nu0, fnu0)

xx = np.linspace(18, 42, 10**4)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

# \nu_+- für U_A/U_E = 1/sqrt(2)
nu_m = nu0 - np.sqrt(-b*np.log(1/(a*np.sqrt(2))))
fnu_m = fnu0 #- np.sqrt(-fb*np.log(1/(fa*np.sqrt(2))))
unu_m = ufloat(nu_m, fnu_m)

nu_p = nu0 + np.sqrt(-b*np.log(1/(a*np.sqrt(2))))
fnu_p = fnu0
unu_p = ufloat(nu_p, fnu_p)
print(unu_m, unu_p)


plt.vlines(nu_m, -1, g(nu_m, *para), linestyle='dotted', colors='black')
plt.vlines(nu_p, -1, g(nu_p, *para), linestyle='dotted', colors='black')
plt.hlines(g(nu_m, *para), 0, nu_p,  linestyle='dotted', colors='black')

plt.vlines(nu0, -1, g(nu0, *para), linestyle='dotted', colors='black')
plt.hlines(g(nu0, *para), 0, nu0,  linestyle='dotted', colors='black')

# x = [nu_m, nu0, nu_p]
# x_labels = [r'$\nu_{-}$', r'$\nu_{0}$', r'$\nu_{+}$']
# plt.xticks(ticks=x, labels=x_labels)
# plt.xlabels()

# plt.axvline(x=nu_m, color='gray', linestyle='--')
# plt.axhline(y=g(nu_m, *para), color='gray', linestyle='--')

# plt.annotate(r'$\frac{1}{\sqrt{2}}$', xy=(35, 0.7), xycoords = 'data', textcoords = 'offset pixels')

plt.xlabel(r'$\nu \, / \, \mathrm{kHz}$')
plt.ylabel(r'$U_A \, / \, U_E$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(22, 40)
plt.ylim(-0.05, 1.05)

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 
