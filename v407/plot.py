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
md1 = pd.read_csv('tables/mess1.csv')
# print(md1.to_latex(index = False, column_format= "c c c c", decimal=',')) 

md2 = pd.read_csv('tables/mess2.csv')
# print(md2.to_latex(index = False, column_format= "c c c c", decimal=','))


md1 = pd.DataFrame(md1).to_numpy()
md2 = pd.DataFrame(md2).to_numpy()

alpha1 = np.append(md1[:-2,0], md1[:,2]) * np.pi / 180
Isenk = np.append(md1[:-2,1],md1[:,3])

alpha2 = np.append(md2[:-2,0], md2[:,2]) * np.pi / 180
Ipar = np.append(md2[:-2,1],md2[:,3])



plt.plot(alpha1, Isenk * 10**3, 'xr', markersize=5 , label = 'Senkrecht', alpha=0.65)
plt.plot(alpha2, Ipar, 'x', markersize=5 , label = 'Parallel', alpha=0.65)

nmittel = 3.465

def Esenk(x):
    return ((np.sqrt(nmittel**2-(np.sin(x))**2)-np.cos(x))**4)/(nmittel**2-1)**2
def Epar(x):
    return ((nmittel**2*np.cos(x)-np.sqrt(nmittel**2-np.sqrt(nmittel**2-(np.sin(x)**2))))/((nmittel**2*np.cos(x)+np.sqrt(nmittel**2-np.sqrt(nmittel**2-(np.sin(x)**2))))))**2
x = np.linspace(0,np.pi/2,1000)
plt.plot(x, Esenk(x) * 10**2.25,label = 'Senktrecht')
plt.plot(x, Epar(x) * 10**2.25,label = 'Parallel')

plt.xlabel(r'$\alpha$')
plt.ylabel(r'$I / nA$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style

# in matplotlibrc leider (noch) nicht möglich
plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 
