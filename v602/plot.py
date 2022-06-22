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

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.signal import peak_widths

# Bragg Bedingung

data = pd.read_csv('tables/messdaten/Bragg/bragg_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

plt.plot(
    x, y , color="b", ms=4, marker="x", linestyle="", label="Messwerte",
)
plt.plot(
    x[x == 28],
    y[x == 28],
    color="red",
    ms=8,
    marker="o",
    linestyle="",
    label="Maximum",
)
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle="dashed")
plt.savefig('build/bragg.pdf', bbox_inches = "tight")
plt.clf() 

# Kupferemission

# Grundlegende Aufnahme

data = pd.read_csv('tables/messdaten/Messung_2/Kupferemission1_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

plt.plot(
    x, y , color="b", ms=4, marker="x", linestyle="", label="Messwerte",
)
N_loc = find_peaks(y, height=1000)
peak_loc = N_loc[0]
plt.plot(
    x[peak_loc],
    y[peak_loc],
    color="red",
    marker="o",
    linestyle="",
    label="Peaks",
)
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle="dashed")
plt.savefig('build/kupfer1.pdf', bbox_inches = "tight")
plt.clf()

# Detailaufnahme

data = pd.read_csv('tables/messdaten/Messung_2/Kupferemission2_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

N_loc = find_peaks(y, height=1000)
peak_loc = N_loc[0]

spline = UnivariateSpline(
    x[peak_loc[0] + 5 : peak_loc[1] + 15],
    y[peak_loc[0] + 5 : peak_loc[1] + 15]
    - np.max(y[peak_loc[0] - 15 : peak_loc[1] + 15]) / 2,
    s=0,
)
r1a, r2a = spline.roots()
# print(r1a, r2a)
spline = UnivariateSpline(
    x[peak_loc[0] - 15 : peak_loc[1] - 5],
    y[peak_loc[0] - 15 : peak_loc[1] - 5]
    - np.max(y[peak_loc[0] - 15 : peak_loc[1] - 5]) / 2,
    s=0,
)
r1b, r2b = spline.roots()
plt.plot(
    [r1a, r2a],
    [N_loc[1]["peak_heights"][1] / 2, N_loc[1]["peak_heights"][1] / 2],
    color='k',
)
plt.plot(
    [r1b, r2b],
    [N_loc[1]["peak_heights"][0] / 2, N_loc[1]["peak_heights"][0] / 2],
    color='k',
)


plt.plot(
    x, y , color="b", ms=4, marker="x", linestyle="", label="Messwerte",
)
N_loc = find_peaks(y, height=1000)
peak_loc = N_loc[0]
plt.plot(
    x[peak_loc],
    y[peak_loc],
    color="red",
    marker="o",
    linestyle="",
    label="Peaks",
)
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle="dashed")
plt.savefig('build/kupfer2.pdf', bbox_inches = "tight")
plt.clf()