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

# Funktionen definieren

def theta(E):
    return 180 / np.pi * np.arcsin(const.h * const.c / (2 * 201.4e-12 * const.e * E))

def sigma(E, Z):
    return Z - np.sqrt((E / 13.6) - ((1 / 137) ** 2 * Z ** 4) / 4)

def s1(E):
    return 29 - np.sqrt(E / 13.6)

def s2(E, s1):
    return 29 - np.sqrt(4 * (29 - s1) ** 2 - 4 * E / 13.6)

def s3(E, s1):
    return 29 - np.sqrt(9 * (29 - s1) ** 2 - 9 * E / 13.6)

def energ(the):
    return const.h * const.c / (2 * 201.4e-12 * np.sin(np.pi / 180 * the) * const.e)

# Plots erstellen für Brom

data = pd.read_csv('tables/messdaten/Messung_Brom/Brom_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

rel_min = np.min(y)
rel_max = np.max(y)
theta_mitte_y = (rel_max - rel_min) / 2
theta_mitte = (x[y==rel_max] - x[y==rel_min]) / 2

x1 = np.diff(y)
tolger = 3
x3 = []
x4 = []
x8 = []
n = 0
for i in range(len(x1)):
    if n == 0:
        if abs(x1[i]) < tolger:
            x3.append(i)
        else:
            x8.append(i)
            n = 1
    else:
        if abs(x1[i]) < tolger:
            x4.append(i)

        else:
            x8.append(i)
            n = 1

x5 = []
for i in range(len(x3)):
    x5.append(y[x3[i]])

plt.axhline(np.mean(x5),c='darkgreen',  label='Plateau')
x6 = []
for i in range(len(x4)):
    x6.append(y[x4[i]])

plt.axhline(np.mean(x6), c='darkgreen')
y2 = np.mean(x6) + (np.mean(x5) - np.mean(x6)) / 2
regx = x[x8]
regy = y[x8]
regx = regx[1:-1]
regy = regy[1:-1]

def f(x, a, b):
    return a * x + b

params, covariance_matrix = curve_fit(f, regx, regy)

# errors = np.sqrt(np.diag(covariance_matrix))
energy = energ((x[y==rel_min] + theta_mitte)[0])
print(energy)
print((x[y==rel_min] + theta_mitte)[0])

plt.plot(
    regx,
    f(regx, params[0], params[1]),
    color="red",
    ms=4,
    marker="",
    linestyle="-",
    label="Regression",
)

plt.axvline(x[y==rel_min] + theta_mitte, c='k', label='Mittleres Theta')
plt.axhline(rel_min + theta_mitte_y, c='grey')

plt.plot(x, y, color="darkred", ms=6, marker=".", linestyle="", label="Messwerte")
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle=":")
plt.savefig('build/brom.pdf', bbox_inches = "tight")
plt.clf()

# Zink

data = pd.read_csv('tables/messdaten/Messung_Zink/Zink_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

rel_min = np.min(y)
rel_max = np.max(y)
theta_mitte_y = (rel_max - rel_min) / 2
theta_mitte = (x[y==rel_max] - x[y==rel_min]) / 2


x1 = np.diff(y)
# print(x1)
tolger = 2
x3 = []
x4 = []
x8 = []
n = 0
for i in range(len(x1)):
    if n == 0:
        if abs(x1[i]) < tolger:
            x3.append(i)
        else:
            x8.append(i)
            n = 1
    else:
        if abs(x1[i]) < tolger:
            x4.append(i)

        else:
            x8.append(i)
            n = 1

x5 = []
for i in range(len(x3)):
    x5.append(y[x3[i]])

plt.axhline(np.mean(x5),c='darkgreen',  label='Plateau')
x6 = []
for i in range(len(x4)):
    x6.append(y[x4[i]])

plt.axhline(np.mean(x6), c='darkgreen')
y2 = np.mean(x6) + (np.mean(x5) - np.mean(x6)) / 2
regx = x[x8]
regy = y[x8]
regx = regx[1:-1]
regy = regy[1:-1]


def f(x, a, b):
    return a * x + b


params, covariance_matrix = curve_fit(f, regx, regy)

# errors = np.sqrt(np.diag(covariance_matrix))
energy = energ((x[y==rel_min] + theta_mitte)[0])
print(energy)
print((x[y==rel_min] + theta_mitte)[1])


plt.plot(
    regx,
    f(regx, params[0], params[1]),
    color="red",
    ms=4,
    marker="",
    linestyle="-",
    label="Regression",
)

plt.axvline((x[y==rel_min])[1] + theta_mitte[1], c='k', label='Mittleres Theta')
plt.axhline(rel_min + theta_mitte_y, c='grey')

plt.plot(x, y, color="darkred", ms=6, marker=".", linestyle="", label="Messwerte")
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle=":")
plt.savefig('build/zink.pdf', bbox_inches = "tight")
plt.clf()

# Strontium

data = pd.read_csv('tables/messdaten/Messung_Strontium/Strontium_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

rel_min = np.min(y)
rel_max = np.max(y)
theta_mitte_y = (rel_max - rel_min) / 2
theta_mitte = (x[y==rel_max] - x[y==rel_min]) / 2


x1 = np.diff(y)
# print(x1)
tolger = 1
x3 = []
x4 = []
x8 = []
n = 0
for i in range(len(x1)):
    if n == 0:
        if abs(x1[i]) < tolger:
            x3.append(i)
        else:
            x8.append(i)
            n = 1
    else:
        if abs(x1[i]) < tolger:
            x4.append(i)

        else:
            x8.append(i)
            n = 1

x5 = []
for i in range(len(x3)):
    x5.append(y[x3[i]])

plt.axhline(np.mean(y[-3]),c='darkgreen',  label='Plateau')
x6 = []
for i in range(len(x4)):
    x6.append(y[x4[i]])

plt.axhline(np.mean(y[:1]), c='darkgreen')
y2 = np.mean(x6) + (np.mean(x5) - np.mean(x6)) / 2
regx = x[x8]
regy = y[x8]
regx = regx[1:-1]
regy = regy[1:-1]


def f(x, a, b):
    return a * x + b


params, covariance_matrix = curve_fit(f, regx, regy)

# errors = np.sqrt(np.diag(covariance_matrix))
energy = energ((x[y==rel_min] + theta_mitte)[0])
print(energy)
print((x[y==rel_min] + theta_mitte))


plt.plot(
    regx,
    f(regx, params[0], params[1]),
    color="red",
    ms=4,
    marker="",
    linestyle="-",
    label="Regression",
)

plt.axvline((x[y==rel_min]) + theta_mitte, c='k', label='Mittleres Theta')
plt.axhline(rel_min + theta_mitte_y, c='grey')

plt.plot(x, y, color="darkred", ms=6, marker=".", linestyle="", label="Messwerte")
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle=":")
plt.savefig('build/strontium.pdf', bbox_inches = "tight")
plt.clf()

# Zirkonium

data = pd.read_csv('tables/messdaten/Messung_zirkonium/zirkonium_2.txt', decimal=',', delimiter = "\t")
data = data.to_numpy()
x = data[:,0]
y = data[:,1]

rel_min = np.min(y)
rel_max = np.max(y)
theta_mitte_y = (rel_max - rel_min) / 2
theta_mitte = (x[y==rel_max] - x[y==rel_min]) / 2


x1 = np.diff(y)
# print(x1)
tolger = 2
x3 = []
x4 = []
x8 = []
n = 0
for i in range(len(x1)):
    if n == 0:
        if abs(x1[i]) < tolger:
            x3.append(i)
        else:
            x8.append(i)
            n = 1
    else:
        if abs(x1[i]) < tolger:
            x4.append(i)

        else:
            x8.append(i)
            n = 1

x5 = []
for i in range(len(x3)):
    x5.append(y[x3[i]])

plt.axhline(np.mean(x5),c='darkgreen',  label='Plateau')
x6 = []
for i in range(len(x4)):
    x6.append(y[x4[i]])

plt.axhline(np.mean(y[-3]), c='darkgreen')
y2 = np.mean(x6) + (np.mean(x5) - np.mean(x6)) / 2
regx = x[x8]
regy = y[x8]
regx = regx[1:-1]
regy = regy[1:-1]


def f(x, a, b):
    return a * x + b


params, covariance_matrix = curve_fit(f, regx, regy)

# errors = np.sqrt(np.diag(covariance_matrix))
energy = energ((x[y==rel_min] + theta_mitte)[0])
print(energy)
print((x[y==rel_min] + theta_mitte)[1])


plt.plot(
    regx,
    f(regx, params[0], params[1]),
    color="red",
    ms=4,
    marker="",
    linestyle="-",
    label="Regression",
)

plt.axvline((x[y==rel_min])[1] + theta_mitte[1], c='k', label='Mittleres Theta')
plt.axhline(rel_min + theta_mitte_y, c='grey')

plt.plot(x, y, color="darkred", ms=6, marker=".", linestyle="", label="Messwerte")
plt.xlabel(r"$\theta \, / \, °$")
plt.ylabel(r"Impulse")
plt.legend(loc="best")
plt.grid(linestyle=":")
plt.savefig('build/zirkonium.pdf', bbox_inches = "tight")
plt.clf()