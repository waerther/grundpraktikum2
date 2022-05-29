import matplotlib.pyplot as plt
import numpy as np

# Erster Plot für Impuls Echo Verfahren

# Daten einlesen
d = np.array([0.0404, 0.0615, 0.0805, 0.1019, 0.1205]) * 10**3
time = np.array([30.3, 46.0, 59.5, 75.5, 88.8])

params, covariance_matrix = np.polyfit(time, 2*d, deg=1, cov=True)

x_plot = np.linspace(0, np.max(time))

plt.plot(time, 2*d, 'x', label="Messwerte")
plt.plot(
    x_plot,
    params[0] * x_plot + params[1],
    label='Lineare Regression',
    linewidth=1,
    color='red',
)
plt.xlabel(r'$t \, / \, ( s \cdot 10^6)$')
plt.ylabel(r'$ 2 d \, / \, mm$')
plt.grid(ls=':')
plt.legend(loc="best")
plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 


# Zweiter Plot für Durchschallungsverfahren

# Daten einlesen
d = np.array([0.0404, 0.0615, 0.0805, 0.1205]) * 10**3
time = np.array([16.0, 23.8, 31.0, 45.0])

params, covariance_matrix = np.polyfit(time, d, deg=1, cov=True)

x_plot = np.linspace(0, np.max(time))

plt.plot(time, d, 'x', label="Messwerte")
plt.plot(
    x_plot,
    params[0] * x_plot + params[1],
    label='Lineare Regression',
    linewidth=1,
    color='red',
)
plt.xlabel(r'$t \, / \, ( s \cdot 10^6)$')
plt.ylabel(r'$ 2 d \, / \, mm$')
plt.grid(ls=':')
plt.legend(loc="best")
plt.savefig('build/plot2.pdf', bbox_inches = "tight")
plt.clf() 


# Erstellen des dritten Plots

# Daten einlesen
d = np.array([0.0404, 0.0615, 0.0805, 0.1019, 0.1205]) * 2
I = np.array([0.92, 0.25, 0.12, 0.025, 0.01])
I_plot = -np.log(I / 1.31) # 1.31 = u_0

params, covariance_matrix = np.polyfit(d, I_plot, deg=1, cov=True)

x_plot = np.linspace(0, np.max(d))

plt.plot(d, I_plot, 'x', label="Messwerte")
plt.plot(
    x_plot,
    params[0] * x_plot + params[1],
    label='Lineare Regression',
    linewidth=1,
    color='red',
)
plt.xlabel(r'$ 2 d \, / \, m$')
plt.ylabel(r'$ln (U_t / U_0)$')
plt.grid(ls=':')
plt.legend(loc="best")
plt.savefig('build/plot3.pdf', bbox_inches = "tight")
plt.clf()