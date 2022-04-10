import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

z = np.polyfit(U, k5, 3)
p = np.poly1d(z)
xp = np.linspace(-50, 280, 100)
plt.plot(xp, p(xp), '-b', label = "Ausgleichsrechnung", linewidth = 1, zorder=1)

dp = p.deriv()
ddp = dp.deriv()
dddp = ddp.deriv()

x = 0.0001046/8.479e-07

xp = np.linspace(-50, 280, 100)
plt.plot(x, p(x), 'Db', label = "Wendepunkt", markersize = 6, zorder=3)

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$I \, / \, \mathrm{mA}$')
plt.legend(loc="best")     
plt.grid(True)
plt.xlim(-10, 280) 
plt.ylim(-0.1, 2.3) 

plt.savefig('build/plot_k5.pdf', bbox_inches = "tight")
plt.clf() 