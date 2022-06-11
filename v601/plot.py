import matplotlib.pyplot as plt
import numpy as np


U = np.array([0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ,7.5, 7.75, 8  ,9  ,10 ])
y = np.array([15.8, 14.1, 12.6, 11.1, 9.6, 7.8, 6, 4, 2.8, 1.5, 0.8, 0.3, 0.3])

delta_U = np.zeros_like(U)
delta_I = np.zeros_like(U)

for i, U_i in enumerate(U):
    if i < np.size(U) - 1:
        delta_U[i + 1] = U[i + 1] - U[i]
        delta_I[i + 1] = y[i + 1] - y[i]

m = -(delta_I / delta_U)

plt.plot(U[1:], m[1:], 'x', c = 'r')
plt.plot(U[1:], m[1:], '--', c = 'b')
plt.xlabel(r'$U \, / \,V$')
plt.ylabel(r'$- m$')
plt.grid()
plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 

U = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6 , 7, 8, 9, 10])
y = np.array([17.8, 14.3, 11, 8.3, 5.8, 3.2, 1.4, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4])

delta_U = np.zeros_like(U)
delta_I = np.zeros_like(U)

for i, U_i in enumerate(U):
    if i < np.size(U) - 1:
        delta_U[i + 1] = U[i + 1] - U[i]
        delta_I[i + 1] = y[i + 1] - y[i]

m = -(delta_I / delta_U)

plt.plot(U[1:], m[1:], 'x', c = 'r')
plt.plot(U[1:], m[1:], '--', c = 'b')
plt.xlabel(r'$U \, / \,V$')
plt.ylabel(r'$- m$')
plt.grid()
plt.savefig('build/plot2.pdf', bbox_inches = "tight")
plt.clf() 