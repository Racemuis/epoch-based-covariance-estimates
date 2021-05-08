# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:37:21 2021

Figure 2.3(a):
Plot a sphere that can conceptually be seen as a manifold. 
"""
import matplotlib.pyplot as plt
from utilities import meshgrid_sphere
from mpl_toolkits.mplot3d import axes3d

x, y, z = meshgrid_sphere()
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')

ax.plot_wireframe(x,y,z, linewidth=1, rstride=1, cstride=2, label = 'Manifold')
ax.view_init(25, 40)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])

plt.legend(loc = 'lower left')
plt.show()

