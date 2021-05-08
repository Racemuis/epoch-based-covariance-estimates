# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:17:44 2021

Figure 2.3(b):
Plot a sphere that can conceptually be seen as a manifold. 
Add "tangent vectors" to the sphere on a point x.
"""

from utilities import meshgrid_sphere, plot_tangent_line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

x, y, z = meshgrid_sphere()
fig = plt.figure(figsize=(6,6))
ax = fig.gca(projection='3d')

ax.plot_wireframe(x,y,z, linewidth=1, rstride=1, cstride=2, label = 'Manifold')
plot_tangent_line(ax, 1)
ax.plot([0], [0], [1], color = 'orange', marker = '.', markersize = 20, alpha = 10, label = 'x', linestyle = 'None')

ax.view_init(25, 40)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])

plt.legend(loc = 'lower left')
plt.show()
