# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:51:52 2021

Figure 2.3(c):
Plot a sphere that can conceptually be seen as a manifold. 
Add a "tangent space" to the sphere on a point x.
"""

from utilities import meshgrid_sphere, meshgrid_plane
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(figsize=(6,6), dpi = 200)
ax = fig.gca(projection='3d')

x, y, z = meshgrid_sphere()
ax.plot_wireframe(x,y,z, linewidth=1, rstride=1, cstride=2, label = 'Manifold')

x,y,z = meshgrid_plane(0, 1, 1.5)
surf = ax.plot_surface(x, y, z, color = 'orange', linewidth=1, rstride=1, cstride=2, alpha = 0.9, label = 'Tangent space')

# fix legend for surfplot
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d

ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
ax.view_init(25, 40)

plt.legend(loc = 'lower left')
plt.show()
