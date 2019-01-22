import numpy as np
import matplotlib.pyplot as plt
import os, shutil, subprocess
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

### Using trick from https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
### to add alpha gradient in color map
cmap = plt.cm.Blues
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

### Using the colorline recipe to draw gradient lines : https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
### Code adapted for Line3DCollection
def make_segments(x, y, z):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 3 (x, y, z) array
    '''
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(x, y, z, k=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if k is None:
        k = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(k, "__iter__"):  # to check for numerical input -- this is a hack
        k = np.array([k])
        
    k = np.asarray(k)
    
    segments = make_segments(x, y, z)
    lc = Line3DCollection(segments, array=k, cmap=cmap, norm=norm, linewidth=linewidth)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

# Simulation part
sigma = 10.0
beta  = 8.0/3.0
rho   = 28.0

x = 2.01
y = 5.0
z = 20.0

# dx/dt, dy/dt, dz/dt
def ddt(x, y, z):
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z

    return dxdt, dydt, dzdt

# Euler method
print('Simulating ...')
max_t = 200.0
t     = 0.0
dt    = 0.001

pos = []
while t < max_t:
    dxdt, dydt, dzdt = ddt(x, y, z)
    x += dxdt * dt
    y += dydt * dt
    z += dzdt * dt

    pos.append((x, y, z))

    t += dt

# Rendering part
print('Rendering ...')
pos = np.asarray(pos)
L = 5000 # Length of the trail
Np = pos.shape[0] - L
cur = L

# Limits of the graph
min_x = pos[:,0].min()
max_x = pos[:,0].max()
min_y = pos[:,1].min()
max_y = pos[:,1].max()
min_z = pos[:,2].min()
max_z = pos[:,2].max()

steps = 100 # Frame skipping

# Violently removing everything from the render directory
shutil.rmtree('render')
os.mkdir('render')
for i in range(Np//steps):
    filename = 'img_{:05}.png'.format(i)
    print('Rendering', filename)
    plt.close('all')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colorline(pos[cur-L:cur,0], pos[cur-L:cur,1], pos[cur-L:cur,2], cmap=my_cmap, linewidth=2)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    ax.set_zlim((min_z, max_z))
    plt.savefig('render/'+filename)
    cur+=steps

# Building ffmpeg command
cmd = 'ffmpeg -framerate 60 -i render/img_%05d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p render/movie.mp4'
print('Converting to video :', cmd)
subprocess.call(cmd.split())
    
