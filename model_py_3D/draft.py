from cell import HealthyCell
import numpy as np
import matplotlib.pyplot as plt

xsize = 3
ysize = 3
zsize = 4
drate = 0.20
l = 1

glucose = np.zeros((xsize,ysize))

for i in range(0,xsize):
    for j in range(0,ysize):
        glucose[i,j] = l
        l += 1

glucose3D = np.zeros((xsize,ysize,zsize))

l = 1

for k in range(0,zsize):
    for i in range(0,xsize):
        for j in range(0,ysize):
            glucose3D[i,j,k] = l
            l += 1

down = np.roll(glucose, 1, axis=0)
up = np.roll(glucose, -1, axis=0)
front = np.roll(glucose3D, 1, axis = 2)
back = np.roll(glucose3D, -1, axis = 2)
right = np.roll(glucose, 1, axis=(0, 1)) 
left = np.roll(glucose, -1, axis=(0, 1))

down_right = np.roll(down, 1, axis=(0, 1))
down_left = np.roll(down, -1, axis=(0, 1))
up_right = np.roll(up, 1, axis=(0, 1))
up_left = np.roll(up, -1, axis=(0, 1))

print(glucose)
print(down_right)
for i in range(ysize):  # Down
    down[0, i] = 0
    down_left[0, i] = 0
    down_right[0, i] = 0
print(down_right)
#for i in range(ysize):  # Up
#    up[xsize - 1, i] = 0
#    up_left[xsize - 1, i] = 0
#    up_right[xsize - 1, i] = 0
#for i in range(xsize):  # Right
#    right[i, 0] = 0
#    down_right[i, 0] = 0
#    up_right[i, 0] = 0
#for i in range(xsize):  # Left
#    left[i, ysize - 1] = 0
#    down_left[i, ysize - 1] = 0
#    up_left[i, ysize - 1] = 0
#return down + up + right + left + down_left + down_right + up_left + up_right

