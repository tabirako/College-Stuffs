import matplotlib.pyplot as plt
import numpy as np
# Set initial Seeds
x = 42

# Generate 10000ish randomn number with walrus operator
rec = [(x := (65539 * x) % (2**31)) for _ in range(10000+2)]

rec1 = np.array(rec[0::3])/2**31
rec2 = np.array(rec[1::3])/2**31
rec3 = np.array(rec[2::3])/2**31

ax = plt.figure().add_subplot(projection='3d')
ax.scatter3D(rec1, rec2, rec3)

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35, roll=0)

plt.show()