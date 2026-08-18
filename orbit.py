import numpy as np
c = np.array([-49.8, 188.5, 14.1])
R = 260
for ang in range(0, 360, 60):
    a = np.radians(ang)
    # orbit in the x-z plane (horizontal circle around the ear)
    pos = c + np.array([-np.cos(a)*R, 25, np.sin(a)*R])
    print(f"{ang:3d}deg  position=[{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")