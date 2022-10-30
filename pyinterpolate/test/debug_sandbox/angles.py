import numpy as np


angles = [0, 180, 30, 90, -120, -290]
my_angle = 45

r_ang = np.deg2rad(my_angle)
r_angles = np.deg2rad(angles)

for r in r_angles:
    out = r - r_ang
    out_rev = r_ang - r
    ang_out = np.rad2deg(out % (2 * np.pi))
    ang_out_rev = np.rad2deg(out_rev % (2 * np.pi))
    print(out, ang_out, ang_out_rev)