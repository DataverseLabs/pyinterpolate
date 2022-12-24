import numpy as np
from pyinterpolate.distance.distance import calc_angles
from pyinterpolate.processing.select_values import generate_triangles, select_points_within_ellipse

angles = [45, -45]

INPUT_ARRAY = np.array(
    [
        [1, 5, 0],  [2, 5, 1 ], [3, 5, 2 ], [4, 5, 3 ], [5, 5, 4 ],
        [1, 4, 5],  [2, 4, 6 ], [3, 4, 7 ], [4, 4, 8 ], [5, 4, 9 ],
        [1, 3, 10], [2, 3, 11], [3, 3, 12], [4, 3, 13], [5, 3, 14],
        [1, 2, 15], [2, 2, 16], [3, 2, 17], [4, 2, 18], [5, 2, 19],
        [1, 1, 20], [2, 1, 21], [3, 1, 22], [4, 1, 23], [5, 1, 24]
    ]
)

ptb = np.array([[4, 3]])
trb = generate_triangles(ptb, 2, 45, 0.2)
print(trb[0][0])
print(trb[0][1])

ell = select_points_within_ellipse(ptb, INPUT_ARRAY[:, :-1], 2, 2, theta=45, minor_axis_size=.2)
print(INPUT_ARRAY[ell])

# my_angles = calc_angles(INPUT_ARRAY[:, :-1])
# for idx, ang in enumerate(my_angles):
#     print(ang)
#     print(INPUT_ARRAY[idx])
#     print('')
#     print('---')
#     print('')
