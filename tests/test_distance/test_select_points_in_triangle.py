import numpy as np

from pyinterpolate.distance.angular import generate_triangles, \
    filter_triangles_mask, triangle_mask


def test_masking_xor_ops():
    coordinates = np.linspace(0, 1, 20)
    coords = []
    for vx in coordinates:
        for vy in coordinates:
            coords.append([vx, vy])

    coords = np.array(coords)
    lag = 0.4
    step_size = 0.2
    theta = 90
    tolerance = 0.25

    trs_a = generate_triangles(coords,
                               lag,
                               theta,
                               tolerance)
    trs_b = generate_triangles(coords,
                               lag + step_size,
                               theta,
                               tolerance)
    trs_c = generate_triangles(coords,
                               lag + step_size * 2,
                               theta,
                               tolerance)

    coord_index = int((len(coords)) / 2) + 4

    mask_a = triangle_mask(
        triangle_1=trs_a[coord_index][0],
        triangle_2=trs_a[coord_index][1],
        coordinates=coords
    )
    mask_b = triangle_mask(
        triangle_1=trs_b[coord_index][0],
        triangle_2=trs_b[coord_index][1],
        coordinates=coords
    )
    mask_c = triangle_mask(
        triangle_1=trs_c[coord_index][0],
        triangle_2=trs_c[coord_index][1],
        coordinates=coords
    )

    mask = filter_triangles_mask(old_mask=mask_a, new_mask=mask_b)
    mask_2 = filter_triangles_mask(old_mask=mask_b, new_mask=mask_c)

    mask_xor = np.logical_xor(mask_a, mask_b)
    mask_xor_2 = np.logical_xor(mask_b, mask_c)

    assert np.array_equal(mask, mask_xor)
    assert np.array_equal(mask_2, mask_xor_2)
