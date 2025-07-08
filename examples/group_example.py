import ode_gen
from ode_gen.symmetry.pointgroup import PointGroup

pg = PointGroup(positions=[[ 0.000000,  0.000000,  0.000000],
                            [ 0.000000,  0.000000,  1.561000],
                            [ 0.000000,  1.561000,  0.000000],
                            [ 0.000000,  0.000000, -1.561000],
                            [ 0.000000, -1.561000,  0.000000],
                            [ 1.561000,  0.000000,  0.000000],
                            [-1.561000,  0.000000,  0.000000]], 
                            symbols=['S', 'F', 'F', 'F', 'F', 'F', 'F'])

pgstr = pg.get_point_group()

print(pgstr)

import numpy as np

# Regular tetrahedron centered at origin, inscribed in unit sphere
tetrahedron_coords = np.array([
    [ 1,  1,  1],
    [-1, -1,  1],
    [-1,  1, -1],
    [ 1, -1, -1]
]) / np.sqrt(3)

print(np.round(tetrahedron_coords, 6))

pg = PointGroup(positions=tetrahedron_coords, symbols=["B", "B", "B", "B"])

pgstr = pg.get_point_group()

print(pgstr)

# Create a dimer: tetrahedron and its C2-rotated partner across XY plane (flip Z)
tetra_A = tetrahedron_coords
tetra_B = tetrahedron_coords
tetra_B[:, 2] *= -1  # flip Z to create C2-related copy

pg = PointGroup(positions=tetrahedron_coords, symbols=["B", "B", "B", "B","B", "B", "B", "B"])

pgstr = pg.get_point_group()

print(pgstr)