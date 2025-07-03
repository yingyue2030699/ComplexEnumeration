"""
This file includes code adapted from the 'pointgroup' package,
originally authored by Abel Carreras (https://github.com/abelcarreras/pointgroup),
and is licensed under the MIT License:

The MIT License (MIT)

Copyright (c) 2023 Efrem Bernuz and Abel Carreras

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from itertools import permutations
import numpy as np


def get_cubed_sphere_grid_points(delta):
    """
    Generate a set of quasi-uniform grid points on the surface of a unit sphere
    using a cubed-sphere projection approach.

    This creates points by placing a grid on each face of a cube,
    projecting those points onto the unit sphere, and symmetrizing via permutations.

    Parameters
    ----------
    delta : float
        Approximate angular resolution (in radians). Smaller delta → more points.

    Returns
    -------
    generator of np.ndarray
        Yields 3D unit vectors uniformly spread on the sphere.

    Examples
    --------
    >>> points = list(get_cubed_sphere_grid_points(0.5))
    >>> len(points)
    216
    >>> np.allclose(np.linalg.norm(points[0]), 1.0)
    True
    """

    num_points = int(1.0 / delta)

    if num_points < 1:
        return [(1, 0, 0)]

    for i in range(-num_points, num_points+1):
        x = i * delta
        for j in range(-num_points, num_points+1):
            y = j * delta
            for p in permutations([x, y, 1]):
                norm = np.linalg.norm([x, y, 1])
                yield np.array(p)/norm
