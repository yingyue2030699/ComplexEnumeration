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
import numpy as np

def rotation_matrix(axis, angle, tol = 1e-8):
    """
    Compute the 3D rotation matrix for a rotation around a given axis by a specified angle.

    Parameters:
    - axis: array-like of shape (3,), the axis of rotation (will be normalized)
    - angle: float, angle in radians
    - tol: float, threshold for nonzero checking of the vectors

    Returns:
    - 3x3 numpy array representing the rotation matrix
    """
    axis = np.array(axis)
    norm = np.linalg.norm(axis)
    assert norm > tol, "Axis must be a non-zero vector"

    axis = axis / norm  # Normalize the axis

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta

    x, y, z = axis

    # Rodrigues' rotation formula
    rot_matrix = np.array([
        [x*x*one_minus_cos + cos_theta,     x*y*one_minus_cos - z*sin_theta, x*z*one_minus_cos + y*sin_theta],
        [y*x*one_minus_cos + z*sin_theta,   y*y*one_minus_cos + cos_theta,   y*z*one_minus_cos - x*sin_theta],
        [z*x*one_minus_cos - y*sin_theta,   z*y*one_minus_cos + x*sin_theta, z*z*one_minus_cos + cos_theta]
    ])

    return rot_matrix

class Rotation:
    """
    Represents a proper rotation (Cn) about a given axis.

    Parameters:
    - axis: array-like, axis of rotation (will be normalized)
    - order: int, rotation order (n-fold symmetry → angle = 2π / n)
    """
    def __init__(self, axis, order=1):
        self._axis = np.array(axis)
        self._order = order

    def get_matrix(self):
        """
        Return the rotation matrix corresponding to a rotation of 2π / order about the axis.
        """
        angle = 2 * np.pi / self._order
        return rotation_matrix(self._axis, angle)


class ImproperRotation:
    """
    Represents an improper rotation (Sn), a combination of a proper rotation and a reflection.

    Parameters:
    - axis: array-like, axis of improper rotation (will be normalized)
    - order: int, rotation order (n-fold → rotation of 2π / n followed by reflection)
    """
    def __init__(self, axis, order=1):
        self._axis = np.array(axis)
        self._order = order

    def get_matrix(self):
        """
        Return the matrix representing the improper rotation: rotation followed by reflection.
        """
        # Rotation matrix (Cn)
        angle = 2 * np.pi / self._order
        rot_matrix = rotation_matrix(self._axis, angle)

        # Reflection matrix through a plane perpendicular to the axis
        u = self._axis
        refl_matrix = np.identity(3) - 2 * np.outer(u, u) / np.dot(u, u)

        # Improper rotation = rotation followed by reflection
        return np.dot(rot_matrix, refl_matrix.T)
