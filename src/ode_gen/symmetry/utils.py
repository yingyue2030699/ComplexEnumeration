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

def magic_formula(n):
    """
    Compute the value of the expression: sqrt(n * 2^(3 - n))

    This "magic formula" may be used in contexts where the geometric scaling or 
    normalization of an n-mer system is required, such as symmetry-based energy or 
    entropy factors in molecular assemblies.

    Parameters
    ----------
    n : int or float
        The input value (typically an integer ≥ 1), e.g., number of subunits or symmetry order.

    Returns
    -------
    float
        The computed value of sqrt(n * 2^(3 - n))

    Examples
    --------
    >>> magic_formula(2)
    2.0
    >>> magic_formula(3)
    1.732...
    """
    return np.sqrt(n * 2 ** (3 - n))

def get_perpendicular_vector(vector, normalize=True, tol=1e-8):
    """
    Returns a vector that is perpendicular to the input vector.

    Parameters:
    ----------
    vector : array_like
        Input non-zero vector of shape (n,).
    normalize : bool, optional (default=True)
        If True, return a unit-length perpendicular vector.
    tol : float, optional
        Tolerance for numerical precision when checking orthogonality and normalization.

    Returns:
    -------
    perpendicular : ndarray of shape (n,)
        A vector orthogonal to the input vector. By default, it is normalized to unit length.

    Raises:
    ------
    ValueError:
        If the input vector is zero or if a valid perpendicular cannot be constructed.
    """
    vector = np.asarray(vector, dtype=float)

    if vector.ndim != 1:
        raise ValueError("Input vector must be one-dimensional.")
    if np.linalg.norm(vector) < tol:
        raise ValueError("Cannot compute perpendicular vector of a zero vector.")

    dim = vector.size

    # Choose a basis vector least aligned with the input to avoid degeneracy
    basis_index = np.argmin(np.abs(vector))
    basis_vector = np.eye(dim)[basis_index]

    # Compute perpendicular using cross product (only defined for 3D)
    if dim == 3:
        perp_vector = np.cross(vector, basis_vector)
    else:
        # For nD: use Gram-Schmidt-like projection
        proj = np.dot(vector, basis_vector) / np.dot(vector, vector) * vector
        perp_vector = basis_vector - proj

    norm = np.linalg.norm(perp_vector)
    if norm < tol:
        raise ValueError("Failed to construct a valid perpendicular vector.")

    if normalize:
        perp_vector = perp_vector / norm

    # Final checks
    if abs(np.dot(perp_vector, vector)) > tol:
        raise ValueError("Resulting vector is not orthogonal to input.")
    if normalize and abs(np.linalg.norm(perp_vector) - 1.0) > tol:
        raise ValueError("Resulting perpendicular vector is not normalized.")

    return perp_vector

def absolute_error_to_angle(error, points, tol=1e-8):
    """
    Convert absolute coordinate errors to angular errors (in radians),
    assuming origin-centered radial vectors.

    Parameters
    ----------
    error : float
        Absolute positional error (e.g., in angstroms or nanometers).
    points : ndarray of shape (N, 3)
        Array of 3D coordinates representing points from origin.
    tol : float
        Minimum radius threshold to avoid divide-by-zero.

    Returns
    -------
    angle_errors : ndarray of shape (N,)
        Angular errors in radians for each point.
    """
    points = np.asarray(points)
    radii = np.linalg.norm(points, axis=1)
    clipped_radii = np.clip(radii, tol, None)
    return error / clipped_radii

def angles_between_vector_and_vectors(reference_vec, targets, tol=1e-5):
    """
    Compute angles (in radians) between a reference vector and each row in a matrix.

    Parameters
    ----------
    reference_vec : array_like of shape (3,)
        The reference 3D vector.
    targets : ndarray of shape (N, 3)
        Array of target 3D vectors to compute angles with respect to.
    tol : float
        Threshold below which vector norms are treated as zero.

    Returns
    -------
    angles : ndarray of shape (N,)
        Array of angles (in radians) between reference vector and each target vector.
    """
    targets = np.asarray(targets)
    ref_norm = np.linalg.norm(reference_vec)
    target_norms = np.linalg.norm(targets, axis=1)

    dot_products = np.dot(targets, reference_vec)

    angles = []
    for dot, target_norm in zip(dot_products, target_norms):
        denom = target_norm * ref_norm
        if denom < tol:
            angles.append(0.0)
        else:
            cos_theta = np.clip(dot / denom, -1.0, 1.0)
            angles.append(np.arccos(cos_theta))
    return np.array(angles)


def normalized_radius_difference(reference_vec, targets, tol=1e-5):
    """
    Compute the relative radial distance differences (unitless),
    normalized by average radius between a reference vector and each target.

    Parameters
    ----------
    reference_vec : array_like of shape (3,)
        Reference 3D vector.
    targets : ndarray of shape (N, 3)
        Array of target 3D vectors.
    tol : float
        Minimum average radius to avoid divide-by-zero.

    Returns
    -------
    rel_differences : ndarray of shape (N,)
        Array of absolute radius differences normalized by average radius.
    """
    ref_norm = np.linalg.norm(reference_vec)
    target_norms = np.linalg.norm(targets, axis=1)

    avg_radii = np.clip((target_norms + ref_norm) / 2.0, tol, None)
    abs_diff = np.abs(target_norms - ref_norm)

    return abs_diff / avg_radii

def get_non_degenerated(eigenvalues, tolerance=0.1):
    """
    Identify the index of a non-degenerate eigenvalue in a set of eigenvalues.

    This function assumes that two of the eigenvalues are approximately equal (degenerate),
    and one is different (non-degenerate). It returns the index of the non-degenerate value.

    Parameters
    ----------
    eigenvalues : list or array-like of float
        A list or array of eigenvalues (typically of an inertia tensor).
        Expected to contain exactly 3 values.

    tolerance : float, optional
        Numerical tolerance for considering two eigenvalues as equal (degenerate). Default is 0.1.

    Returns
    -------
    int
        The index of the non-degenerate eigenvalue in the input list.

    Raises
    ------
    Exception
        If no non-degenerate eigenvalue is found.

    Examples
    --------
    >>> get_non_degenerated([1.0, 1.0, 2.0])
    2
    >>> get_non_degenerated([2.0, 1.0, 2.0])
    1
    """
    eigenvalues = list(eigenvalues)
    n = len(eigenvalues)

    for i in range(n):
        degenerate_count = sum(
            abs(eigenvalues[i] - eigenvalues[j]) < tolerance
            for j in range(n) if j != i
        )
        if degenerate_count == 0:
            return i

    raise Exception('Non-degenerate eigenvalue not found')
