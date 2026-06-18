"""
Tests for Lie algebra basis matrix generators.
"""

import itertools
import pytest
import numpy as np

# Adjust this import to match your actual module structure
from paulie import get_pauli_string as p
from paulie.common.algebra_basis import (
    TypeAlgebra,
    get_u_basis,
    get_n_so_basis,
    get_so_basis,
    get_n_su_basis,
    get_su_basis,
    get_n_sp_basis,
    get_sp_basis,
    get_group_basis,
    get_n_basis,
    get_algebras_basis,
)
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection

# ---------------------------------------------------------------------------
# Mathematical Property Helpers
# ---------------------------------------------------------------------------


def is_skew_hermitian(mat: np.ndarray) -> bool:
    """Check if a matrix is skew-Hermitian (X^dagger = -X)."""
    return np.allclose(mat + mat.conj().T, 0)


def is_antisymmetric(mat: np.ndarray) -> bool:
    """Check if a matrix is antisymmetric (X^T = -X)."""
    return np.allclose(mat + mat.T, 0)


def is_traceless(mat: np.ndarray) -> bool:
    """Check if a matrix is traceless (Tr(X) = 0)."""
    return np.isclose(np.trace(mat), 0)


def is_symplectic(mat: np.ndarray, n: int) -> bool:
    """
    Check the symplectic condition (X^T J + J X = 0).
    J = [[0, I_n], [-I_n, 0]]
    """
    identity = np.eye(n)
    zeros = np.zeros((n, n))
    J = np.block([[zeros, identity], [-identity, zeros]])
    return np.allclose(mat.T @ J + J @ mat, 0)


# ---------------------------------------------------------------------------
# u(1) Tests
# ---------------------------------------------------------------------------


def test_u1_basis_properties() -> None:
    """Test dimension, shape, and properties of u(1) basis."""
    basis = get_u_basis(1)
    assert basis.shape == (1, 1, 1)
    assert np.allclose(basis[0], [[1j]])
    assert is_skew_hermitian(basis[0])


def test_u1_basis_errors() -> None:
    """Test that u(n) raises ValueError for n != 1."""
    with pytest.raises(ValueError, match="Invalid construction"):
        get_u_basis(2)


# ---------------------------------------------------------------------------
# so(n) Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4])
def test_so_basis_properties(n: int) -> None:
    """Test dimension calculation and matrix properties for so(n)."""
    expected_dim = n * (n - 1) // 2
    assert get_n_so_basis(n) == expected_dim

    basis = get_so_basis(n)
    assert basis.shape == (expected_dim, n, n)

    for mat in basis:
        assert is_skew_hermitian(mat)
        assert is_antisymmetric(mat)
        assert np.all(np.isreal(mat))  # so(n) generators are strictly real


def test_so_basis_errors() -> None:
    """Test error handling for so(n) edge cases."""
    with pytest.raises(ValueError, match="positive integer"):
        get_n_so_basis(0)
    with pytest.raises(TypeError, match="positive integer"):
        get_n_so_basis(3.5)  # type: ignore


# ---------------------------------------------------------------------------
# su(n) Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [8, 16])
def test_su_basis_properties(n: int) -> None:
    """Test dimension calculation and matrix properties for su(n)."""
    expected_dim = n**2 - 1
    assert get_n_su_basis(n) == expected_dim

    basis = get_su_basis(n)
    assert basis.shape == (expected_dim, n, n)

    for mat in basis:
        assert is_skew_hermitian(mat)
        assert is_traceless(mat)


def test_su_basis_errors() -> None:
    """Test error handling for su(n), bounded by n > 2^2 (4)."""
    with pytest.raises(ValueError, match=r"greater than 2\^2"):
        get_n_su_basis(4)
    with pytest.raises(TypeError):
        get_n_su_basis(5.5)  # type: ignore


# ---------------------------------------------------------------------------
# sp(n) Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4])
def test_sp_basis_properties(n: int) -> None:
    """Test dimension calculation and matrix properties for sp(n)."""
    expected_dim = n * (2 * n + 1)
    assert get_n_sp_basis(n) == expected_dim

    basis = get_sp_basis(n)
    # sp(n) matrices are 2n x 2n
    assert basis.shape == (expected_dim, 2 * n, 2 * n)

    for mat in basis:
        assert is_skew_hermitian(mat)
        assert is_symplectic(mat, n)


def test_sp_basis_errors() -> None:
    """Test error handling for sp(n) power-of-2 requirements."""
    with pytest.raises(ValueError, match="power of 2 AND greater or equal to 2"):
        get_n_sp_basis(3)  # 3 is not a power of 2
    with pytest.raises(TypeError):
        get_n_sp_basis(2.0)  # type: ignore


# ---------------------------------------------------------------------------
# Dispatcher & Direct Sum Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "algebra_type, n",
    [
        (TypeAlgebra.U, 1),
        (TypeAlgebra.SO, 3),
        (TypeAlgebra.SU, 8),
        (TypeAlgebra.SP, 2),
    ],
)
def test_get_n_basis_dispatch(algebra_type: TypeAlgebra, n: int) -> None:
    """Test that get_n_basis correctly calculates total dimensions for each algebra type."""
    match algebra_type:
        case TypeAlgebra.U:
            expected_dim = 1
        case TypeAlgebra.SO:
            expected_dim = n * (n - 1) // 2
        case TypeAlgebra.SU:
            expected_dim = n**2 - 1
        case TypeAlgebra.SP:
            expected_dim = n * (2 * n + 1)
    assert get_n_basis(algebra_type, n) == expected_dim


def test_get_n_basis_errors() -> None:
    """Test error handling for unsupported algebra types in get_n_basis."""
    with pytest.raises(ValueError, match="Unsupported algebra type"):
        get_n_basis("INVALID_TYPE", 2)  # type: ignore


def test_get_group_basis_dispatch() -> None:
    """Test that the dispatcher correctly routes to the right generator."""
    assert get_group_basis(TypeAlgebra.U, 1).shape == (1, 1, 1)
    assert get_group_basis(TypeAlgebra.SO, 3).shape == (3, 3, 3)
    assert get_group_basis(TypeAlgebra.SU, 8).shape == (63, 8, 8)
    assert get_group_basis(TypeAlgebra.SP, 2).shape == (10, 4, 4)

    # Use a dummy invalid string to test the default fallback
    with pytest.raises(ValueError, match="Unsupported algebra type"):
        get_group_basis("INVALID_TYPE", 2)  # type: ignore


def test_get_algebras_basis_direct_sum() -> None:
    """Test block-diagonal allocation for direct sums of Lie algebras."""
    multipliers = [2, 1]
    groups = [TypeAlgebra.U, TypeAlgebra.SO]
    sizes = [1, 3]

    # Total expected dimension: 2*(u1 dim 1) + 1*(so3 dim 3) = 5
    # Total expected matrix size: 2*(1) + 1*(3) = 5
    basis = get_algebras_basis(multipliers, groups, sizes)
    assert basis.shape == (5, 5, 5)

    # Validate the block diagonal isolation for the first matrix
    # The first U(1) generator should only occupy the [0, 0] cell.
    assert basis[0, 0, 0] != 0
    assert np.allclose(basis[0, 1:, :], 0)
    assert np.allclose(basis[0, :, 1:], 0)

    # Validate the last generator belongs to the SO(3) block
    # and leaves the upper blocks isolated (0 to 1).
    assert np.allclose(basis[-1, 0:2, 0:2], 0)


def test_get_algebras_basis_errors() -> None:
    """Test error handling for mismatched list inputs in direct sums."""
    with pytest.raises(ValueError, match="same length"):
        get_algebras_basis([1, 2], [TypeAlgebra.U], [1])


# ---------------------------------------------------------------------------
# Pauli String Interface Tests
# ---------------------------------------------------------------------------


def test_pauli_string_algebra_basis_interface() -> None:
    """Test the interface for generating an algebra basis directly from Pauli strings."""
    generators = p(["XY", "XZ"], n=4)
    basis = generators.get_algebra_basis()

    # 3. Verify the output type and structure
    assert basis is not None, "Basis generation should not return None."
    assert isinstance(basis, np.ndarray), "Basis should be returned as a NumPy array."
    assert basis.ndim == 3, "Basis should be a 3D array of matrices (num_matrices, n, n)."

    num_matrices, rows, cols = basis.shape
    assert rows == cols, f"Basis matrices must be square, but got {rows}x{cols}."
    assert num_matrices > 0, "Basis should contain at least one matrix."

    # 4. Verify mathematical consistency (Lie algebra generators must be skew-Hermitian)
    for mat in basis:
        assert np.allclose(mat.conj().T, -mat), "Basis matrices must be skew-Hermitian"


# ---------------------------------------------------------------------------
# Lie Closure / Span Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "o, n",
    [
        (["XY"], None),
        (["XY", "XZ"], 4),
        (["IYZI", "IIXX", "IIYZ", "IXXI", "XXII", "YZII"], None),
    ],
)
def test_basis_commutators_are_real_linear_combinations(
    o: list[str] | list[PauliString] | PauliStringCollection, n: int | None
) -> None:
    """Test that any commutator element in the Lie closure is expressible
    as a strictly REAL linear combination of the basis elements.

    This works dynamically for n x n or 2n x 2n (Sp(n)) matrix representations.
    """
    generators = p(o, n)
    basis = generators.get_algebra_basis()

    # rows and cols will correctly capture 2n x 2n for sp(n) automatically
    num_matrices, rows, cols = basis.shape
    if num_matrices < 2:
        return

    # 1. Flatten each matrix: shape becomes (num_matrices, rows * cols)
    basis_flat = basis.reshape(num_matrices, -1)

    # 2. Transpose first, then stack vertically.
    # basis_flat.T has shape (rows * cols, num_matrices)
    # A will have shape (2 * rows * cols, num_matrices) -> Exactly M columns!
    A = np.vstack([np.real(basis_flat.T), np.imag(basis_flat.T)])

    # 3. Loop over every unique pair to compute their commutators [A, B] = AB - BA
    for mat_a, mat_b in itertools.combinations(basis, 2):
        commutator = mat_a @ mat_b - mat_b @ mat_a
        commutator_flat = commutator.reshape(-1)

        # Target vector y matches the vertical real/imag split layout of A
        y = np.concatenate([np.real(commutator_flat), np.imag(commutator_flat)])

        # 4. Solve the linear system A @ c = y for the M real coefficients
        c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

        # 5. Verify that the real linear combination reconstructs the commutator
        assert np.allclose(A @ c, y), (
            "Lie Closure Failure: A commutator cannot be expressed as a REAL linear "
            f"combination of the basis elements. Matrix block size was {rows}x{cols}."
        )
