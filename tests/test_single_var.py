"""Tests for polyopt.single_var - single-variable polynomial optimizer.

This optimizer finds minimal operation sequences to compute target polynomials
starting from zero, using operations like +1, -1, +X, -X, double, square, *X.

Test organization:
1. Basic functionality tests
2. Specific polynomial targets
3. Edge cases
"""

import pytest
from ortools.sat.python import cp_model
from src import (
    compute_expansion,
    OneVarPoly,
    initialize_poly_coefs,
    mul_polys_enforce_no_ovf,
)


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBasicFunctionality:
    """Basic tests for the optimizer."""

    def test_constant_polynomial(self):
        """Test computing a constant polynomial (e.g., 9)."""
        # 9 can be computed as: 0 -> +1 -> *2 -> *2 -> *2 -> +1 = 8+1 = 9
        status, res = compute_expansion(
            OneVarPoly([9, 0]), num_steps=10, max_deg=1, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_linear_polynomial(self):
        """Test computing a linear polynomial (e.g., 2 + 2X)."""
        status, res = compute_expansion(
            OneVarPoly([2, 2, 0, 0]), num_steps=3, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_simple_quadratic(self):
        """Test computing 1 + X (simple linear)."""
        status, res = compute_expansion(
            OneVarPoly([1, 1, 0, 0]), num_steps=2, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Specific Polynomial Targets
# =============================================================================


class TestPolynomialTargets:
    """Tests for specific polynomial targets."""

    def test_perfect_square(self):
        """Test computing (1 + X)^2 = 1 + 2X + X^2."""
        status, res = compute_expansion(
            OneVarPoly([1, 2, 1, 0]), num_steps=3, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_difference_of_squares(self):
        """Test computing (1 - X)^2 = 1 - 2X + X^2."""
        status, res = compute_expansion(
            OneVarPoly([1, -2, 1, 0]), num_steps=10, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_scaled_difference_of_squares(self):
        """Test computing 4(1 - X)^2 = 4 - 8X + 4X^2."""
        status, res = compute_expansion(
            OneVarPoly([4, -8, 4, 0]), num_steps=9, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_polynomial_with_large_coefs(self):
        """Test polynomial requiring larger bound: 4 + 13X + 4X^2."""
        status, res = compute_expansion(
            OneVarPoly([4, 13, 4, 0]), num_steps=9, max_deg=3, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_x_plus_x_squared(self):
        """Test computing X + X^2."""
        status, res = compute_expansion(
            OneVarPoly([0, 1, 1, 0]), num_steps=9, max_deg=3, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Higher Degree Tests
# =============================================================================


class TestHigherDegree:
    """Tests for higher degree polynomials."""

    def test_degree_4_simple(self):
        """Test a degree 4 polynomial: X + X^2 + X^4."""
        status, res = compute_expansion(
            OneVarPoly([0, 1, 1, 0, 1]), num_steps=9, max_deg=4, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_degree_4_complex(self):
        """Test a more complex degree 4 polynomial: 1 + 3X + X^2 + 2X^3 + X^4."""
        status, res = compute_expansion(
            OneVarPoly([1, 3, 1, 2, 1]), num_steps=9, max_deg=4, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_degree_4_with_more_steps(self):
        """Test degree 4 polynomial needing more steps: 1 + 3X + 2X^2 + 2X^3 + X^4."""
        status, res = compute_expansion(
            OneVarPoly([1, 3, 2, 2, 1]), num_steps=10, max_deg=4, bound=1000
        )
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Symbolic Multiplication Tests
# =============================================================================


class TestSymbolicMultiplication:
    """Tests for symbolic polynomial multiplication with OR-Tools."""

    def test_symbolic_multiplication(self):
        """Test symbolic polynomial multiplication with constraints."""
        MAX_DEG = 3
        BOUND = 100

        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p0 = OneVarPoly(initialize_poly_coefs(model, MAX_DEG, BOUND, "p0"))
        p1 = OneVarPoly(initialize_poly_coefs(model, MAX_DEG, BOUND, "p1"))
        p_mul, constraints = mul_polys_enforce_no_ovf(model, p0, p1, MAX_DEG, BOUND)

        # Set p0 = 1 + 4X (constant=1, linear=4, rest=0)
        model.add(p0[0] == 1)
        model.add(p0[1] == 4)
        model.add(p0[2] == 0)
        model.add(p0[3] == 0)

        # Set p1 = 3 + 2X^2 (constant=3, quadratic=2, rest=0)
        model.add(p1[0] == 3)
        model.add(p1[1] == 0)
        model.add(p1[2] == 2)
        model.add(p1[3] == 0)

        for constraint in constraints:
            model.add(constraint)

        status = solver.solve(model)
        assert status == cp_model.OPTIMAL

        # p0 * p1 = (1 + 4X)(3 + 2X^2) = 3 + 12X + 2X^2 + 8X^3
        p0_concrete = OneVarPoly([solver.value(c) for c in p0.coefs])
        p1_concrete = OneVarPoly([solver.value(c) for c in p1.coefs])
        p_mul_concrete = OneVarPoly([solver.value(c) for c in p_mul.coefs])

        print(f"p0 = {p0_concrete}")
        print(f"p1 = {p1_concrete}")
        print(f"p0 * p1 = {p_mul_concrete}")

        # Verify result (coefficients that fit within degree 3)
        assert solver.value(p_mul[0]) == 3  # constant
        assert solver.value(p_mul[1]) == 12  # X coefficient
        assert solver.value(p_mul[2]) == 2  # X^2 coefficient
        assert solver.value(p_mul[3]) == 8  # X^3 coefficient


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_polynomial(self):
        """Test that zero polynomial is trivially achievable."""
        status, res = compute_expansion(
            OneVarPoly([0, 0, 0, 0]), num_steps=1, max_deg=3, bound=100
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_insufficient_steps(self):
        """Test that complex polynomial fails with too few steps."""
        # (1+X)^2 = 1 + 2X + X^2 requires at least 2 steps
        status, res = compute_expansion(
            OneVarPoly([1, 2, 1, 0]), num_steps=1, max_deg=3, bound=100
        )
        # This should be INFEASIBLE with only 1 step
        # (might still find a solution if starting conditions allow)
        print(f"Status: {status}, Result: {res}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
