"""Tests for polyopt.poly - polynomial classes and utilities.

Test organization:
1. OneVarPoly - single variable polynomial
2. MultiVarPoly - multi-variable polynomial base class
3. ConcreteMultiVarPoly - concrete coefficients
4. SymbolicMultiVarPoly - OR-Tools IntVar coefficients
5. Utility functions
"""

import pytest
from ortools.sat.python import cp_model
from src import (
    OneVarPoly,
    MultiVarPoly,
    ConcreteMultiVarPoly,
    SymbolicMultiVarPoly,
    add_keys,
    generate_all_keys,
    initialize_poly_coefs,
    initialize_multivarpoly_coefs,
)
from src.poly import clear_zero_coefs


# =============================================================================
# OneVarPoly Tests
# =============================================================================


class TestOneVarPoly:
    """Tests for single-variable polynomial class."""

    def test_creation(self):
        """Test basic polynomial creation."""
        p = OneVarPoly([1, 2, 3])
        assert len(p) == 3
        assert p[0] == 1
        assert p[1] == 2
        assert p[2] == 3

    def test_repr(self):
        """Test string representation."""
        assert str(OneVarPoly([3])) == "3"
        assert str(OneVarPoly([0, 1])) == "X"
        assert str(OneVarPoly([1, 1])) == "X + 1"
        assert str(OneVarPoly([1, 2, 3])) == "3X^2 + 2X + 1"
        assert str(OneVarPoly([0, 0, 0])) == "0"

    def test_addition(self):
        """Test polynomial addition."""
        p0 = OneVarPoly([1, 2, 3])
        p1 = OneVarPoly([4, 5])
        result = p0 + p1
        assert result == OneVarPoly([5, 7, 3])

    def test_addition_different_degrees(self):
        """Test addition with different degree polynomials."""
        p0 = OneVarPoly([1])
        p1 = OneVarPoly([0, 0, 5])
        result = p0 + p1
        assert result == OneVarPoly([1, 0, 5])

    def test_subtraction(self):
        """Test polynomial subtraction."""
        p0 = OneVarPoly([5, 7, 3])
        p1 = OneVarPoly([4, 5])
        result = p0 - p1
        assert result == OneVarPoly([1, 2, 3])

    def test_multiplication(self):
        """Test polynomial multiplication."""
        p0 = OneVarPoly([1, 2, 3])  # 1 + 2X + 3X^2
        p1 = OneVarPoly([4, 5])  # 4 + 5X
        result = p0 * p1
        # (1 + 2X + 3X^2)(4 + 5X) = 4 + 5X + 8X + 10X^2 + 12X^2 + 15X^3
        #                        = 4 + 13X + 22X^2 + 15X^3
        assert result == OneVarPoly([4, 13, 22, 15])

    def test_multiplication_empty(self):
        """Test multiplication with empty polynomial."""
        p0 = OneVarPoly([1, 2])
        p1 = OneVarPoly([])
        result = p0 * p1
        assert result == OneVarPoly([])

    def test_canonize_removes_trailing_zeros(self):
        """Test canonization removes trailing zeros."""
        p = OneVarPoly([1, 2, 3, 0, 0])
        canonical = p.canonize()
        assert canonical.coefs == [1, 2, 3]

    def test_canonize_preserves_internal_zeros(self):
        """Test canonization keeps internal zeros."""
        p = OneVarPoly([1, 0, 3, 0, 5])
        canonical = p.canonize()
        assert canonical.coefs == [1, 0, 3, 0, 5]

    def test_equality(self):
        """Test polynomial equality."""
        p0 = OneVarPoly([1, 2, 3])
        p1 = OneVarPoly([1, 2, 3])
        p2 = OneVarPoly([1, 2, 3, 0, 0])  # Same after canonization
        assert p0 == p1
        assert p0 == p2

    def test_equality_different(self):
        """Test inequality of different polynomials."""
        p0 = OneVarPoly([1, 2, 3])
        p1 = OneVarPoly([1, 2, 4])
        assert not (p0 == p1)


# =============================================================================
# MultiVarPoly Tests
# =============================================================================


class TestMultiVarPoly:
    """Tests for multi-variable polynomial base class."""

    def test_creation(self):
        """Test basic creation."""
        p = MultiVarPoly(["X", "Y"], {(1, 0): 3, (0, 1): 2})
        assert p.vars == ["X", "Y"]
        assert p.n_vars == 2
        assert p[(1, 0)] == 3
        assert p[(0, 1)] == 2

    def test_vars_must_be_sorted(self):
        """Test that variables must be alphabetically sorted."""
        with pytest.raises(AssertionError):
            MultiVarPoly(["Y", "X"], {})

    def test_deg_property_explicit(self):
        """Test degree property with explicit degree."""
        p = MultiVarPoly(["X", "Y"], {(1, 0): 1}, deg=5)
        assert p.deg == 5

    def test_deg_property_computed(self):
        """Test degree property computed from coefficients."""
        p = MultiVarPoly(["X", "Y"], {(2, 1): 1, (1, 0): 1})
        assert p.deg == 3  # max(2+1, 1+0) = 3

    def test_deg_property_empty(self):
        """Test degree property for empty polynomial."""
        p = MultiVarPoly(["X", "Y"], {})
        assert p.deg == 0

    def test_repr_single_term(self):
        """Test string representation with single term."""
        p = MultiVarPoly(["X", "Y"], {(2, 1): 3})
        assert str(p) == "3X^2Y"

    def test_repr_multiple_terms(self):
        """Test string representation with multiple terms."""
        p = MultiVarPoly(["X", "Y"], {(1, 0): 1, (0, 1): 1})
        result = str(p)
        assert "X" in result and "Y" in result

    def test_repr_zero(self):
        """Test string representation of zero polynomial."""
        p = MultiVarPoly(["X", "Y"], {(1, 0): 0})
        assert str(p) == "0"

    def test_addition(self):
        """Test polynomial addition."""
        p0 = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 3})
        p1 = MultiVarPoly(["X", "Y"], {(1, 0): 1, (0, 2): 4})
        result = p0 + p1
        assert result.coefs[(1, 0)] == 3
        assert result.coefs[(0, 1)] == 3
        assert result.coefs[(0, 2)] == 4

    def test_subtraction(self):
        """Test polynomial subtraction."""
        p0 = MultiVarPoly(["X", "Y"], {(1, 0): 5, (0, 1): 3})
        p1 = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 1})
        result = p0 - p1
        assert result.coefs[(1, 0)] == 3
        assert result.coefs[(0, 1)] == 2

    def test_multiplication_by_scalar(self):
        """Test multiplication by integer."""
        p = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 3})
        result = p * 5
        assert result.coefs[(1, 0)] == 10
        assert result.coefs[(0, 1)] == 15

    def test_multiplication_by_poly(self):
        """Test polynomial multiplication."""
        x = MultiVarPoly(["X", "Y"], {(1, 0): 1})
        y = MultiVarPoly(["X", "Y"], {(0, 1): 1})
        result = x * y
        assert result.coefs[(1, 1)] == 1

    def test_foil_expansion(self):
        """Test (x+y)^2 = x^2 + 2xy + y^2."""
        x = MultiVarPoly(["X", "Y"], {(1, 0): 1})
        y = MultiVarPoly(["X", "Y"], {(0, 1): 1})
        x_plus_y = x + y
        result = x_plus_y * x_plus_y
        assert result.coefs.get((2, 0), 0) == 1
        assert result.coefs.get((1, 1), 0) == 2
        assert result.coefs.get((0, 2), 0) == 1

    def test_equality(self):
        """Test polynomial equality."""
        p0 = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 3})
        p1 = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 3})
        assert p0 == p1

    def test_equality_ignores_zero_coefs(self):
        """Test equality ignores zero coefficients."""
        p0 = MultiVarPoly(["X", "Y"], {(1, 0): 2})
        p1 = MultiVarPoly(["X", "Y"], {(1, 0): 2, (0, 1): 0})
        assert p0 == p1


# =============================================================================
# ConcreteMultiVarPoly Tests
# =============================================================================


class TestConcreteMultiVarPoly:
    """Tests for concrete multi-variable polynomial."""

    def test_creation(self):
        """Test creation with explicit degree."""
        p = ConcreteMultiVarPoly(["X", "Y"], deg=3, coefs={(1, 0): 5})
        assert p._deg == 3
        assert p[(1, 0)] == 5

    def test_addition_preserves_type(self):
        """Test addition returns ConcreteMultiVarPoly."""
        p0 = ConcreteMultiVarPoly(["X", "Y"], 2, {(1, 0): 1})
        p1 = ConcreteMultiVarPoly(["X", "Y"], 2, {(0, 1): 1})
        result = p0 + p1
        assert isinstance(result, ConcreteMultiVarPoly)

    def test_subtraction_preserves_type(self):
        """Test subtraction returns ConcreteMultiVarPoly."""
        p0 = ConcreteMultiVarPoly(["X", "Y"], 2, {(1, 0): 5})
        p1 = ConcreteMultiVarPoly(["X", "Y"], 2, {(1, 0): 2})
        result = p0 - p1
        assert isinstance(result, ConcreteMultiVarPoly)
        assert result[(1, 0)] == 3

    def test_multiplication_by_int(self):
        """Test multiplication by integer."""
        p = ConcreteMultiVarPoly(["X", "Y"], 2, {(1, 0): 3})
        result = p * 4
        assert isinstance(result, ConcreteMultiVarPoly)
        assert result[(1, 0)] == 12

    def test_multiplication_by_poly(self):
        """Test polynomial multiplication."""
        x = ConcreteMultiVarPoly(["X", "Y"], 2, {(1, 0): 1})
        y = ConcreteMultiVarPoly(["X", "Y"], 2, {(0, 1): 1})
        result = x * y
        assert isinstance(result, ConcreteMultiVarPoly)
        assert result[(1, 1)] == 1

    def test_karatsuba_terms(self):
        """Test building Karatsuba multiplication terms."""
        vars = ["a0", "a1", "b0", "b1"]
        a0 = ConcreteMultiVarPoly(vars, 2, {(1, 0, 0, 0): 1})
        a1 = ConcreteMultiVarPoly(vars, 2, {(0, 1, 0, 0): 1})
        b0 = ConcreteMultiVarPoly(vars, 2, {(0, 0, 1, 0): 1})
        b1 = ConcreteMultiVarPoly(vars, 2, {(0, 0, 0, 1): 1})

        # Cross term: a1*b0 + a0*b1
        cross = a1 * b0 + a0 * b1
        assert cross[(0, 1, 1, 0)] == 1  # a1*b0
        assert cross[(1, 0, 0, 1)] == 1  # a0*b1


# =============================================================================
# SymbolicMultiVarPoly Tests
# =============================================================================


class TestSymbolicMultiVarPoly:
    """Tests for symbolic multi-variable polynomial with OR-Tools IntVars."""

    def test_creation(self):
        """Test creation with OR-Tools model."""
        model = cp_model.CpModel()
        p = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=2, bound=100, name_prefix="test"
        )
        assert p.vars == ["X", "Y"]
        assert p._deg == 2
        # Should have coefs for all monomials up to degree 2
        expected_keys = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for key in expected_keys:
            assert key in p.coefs

    def test_equate_to_concrete(self):
        """Test equating symbolic poly to concrete values."""
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p_sym = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=2, bound=100, name_prefix="test"
        )
        p_concrete = MultiVarPoly(["X", "Y"], {(1, 0): 5, (0, 1): 3})

        p_sym.equate_to(p_concrete)
        status = solver.solve(model)

        assert status == cp_model.OPTIMAL
        assert solver.value(p_sym.coefs[(1, 0)]) == 5
        assert solver.value(p_sym.coefs[(0, 1)]) == 3
        assert solver.value(p_sym.coefs[(0, 0)]) == 0

    def test_to_concrete(self):
        """Test conversion to ConcreteMultiVarPoly."""
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p_sym = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=1, bound=100, name_prefix="test", solver=solver
        )
        p_target = MultiVarPoly(["X", "Y"], {(1, 0): 7})

        p_sym.equate_to(p_target)
        solver.solve(model)

        concrete = p_sym.to_concrete()
        assert isinstance(concrete, ConcreteMultiVarPoly)
        assert concrete[(1, 0)] == 7

    def test_eq_adds_constraints(self):
        """Test that == operator adds constraints."""
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p_sym = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=1, bound=100, name_prefix="test", solver=solver
        )
        p_target = MultiVarPoly(["X", "Y"], {(1, 0): 9, (0, 1): 4})

        p_sym == p_target  # This adds constraints
        solver.solve(model)

        assert solver.value(p_sym.coefs[(1, 0)]) == 9
        assert solver.value(p_sym.coefs[(0, 1)]) == 4


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_add_keys(self):
        """Test element-wise tuple addition."""
        assert add_keys((1, 2, 3), (4, 5, 6)) == (5, 7, 9)
        assert add_keys((0, 0), (1, 1)) == (1, 1)
        assert add_keys((), ()) == ()

    def test_generate_all_keys_homogenous(self):
        """Test generating homogenous keys of specific degree."""
        keys = generate_all_keys(2, 2, homogenous=True)
        # For 2 vars, degree 2: (2,0), (1,1), (0,2)
        assert (2, 0) in keys
        assert (1, 1) in keys
        assert (0, 2) in keys
        assert len(keys) == 3

    def test_generate_all_keys_non_homogenous(self):
        """Test generating all keys up to degree."""
        keys = generate_all_keys(2, 2, homogenous=False)
        # For 2 vars, up to degree 2: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
        expected = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for key in expected:
            assert key in keys

    def test_clear_zero_coefs(self):
        """Test removing zero coefficients."""
        coefs = {(1, 0): 5, (0, 1): 0, (2, 0): 3, (0, 0): 0}
        result = clear_zero_coefs(coefs)
        assert result == {(1, 0): 5, (2, 0): 3}

    def test_initialize_poly_coefs(self):
        """Test creating OR-Tools IntVars for single-var poly."""
        model = cp_model.CpModel()
        coefs = initialize_poly_coefs(model, deg=3, bound=100, name_prefix="p")
        assert len(coefs) == 4  # degree 0, 1, 2, 3

    def test_initialize_multivarpoly_coefs(self):
        """Test creating OR-Tools IntVars for multi-var poly."""
        model = cp_model.CpModel()
        coefs = initialize_multivarpoly_coefs(
            model, deg=2, vars=["X", "Y"], bound=100, name_prefix="p"
        )
        # Should have keys for all monomials up to degree 2
        expected_keys = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        for key in expected_keys:
            assert key in coefs


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_symbolic_solve_quadratic(self):
        """Test solving for coefficients of x^2 + 2xy + y^2."""
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p_sym = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=2, bound=100, name_prefix="q", solver=solver
        )

        # Build target: (x+y)^2 = x^2 + 2xy + y^2
        x = MultiVarPoly(["X", "Y"], {(1, 0): 1})
        y = MultiVarPoly(["X", "Y"], {(0, 1): 1})
        target = (x + y) * (x + y)

        p_sym.equate_to(target)
        status = solver.solve(model)

        assert status == cp_model.OPTIMAL
        concrete = p_sym.to_concrete()
        assert concrete[(2, 0)] == 1
        assert concrete[(1, 1)] == 2
        assert concrete[(0, 2)] == 1

    def test_four_variable_polynomial(self):
        """Test polynomial with 4 variables."""
        vars = ["W", "X", "Y", "Z"]
        p = ConcreteMultiVarPoly(
            vars,
            2,
            {
                (1, 0, 0, 0): 1,  # W
                (0, 1, 0, 0): 2,  # 2X
                (0, 0, 1, 0): 3,  # 3Y
                (0, 0, 0, 1): 4,  # 4Z
            },
        )
        result = p * 2
        assert result[(1, 0, 0, 0)] == 2
        assert result[(0, 1, 0, 0)] == 4
        assert result[(0, 0, 1, 0)] == 6
        assert result[(0, 0, 0, 1)] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
