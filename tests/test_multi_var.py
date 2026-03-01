"""Tests for multi-variable polynomial optimizers.

Two register models are available:
1. Register file model (PolyOptimizer) - fixed registers, one modified per step
   - Good for simple problems
   - From polyopt.multi_var

2. Chain/list model (compute_multivar_expansion) - each step appends a register
   - Scales better for complex problems like Karatsuba
   - From polyopt.multi_var

Test organization:
1. Register File Model - basic tests
2. Register File Model - multi-output tests
3. Chain Model - basic tests
4. Chain Model - algorithm synthesis (Karatsuba, etc.)
"""

import pytest
from ortools.sat.python import cp_model
from src import (
    PolyOptimizer,
    ChainPolyOptimizer,
    compute_multivar_expansion,
    ConcreteMultiVarPoly,
    MultiVarPoly,
    SymbolicMultiVarPoly,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_basis(vars, deg):
    """Create basis polynomials (one for each variable)."""
    return [
        ConcreteMultiVarPoly(
            vars, deg, {tuple(1 if i == k else 0 for k in range(len(vars))): 1}
        )
        for i in range(len(vars))
    ]


# =============================================================================
# Register File Model - Basic Tests
# =============================================================================


class TestRegisterFileBasic:
    """Basic tests for the register file model (PolyOptimizer)."""

    def test_simple_addition(self):
        """Test simple addition: x + y."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 2, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 2, {(0, 1): 1})
        target = x + y

        opt = PolyOptimizer(vars, deg=2, n_regs=3, n_steps=1)
        status, res = opt.find_code([x, y], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_simple_subtraction(self):
        """Test simple subtraction: x - y."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 2, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 2, {(0, 1): 1})
        target = x - y

        opt = PolyOptimizer(vars, deg=2, n_regs=3, n_steps=1)
        status, res = opt.find_code([x, y], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_simple_multiplication(self):
        """Test simple multiplication: x * y."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 2, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 2, {(0, 1): 1})
        target = x * y

        opt = PolyOptimizer(vars, deg=2, n_regs=3, n_steps=1)
        status, res = opt.find_code([x, y], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_square(self):
        """Test squaring: x^2."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 3, {(1, 0): 1})
        target = x * x

        opt = PolyOptimizer(vars, deg=3, n_regs=3, n_steps=1)
        status, res = opt.find_code([x], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_x_squared_plus_y(self):
        """Test x^2 + y with multiple steps."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 3, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 3, {(0, 1): 1})
        target = x * x + y

        opt = PolyOptimizer(vars, deg=3, n_regs=4, n_steps=3)
        status, res = opt.find_code([x, y], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_foil_expansion(self):
        """Test (x + y)^2 = x^2 + 2xy + y^2 with 2 steps."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 3, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 3, {(0, 1): 1})
        target = x * x + x * y * 2 + y * y

        opt = PolyOptimizer(vars, deg=3, n_regs=4, n_steps=2)
        status, res = opt.find_code([x, y], [target])
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Register File Model - Multi-Output Tests
# =============================================================================


class TestRegisterFileMultiOutput:
    """Tests for multiple outputs with register file model."""

    def test_two_outputs(self):
        """Test computing x^2 + y^2 and 2xy simultaneously."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 3, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 3, {(0, 1): 1})

        targets = [x * x + y * y, x * y * 2]

        opt = PolyOptimizer(vars, deg=3, n_regs=4, n_steps=5)
        status, res = opt.find_code([x, y], targets)
        assert status == cp_model.OPTIMAL
        print(res)

    def test_three_outputs(self):
        """Test computing three related outputs."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 2, {(1, 0): 1})
        y = ConcreteMultiVarPoly(vars, 2, {(0, 1): 1})

        targets = [x + y, x - y, x * y]

        opt = PolyOptimizer(vars, deg=2, n_regs=5, n_steps=3)
        status, res = opt.find_code([x, y], targets)
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Register File Model - get_basis Tests
# =============================================================================


class TestRegisterFileBasis:
    """Tests for the get_basis helper method."""

    def test_get_basis_two_vars(self):
        """Test get_basis with 2 variables."""
        vars = ["X", "Y"]
        opt = PolyOptimizer(vars, deg=2, n_regs=3, n_steps=1)
        basis = opt.get_basis()

        assert len(basis) == 2
        assert basis[0][(1, 0)] == 1  # X
        assert basis[1][(0, 1)] == 1  # Y

    def test_get_basis_four_vars(self):
        """Test get_basis with 4 variables."""
        vars = ["A", "B", "C", "D"]
        opt = PolyOptimizer(vars, deg=2, n_regs=5, n_steps=1)
        basis = opt.get_basis()

        assert len(basis) == 4
        assert basis[0][(1, 0, 0, 0)] == 1  # A
        assert basis[1][(0, 1, 0, 0)] == 1  # B
        assert basis[2][(0, 0, 1, 0)] == 1  # C
        assert basis[3][(0, 0, 0, 1)] == 1  # D


# =============================================================================
# Chain Model - Basic Tests
# =============================================================================


class TestChainModelBasic:
    """Basic tests for the chain/list model (compute_multivar_expansion)."""

    def test_simple_product(self):
        """Test simple product with chain model."""
        vars = ["X", "Y"]
        x = MultiVarPoly(vars, {(1, 0): 1})
        y = MultiVarPoly(vars, {(0, 1): 1})

        target = x * y

        status, res = compute_multivar_expansion(
            [x, y], vars, [target], num_steps_=1, max_deg=2
        )
        assert status == cp_model.OPTIMAL
        print(res)

    def test_foil_chain(self):
        """Test (x+y)^2 with chain model."""
        vars = ["X", "Y"]
        x = MultiVarPoly(vars, {(1, 0): 1})
        y = MultiVarPoly(vars, {(0, 1): 1})

        target = (x + y) * (x + y)  # x^2 + 2xy + y^2

        status, res = compute_multivar_expansion(
            [x, y], vars, [target], num_steps_=3, max_deg=2
        )
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Chain Model - Algorithm Synthesis
# =============================================================================


class TestAlgorithmSynthesis:
    """Tests for synthesizing known algorithms."""

    def test_karatsuba_multiplication(self):
        """Test finding Karatsuba multiplication with only 3 multiplications.

        Karatsuba computes (a0 + a1*B)(b0 + b1*B) using only 3 multiplications
        instead of the naive 4, by computing:
        - z0 = a0*b0 (low)
        - z2 = a1*b1 (high)
        - z1 = (a0+a1)(b0+b1) - z0 - z2 (middle)
        """
        vars = ["a0", "a1", "b0", "b1"]
        a0 = MultiVarPoly(vars, {(1, 0, 0, 0): 1})
        a1 = MultiVarPoly(vars, {(0, 1, 0, 0): 1})
        b0 = MultiVarPoly(vars, {(0, 0, 1, 0): 1})
        b1 = MultiVarPoly(vars, {(0, 0, 0, 1): 1})

        # Targets: cross term, low product, high product
        targets = [a1 * b0 + b1 * a0, a0 * b0, a1 * b1]

        status, res = compute_multivar_expansion(
            [a0, a1, b0, b1], vars, targets, num_steps_=7, max_deg=2, minimize_mul=3
        )
        assert status == cp_model.OPTIMAL
        print(res)

    @pytest.mark.manual
    def test_complex_multiplication(self):
        """Test complex number multiplication: (a0+a1*i)(b0+b1*i).

        Complex multiplication normally requires 4 real multiplications,
        but can be done with 3 using Karatsuba-like approach:
        - Real part: a0*b0 - a1*b1
        - Imag part: (a0+a1)(b0+b1) - a0*b0 - a1*b1 = a0*b1 + a1*b0

        This test is slow (~3+ minutes).
        """
        vars = ["a0", "a1", "b0", "b1"]
        a0 = MultiVarPoly(vars, {(1, 0, 0, 0): 1})
        a1 = MultiVarPoly(vars, {(0, 1, 0, 0): 1})
        b0 = MultiVarPoly(vars, {(0, 0, 1, 0): 1})
        b1 = MultiVarPoly(vars, {(0, 0, 0, 1): 1})

        # Real part: a0*b0 - a1*b1
        # Imag part: a0*b1 + a1*b0
        targets = [a0 * b0 - a1 * b1, a0 * b1 + a1 * b0]

        status, res = compute_multivar_expansion(
            [a0, a1, b0, b1], vars, targets, num_steps_=8, max_deg=2, minimize_mul=3
        )
        assert status == cp_model.OPTIMAL
        print(res)


# =============================================================================
# Symbolic Polynomial Tests
# =============================================================================


class TestSymbolicPolynomial:
    """Tests for SymbolicMultiVarPoly integration."""

    def test_symbolic_constraint_solving(self):
        """Test solving for symbolic polynomial coefficients."""
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        p_sym = SymbolicMultiVarPoly(
            model, ["X", "Y"], max_deg=3, bound=10, name_prefix="test", solver=solver
        )
        p_target = MultiVarPoly(["X", "Y"], {(1, 0): 3, (2, 1): -1, (0, 2): 2})

        p_sym == p_target  # Adds constraints

        status = solver.solve(model)
        assert status == cp_model.OPTIMAL

        # Verify solved values
        assert solver.value(p_sym.coefs[(1, 0)]) == 3
        assert solver.value(p_sym.coefs[(2, 1)]) == -1
        assert solver.value(p_sym.coefs[(0, 2)]) == 2
        assert solver.value(p_sym.coefs[(0, 0)]) == 0

        print(f"Symbolic: {p_sym}")
        print(f"Target: {p_target}")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_identity_operation(self):
        """Test that identity (no-op) works."""
        vars = ["X", "Y"]
        x = ConcreteMultiVarPoly(vars, 2, {(1, 0): 1})

        # Output = input (should be achievable with just copying)
        opt = PolyOptimizer(vars, deg=2, n_regs=2, n_steps=1)
        status, res = opt.find_code([x], [x])
        assert status == cp_model.OPTIMAL
        print(res)

    def test_single_variable(self):
        """Test with single variable polynomial."""
        vars = ["X"]
        x = ConcreteMultiVarPoly(vars, 2, {(1,): 1})
        target = x * x  # x^2

        opt = PolyOptimizer(vars, deg=2, n_regs=2, n_steps=1)
        status, res = opt.find_code([x], [target])
        assert status == cp_model.OPTIMAL
        print(res)

    @pytest.mark.skip(reason="Edge case with 0 variables - not supported yet")
    def test_constant_only(self):
        """Test with no variables (constants only)."""
        vars = []
        one = MultiVarPoly(vars, {(): 1})
        target = MultiVarPoly(vars, {(): 123454})

        status, res = compute_multivar_expansion(
            [one], vars, [target], num_steps_=13, max_deg=0, bound=200000
        )
        assert status == cp_model.OPTIMAL
        print(res)
