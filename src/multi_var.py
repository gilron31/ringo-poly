"""
Unified polynomial optimizer with configurable register models and operations.

This module provides a flexible framework for finding minimal sequences of
arithmetic operations to compute target polynomials from inputs.
"""

from .poly import (
    initialize_multivarpoly_coefs,
    generate_all_keys,
    ConcreteMultiVarPoly,
    SymbolicMultiVarPoly,
)
from .ops import OpContext, AddOp, SubOp, MulOp, IdentityOp, DEFAULT_OPS
from ortools.sat.python import cp_model


# =============================================================================
# Base Optimizer
# =============================================================================


class _BaseOptimizer:
    """Base class with shared optimizer functionality."""

    def __init__(self, vars, deg, bound=10**9):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.vars = vars
        self.deg = deg
        self.bound = bound
        self.num_polys_generated = 0

        # Generate all coefficient keys for this degree
        self.all_coef_keys = generate_all_keys(
            len(self.vars), self.deg, homogenous=False
        )

        # Create operation context for constraint generation
        self.ctx = OpContext(
            model=self.model,
            bound=self.bound,
            all_coef_keys=self.all_coef_keys,
        )

    def _new_poly(self):
        """Create a new symbolic polynomial."""
        name = f"P_{self.num_polys_generated}"
        self.num_polys_generated += 1
        # coefs = initialize_multivarpoly_coefs(
        #     self.model, self.deg, self.vars, self.bound, name
        # )
        return SymbolicMultiVarPoly(
            self.model,
            self.vars,
            self.deg,
            self.bound,
            name_prefix=name,
            solver=self.solver,
        )


# =============================================================================
# Register-File Model Optimizer
# =============================================================================


class PolyOptimizer(_BaseOptimizer):
    """
    Unified polynomial code optimizer using register-file model.

    This optimizer finds minimal sequences of arithmetic operations to compute
    target polynomials from input polynomials using a register-file model where
    each step can read from any register and write to any register.

    Args:
        vars: List of variable names (e.g., ["X", "Y"])
        deg: Maximum polynomial degree
        n_regs: Number of registers
        n_steps: Number of operation steps
        ops: List of operations to use (default: ADD, SUB, MUL)
        bound: Coefficient bound for constraint solver
    """

    def __init__(self, vars, deg, n_regs, n_steps, ops=None, bound=10**9):
        super().__init__(vars, deg, bound)
        self.n_regs = n_regs
        self.n_steps = n_steps

        # Default operations
        self.ops = ops if ops is not None else DEFAULT_OPS

        # Initialize register file: regs[step][reg_idx]
        self.regs = [
            [self._new_poly() for _ in range(self.n_regs)]
            for _ in range(self.n_steps + 1)
        ]

        # Initialize selectors
        self._declare_selectors()
        self._setup_selector_constraints()

    def _declare_selectors(self):
        """Declare selector variables for each step."""
        self.dst_selectors = [
            [
                self.model.new_bool_var(f"step_{step}_dst_{reg}")
                for reg in range(self.n_regs)
            ]
            for step in range(self.n_steps)
        ]
        self.lhs_selectors = [
            [
                self.model.new_bool_var(f"step_{step}_lhs_{reg}")
                for reg in range(self.n_regs)
            ]
            for step in range(self.n_steps)
        ]
        self.rhs_selectors = [
            [
                self.model.new_bool_var(f"step_{step}_rhs_{reg}")
                for reg in range(self.n_regs)
            ]
            for step in range(self.n_steps)
        ]
        self.op_selectors = [
            [
                self.model.new_bool_var(f"step_{step}_op_{op.codename}")
                for op in self.ops
            ]
            for step in range(self.n_steps)
        ]

    def _setup_selector_constraints(self):
        """Set up exactly-one constraints for selectors."""
        for step in range(self.n_steps):
            self.model.add_exactly_one(self.dst_selectors[step])
            self.model.add_exactly_one(self.lhs_selectors[step])
            self.model.add_exactly_one(self.rhs_selectors[step])
            self.model.add_exactly_one(self.op_selectors[step])

    def get_basis(self):
        """Get basis polynomials (one for each variable)."""
        return [
            ConcreteMultiVarPoly(
                self.vars,
                self.deg,
                {tuple(1 if i == k else 0 for k in range(len(self.vars))): 1},
            )
            for i in range(len(self.vars))
        ]

    def find_code(self, inputs, outputs, minimize_mul=None):
        """
        Find code that transforms inputs to outputs.

        Args:
            inputs: List of input ConcreteMultiVarPoly
            outputs: List of target ConcreteMultiVarPoly
            minimize_mul: If set, constrain or minimize multiplication count

        Returns:
            (status, result_string) tuple
        """
        assert len(inputs) <= self.n_regs
        assert len(outputs) <= self.n_regs

        # Set up input constraints
        for reg_idx in range(self.n_regs):
            input_poly = (
                inputs[reg_idx]
                if reg_idx < len(inputs)
                else ConcreteMultiVarPoly(self.vars, self.deg, coefs={})
            )
            self.regs[0][reg_idx].equate_to(input_poly)

        # Set up output constraints (permutation-invariant)
        for i, output in enumerate(outputs):
            output_selector = [
                self.model.new_bool_var(f"output_{i}_reg_{reg}")
                for reg in range(self.n_regs)
            ]
            self.model.add_exactly_one(output_selector)
            for reg_idx in range(self.n_regs):
                self.regs[self.n_steps][reg_idx].equate_to(
                    output, output_selector[reg_idx]
                )

        # Set up operation constraints for each step
        for step in range(self.n_steps):
            for dst_idx in range(self.n_regs):
                # Non-destination registers stay unchanged
                for reg_idx in range(self.n_regs):
                    if reg_idx != dst_idx:
                        self.regs[step + 1][reg_idx].equate_to(
                            self.regs[step][reg_idx], self.dst_selectors[step][dst_idx]
                        )

                # Operation constraints
                for op_idx, op in enumerate(self.ops):
                    for lhs_idx in range(self.n_regs):
                        for rhs_idx in range(self.n_regs):
                            # Symmetry breaking for commutative ops
                            if op.is_commutative and lhs_idx > rhs_idx:
                                self.model.add(False).only_enforce_if(
                                    [
                                        self.lhs_selectors[step][lhs_idx],
                                        self.rhs_selectors[step][rhs_idx],
                                    ]
                                )
                                continue

                            cond = [
                                self.op_selectors[step][op_idx],
                                self.lhs_selectors[step][lhs_idx],
                                self.rhs_selectors[step][rhs_idx],
                                self.dst_selectors[step][dst_idx],
                            ]

                            # Use self-describing op to generate constraints
                            op.apply_constraints(
                                self.ctx,
                                self.regs[step][lhs_idx],
                                self.regs[step][rhs_idx],
                                self.regs[step + 1][dst_idx],
                                cond,
                            )

        # Handle multiplication minimization
        if minimize_mul is not None:
            mul_op_idx = next(
                (i for i, op in enumerate(self.ops) if op.codename == "MUL"), None
            )
            if mul_op_idx is not None:
                mul_count = sum(
                    self.op_selectors[step][mul_op_idx] for step in range(self.n_steps)
                )
                if minimize_mul > 0:
                    self.model.add(mul_count == minimize_mul)
                else:
                    self.model.minimize(mul_count)

        # Solve
        print(self.model.Validate())
        status = self.solver.solve(self.model)

        if status == cp_model.OPTIMAL:
            return status, self.dump_solution()
        else:
            return status, f"UNSAT {status}\n"

    def dump_solution(self):
        """Format the solved state as a string."""
        dst_realizations = [
            next(i for i, s in enumerate(step) if self.solver.value(s))
            for step in self.dst_selectors
        ]
        lhs_realizations = [
            next(i for i, s in enumerate(step) if self.solver.value(s))
            for step in self.lhs_selectors
        ]
        rhs_realizations = [
            next(i for i, s in enumerate(step) if self.solver.value(s))
            for step in self.rhs_selectors
        ]
        op_realizations = [
            next(
                self.ops[i].codename for i, s in enumerate(step) if self.solver.value(s)
            )
            for step in self.op_selectors
        ]
        regs_realization = [[poly.to_concrete() for poly in step] for step in self.regs]

        rv = f"n_steps: {self.n_steps}, n_regs: {self.n_regs}\n"
        for step in range(self.n_steps):
            rv += f"s: {step:2} {regs_realization[step]}\n"
            rv += f"{op_realizations[step]:4} {lhs_realizations[step]:2}, "
            rv += f"{rhs_realizations[step]:2} -> {dst_realizations[step]}\n"
        rv += f"s: {self.n_steps:2} {regs_realization[self.n_steps]}\n"

        return rv

    def assert_solution_valid(self, inputs, outputs):
        """Verify that the solution is valid."""
        for reg_idx in range(self.n_regs):
            expected = (
                inputs[reg_idx]
                if reg_idx < len(inputs)
                else ConcreteMultiVarPoly(self.vars, self.deg, coefs={})
            )
            actual = self.regs[0][reg_idx].to_concrete()
            assert actual == expected, f"Input mismatch at reg {reg_idx}"

        for output in outputs:
            found = any(
                self.regs[self.n_steps][reg_idx].to_concrete() == output
                for reg_idx in range(self.n_regs)
            )
            assert found, f"{output} not in outputs"


# =============================================================================
# Chain Model Optimizer
# =============================================================================


class ChainPolyOptimizer(_BaseOptimizer):
    """
    Polynomial optimizer using chain/list register model.

    Each step appends a new register (DAG-style). Can read from any previous
    register. This model scales better for complex problems like Karatsuba.

    Args:
        vars: List of variable names
        deg: Maximum polynomial degree
        n_steps: Number of operation steps
        ops: List of operations (default: ADD, SUB, IDN, MUL)
        bound: Coefficient bound
    """

    def __init__(self, vars, deg, n_steps, ops=None, bound=10**9):
        super().__init__(vars, deg, bound)
        self.n_steps = n_steps

        # Default ops for chain model includes Identity
        self.ops = ops if ops is not None else [AddOp(), SubOp(), IdentityOp(), MulOp()]

    def find_code(self, inputs, outputs, minimize_mul=None):
        """
        Find code that transforms inputs to outputs.

        Args:
            inputs: List of input polynomials (MultiVarPoly or ConcreteMultiVarPoly)
            outputs: List of target polynomials
            minimize_mul: If set, constrain or minimize multiplication count

        Returns:
            (status, result_string) tuple
        """
        n_inputs = len(inputs)
        n_outputs = len(outputs)

        # Adjust steps to account for final identity ops for outputs
        total_steps = self.n_steps + n_outputs

        # Create registers: inputs + one per step
        self.regs = [self._new_poly() for _ in range(total_steps + n_inputs)]

        # Constrain input registers
        for i, inp in enumerate(inputs):
            self.regs[i].equate_to(inp)

        # Create selectors for each step
        self.op_selectors = [
            [self.model.new_bool_var(f"step_{i}_op_{op.codename}") for op in self.ops]
            for i in range(total_steps)
        ]
        self.lhs_selectors = [
            [self.model.new_bool_var(f"step_{i}_lhs_{j}") for j in range(i + n_inputs)]
            for i in range(total_steps)
        ]
        self.rhs_selectors = [
            [self.model.new_bool_var(f"step_{i}_rhs_{j}") for j in range(i + n_inputs)]
            for i in range(total_steps)
        ]

        # Exactly-one constraints
        for i in range(total_steps):
            self.model.add_exactly_one(self.op_selectors[i])
            self.model.add_exactly_one(self.lhs_selectors[i])
            self.model.add_exactly_one(self.rhs_selectors[i])

        # Operation constraints for each step
        for step in range(total_steps):
            dest_reg = self.regs[n_inputs + step]

            for op_idx, op in enumerate(self.ops):
                for lhs_idx in range(n_inputs + step):
                    for rhs_idx in range(n_inputs + step):
                        # Symmetry breaking: lhs_idx <= rhs_idx for all ops
                        if lhs_idx > rhs_idx:
                            self.model.add(False).only_enforce_if(
                                [
                                    self.lhs_selectors[step][lhs_idx],
                                    self.rhs_selectors[step][rhs_idx],
                                ]
                            )
                            continue

                        cond = [
                            self.op_selectors[step][op_idx],
                            self.lhs_selectors[step][lhs_idx],
                            self.rhs_selectors[step][rhs_idx],
                        ]

                        op.apply_constraints(
                            self.ctx,
                            self.regs[lhs_idx],
                            self.regs[rhs_idx],
                            dest_reg,
                            cond,
                        )

        # Constrain outputs (last n_outputs registers must match targets)
        for i, target in enumerate(outputs):
            self.regs[len(self.regs) - n_outputs + i].equate_to(target)

        # Force last n_outputs steps to use Identity op
        idn_idx = next(
            (i for i, op in enumerate(self.ops) if op.codename == "IDN"), None
        )
        if idn_idx is not None:
            for i in range(self.n_steps, total_steps):
                self.model.add(self.op_selectors[i][idn_idx] == True)

        # Multiplication minimization
        if minimize_mul is not None:
            mul_idx = next(
                (i for i, op in enumerate(self.ops) if op.codename == "MUL"), None
            )
            if mul_idx is not None:
                mul_count = sum(
                    self.op_selectors[step][mul_idx] for step in range(total_steps)
                )
                if minimize_mul > 0:
                    self.model.add(mul_count == minimize_mul)
                else:
                    self.model.minimize(mul_count)

        # Solve
        print(self.model.Validate())
        status = self.solver.solve(self.model)

        if status == cp_model.OPTIMAL:
            return status, self._dump_solution(inputs, total_steps)
        else:
            return status, f"UNSAT {status}\n"

    def _dump_solution(self, inputs, total_steps):
        """Format the solution as a string."""
        rv = ""
        ln = 0

        # Print inputs
        for inp in inputs:
            rv += f"{ln:2d}:               {inp}\n"
            ln += 1

        # Print each step
        for step in range(total_steps):
            op_idx = next(
                i for i, s in enumerate(self.op_selectors[step]) if self.solver.value(s)
            )
            lhs_idx = next(
                i
                for i, s in enumerate(self.lhs_selectors[step])
                if self.solver.value(s)
            )
            rhs_idx = next(
                i
                for i, s in enumerate(self.rhs_selectors[step])
                if self.solver.value(s)
            )

            op = self.ops[op_idx]
            result = self.regs[len(inputs) + step].to_concrete()

            rv += f"{ln:2d}: {op.codename} {lhs_idx}"
            rv += f",{rhs_idx}" if op.op_type == "binary" else "  "
            rv += f" ===> {result}\n"
            ln += 1

        return rv


# =============================================================================
# Convenience function (backward compatible with poly_optimizer2)
# =============================================================================


def compute_multivar_expansion(
    init_polys, vars, targets, num_steps_, max_deg, bound=10**10, minimize_mul=None
):
    """
    Find code to compute targets from init_polys using chain model.

    This is a convenience wrapper around ChainPolyOptimizer for backward
    compatibility with poly_optimizer2.

    Args:
        init_polys: List of input polynomials
        vars: List of variable names
        targets: List of target polynomials
        num_steps_: Number of operation steps
        max_deg: Maximum polynomial degree
        bound: Coefficient bound
        minimize_mul: If set, constrain or minimize multiplication count

    Returns:
        (status, result_string) tuple
    """
    opt = ChainPolyOptimizer(vars, max_deg, num_steps_, bound=bound)
    return opt.find_code(init_polys, targets, minimize_mul=minimize_mul)
