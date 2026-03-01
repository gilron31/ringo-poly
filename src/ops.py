"""Self-describing operation classes for polynomial optimizers.

Each operation knows how to generate its own constraints, eliminating
coupling between operation types and optimizer implementations.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple
from ortools.sat.python import cp_model

from .poly import add_keys


@dataclass
class OpContext:
    """Context passed to operations for constraint generation."""
    model: cp_model.CpModel
    bound: int
    all_coef_keys: List[Tuple[int, ...]]

    def add_constraint(self, constraint, cond=None):
        """Add a constraint, optionally conditional."""
        if cond is None:
            self.model.add(constraint)
        else:
            self.model.add(constraint).only_enforce_if(cond)


class Op(ABC):
    """Base class for all operations."""
    codename: str
    op_type: str  # "unary" or "binary"
    is_commutative: bool = False

    @abstractmethod
    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """Add constraints to make out = op(lhs, rhs).

        Args:
            ctx: Operation context with model, bound, and coefficient keys
            lhs: Left-hand side polynomial (symbolic)
            rhs: Right-hand side polynomial (symbolic), ignored for unary ops
            out: Output polynomial (symbolic)
            cond: Optional condition for the constraints
        """
        pass


class UnaryOp(Op):
    """Base class for unary operations."""

    def __init__(self):
        self.op_type = "unary"


class BinaryOp(Op):
    """Base class for binary operations."""

    def __init__(self):
        self.op_type = "binary"


class IdentityOp(UnaryOp):
    """Identity operation (copy)."""
    codename = "IDN"
    is_commutative = True

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs (copy)."""
        for k in ctx.all_coef_keys:
            ctx.add_constraint(lhs.coefs[k] == out.coefs[k], cond)


class AddOp(BinaryOp):
    """Addition operation."""
    codename = "ADD"
    is_commutative = True

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs + rhs."""
        for k in ctx.all_coef_keys:
            ctx.add_constraint(
                lhs.coefs[k] + rhs.coefs[k] == out.coefs[k], cond
            )


class SubOp(BinaryOp):
    """Subtraction operation."""
    codename = "SUB"
    is_commutative = False

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs - rhs."""
        for k in ctx.all_coef_keys:
            ctx.add_constraint(
                lhs.coefs[k] - rhs.coefs[k] == out.coefs[k], cond
            )


class MulOp(BinaryOp):
    """Multiplication operation."""
    codename = "MUL"
    is_commutative = True

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs * rhs.

        Creates intermediate variables for each term product and sums them
        into the appropriate output coefficient.
        """
        # Create intermediate variables for each pair of coefficient products
        multiplicands = {
            (k_lhs, k_rhs): ctx.model.new_int_var(
                -ctx.bound,
                ctx.bound,
                f"{lhs.coefs[k_lhs].name}_x_{rhs.coefs[k_rhs].name}",
            )
            for k_lhs in lhs.coefs.keys()
            for k_rhs in rhs.coefs.keys()
        }

        # Map each target key to the pairs that contribute to it
        target_coef_map = defaultdict(list)
        for k_lhs, k_rhs in multiplicands.keys():
            # Add multiplication equality constraint
            ctx.model.add_multiplication_equality(
                multiplicands[(k_lhs, k_rhs)],
                [lhs.coefs[k_lhs], rhs.coefs[k_rhs]],
            )
            target_coef_map[add_keys(k_lhs, k_rhs)].append((k_rhs, k_lhs))

        # Sum contributions for each output coefficient
        for k, pairs in target_coef_map.items():
            contribution = sum(multiplicands[(k_lhs, k_rhs)] for k_lhs, k_rhs in pairs)
            if k in out.coefs:
                ctx.add_constraint(out.coefs[k] == contribution, cond)
            else:
                # Coefficient outside degree bound must be zero
                ctx.add_constraint(0 == contribution, cond)


class IncOp(UnaryOp):
    """Increment by 1 (add constant 1)."""
    codename = "INC"
    is_commutative = False

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs + 1."""
        for k in ctx.all_coef_keys:
            if all(ki == 0 for ki in k):  # Constant term
                ctx.add_constraint(lhs.coefs[k] + 1 == out.coefs[k], cond)
            else:
                ctx.add_constraint(lhs.coefs[k] == out.coefs[k], cond)


class DecOp(UnaryOp):
    """Decrement by 1 (subtract constant 1)."""
    codename = "DEC"
    is_commutative = False

    def __init__(self):
        super().__init__()

    def apply_constraints(self, ctx: OpContext, lhs, rhs, out, cond=None):
        """out = lhs - 1."""
        for k in ctx.all_coef_keys:
            if all(ki == 0 for ki in k):  # Constant term
                ctx.add_constraint(lhs.coefs[k] - 1 == out.coefs[k], cond)
            else:
                ctx.add_constraint(lhs.coefs[k] == out.coefs[k], cond)


# Default operation set for multi-variable polynomial optimization
DEFAULT_OPS = [AddOp(), SubOp(), MulOp()]
