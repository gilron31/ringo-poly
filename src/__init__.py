"""
Polynomial Optimizer Package

This package provides tools for finding minimal sequences of arithmetic
operations to compute target polynomials from inputs using constraint
programming (Google OR-Tools CP-SAT solver).

Main components:
- Polynomial classes: OneVarPoly, MultiVarPoly, ConcreteMultiVarPoly, SymbolicMultiVarPoly
- Operations: AddOp, SubOp, MulOp, IdentityOp
- Optimizers: PolyOptimizer (register-file), ChainPolyOptimizer (chain/DAG)
"""

# Polynomial classes
from .poly import (
    OneVarPoly,
    MultiVarPoly,
    ConcreteMultiVarPoly,
    SymbolicMultiVarPoly,
    # Utility functions
    generate_all_keys,
    add_keys,
    initialize_poly_coefs,
    initialize_multivarpoly_coefs,
    assert_poly_eq,
    mul_polys_enforce_no_ovf,
    mul_multivar_polys_enforce_no_ovf,
)

# Operations
from .ops import (
    Op,
    UnaryOp,
    BinaryOp,
    AddOp,
    SubOp,
    MulOp,
    IdentityOp,
    IncOp,
    DecOp,
    OpContext,
    DEFAULT_OPS,
)

# Single-variable optimizer
from .single_var import compute_expansion

# Multi-variable optimizers
from .multi_var import (
    PolyOptimizer,
    ChainPolyOptimizer,
    compute_multivar_expansion,
)

__all__ = [
    # Polynomial classes
    "OneVarPoly",
    "MultiVarPoly",
    "ConcreteMultiVarPoly",
    "SymbolicMultiVarPoly",
    # Utility functions
    "generate_all_keys",
    "add_keys",
    "initialize_poly_coefs",
    "initialize_multivarpoly_coefs",
    "assert_poly_eq",
    "mul_polys_enforce_no_ovf",
    "mul_multivar_polys_enforce_no_ovf",
    # Operations
    "Op",
    "UnaryOp",
    "BinaryOp",
    "AddOp",
    "SubOp",
    "MulOp",
    "IdentityOp",
    "IncOp",
    "DecOp",
    "OpContext",
    "DEFAULT_OPS",
    # Optimizers
    "compute_expansion",
    "PolyOptimizer",
    "ChainPolyOptimizer",
    "compute_multivar_expansion",
]
