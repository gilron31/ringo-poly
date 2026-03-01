# ringo-poly

A constraint programming tool that synthesizes minimal sequences of arithmetic operations to evaluate target polynomials. Given a set of input polynomials, it finds the shortest sequence of additions, subtractions, and multiplications that produces the desired outputs — using Google OR-Tools CP-SAT under the hood.

The primary use-case is **hardware-aware polynomial evaluation**: for instance, finding an algorithm that computes a set of polynomial outputs using the fewest multiplications (expensive on FPGAs or custom silicon), a problem equivalent to algorithm synthesis for operations like [Karatsuba multiplication](https://en.wikipedia.org/wiki/Karatsuba_algorithm).

## Installation

```bash
pip install ortools
```

## Polynomial Types

The polynomial classes can be used standalone for arithmetic, with no solver involved.

### `OneVarPoly`

Single-variable polynomial. Coefficients are a list `[c0, c1, c2, ...]` representing `c0 + c1*X + c2*X^2 + ...`.

```python
from src import OneVarPoly

p = OneVarPoly([1, 2, 3])   # 1 + 2X + 3X^2
q = OneVarPoly([1, 1])      # 1 + X

print(p + q)   # 3X^2 + 3X + 2
print(p * q)   # 3X^3 + 5X^2 + 3X + 1
print(p - q)   # 3X^2 + X
```

### `MultiVarPoly`

Multi-variable polynomial. Coefficients are a dict mapping exponent tuples to integer values. For `vars = ["X", "Y"]`, the key `(2, 1)` represents `X^2 * Y`.

Variable names must be alphabetically sorted.

```python
from src import MultiVarPoly

x = MultiVarPoly(["X", "Y"], {(1, 0): 1})   # X
y = MultiVarPoly(["X", "Y"], {(0, 1): 1})   # Y

print(x * y)           # XY        -> {(1,1): 1}
print((x + y) * 3)     # 3X + 3Y
p = (x + y) * (x + y)  # X^2 + 2XY + Y^2
```

### `ConcreteMultiVarPoly`

A `MultiVarPoly` subclass that carries an explicit degree bound. **Required when passing inputs to a solver** — the degree bound is used to enumerate all coefficient slots.

```python
from src import ConcreteMultiVarPoly

vars = ["X", "Y"]
x = ConcreteMultiVarPoly(vars, deg=2, coefs={(1, 0): 1})   # X, degree ≤ 2
y = ConcreteMultiVarPoly(vars, deg=2, coefs={(0, 1): 1})   # Y, degree ≤ 2

target = x + y    # ConcreteMultiVarPoly representing X + Y
product = x * y   # ConcreteMultiVarPoly representing XY
```

## Solvers

### Register-file solver (`PolyOptimizer`)

Models computation as a fixed bank of `n_regs` registers. Each of `n_steps` steps reads two registers, applies one operation, and writes the result to a destination register. All other registers are unchanged.

Best for simple problems where the number of registers and steps are known.

```python
from src import PolyOptimizer, ConcreteMultiVarPoly

vars = ["X", "Y"]
x = ConcreteMultiVarPoly(vars, deg=2, coefs={(1, 0): 1})
y = ConcreteMultiVarPoly(vars, deg=2, coefs={(0, 1): 1})
target = x + y

opt = PolyOptimizer(vars, deg=2, n_regs=3, n_steps=1)
status, result = opt.find_code([x, y], [target])
print(result)
# n_steps: 1, n_regs: 3
# s:  0 [X, Y, 0]
# ADD  0,  1 -> 2
# s:  1 [X, Y, X + Y]
```

Multiple outputs are supported and matched permutation-invariantly:

```python
targets = [x * x + y * y, x * y * 2]   # compute both simultaneously
opt = PolyOptimizer(vars, deg=3, n_regs=4, n_steps=5)
status, result = opt.find_code([x, y], targets)
```

### Chain solver (`ChainPolyOptimizer` / `compute_multivar_expansion`)

Models computation as a growing list of registers (DAG). Each step appends a new register that can read from any previously computed register. This scales better for complex multi-output problems.

The convenience function `compute_multivar_expansion` wraps `ChainPolyOptimizer` and accepts plain `MultiVarPoly` inputs (no need to set explicit degrees).

```python
from src import compute_multivar_expansion, MultiVarPoly

vars = ["X", "Y"]
x = MultiVarPoly(vars, {(1, 0): 1})
y = MultiVarPoly(vars, {(0, 1): 1})
target = (x + y) * (x + y)   # X^2 + 2XY + Y^2

status, result = compute_multivar_expansion(
    [x, y], vars, [target], num_steps_=3, max_deg=2
)
print(result)
# One possible solution (OR-Tools may find any valid sequence):
#  0:               X
#  1:               Y
#  2: ADD 0,1 ===> X + Y
#  3: SUB 0,2 ===> -Y
#  4: MUL 2,2 ===> X^2 + 2XY + Y^2
#  5: IDN 4   ===> X^2 + 2XY + Y^2
```

#### Constraining multiplications

The `minimize_mul` parameter is the key hardware optimization knob:

```python
# Exactly k multiplications
status, result = compute_multivar_expansion(
    inputs, vars, targets, num_steps_=7, max_deg=2, minimize_mul=3
)

# Minimize the number of multiplications
status, result = compute_multivar_expansion(
    inputs, vars, targets, num_steps_=10, max_deg=2, minimize_mul=-1
)
```

### Algorithm Synthesis: Karatsuba Multiplication

The chain solver can rediscover known fast algorithms. Karatsuba multiplies two degree-1 polynomials `(a0 + a1*B)(b0 + b1*B)` using 3 multiplications instead of the naive 4:

```python
from src import compute_multivar_expansion, MultiVarPoly

vars = ["a0", "a1", "b0", "b1"]
a0 = MultiVarPoly(vars, {(1, 0, 0, 0): 1})
a1 = MultiVarPoly(vars, {(0, 1, 0, 0): 1})
b0 = MultiVarPoly(vars, {(0, 0, 1, 0): 1})
b1 = MultiVarPoly(vars, {(0, 0, 0, 1): 1})

# Targets: cross term (a0b1 + a1b0), low product (a0b0), high product (a1b1)
targets = [a1 * b0 + b1 * a0, a0 * b0, a1 * b1]

status, result = compute_multivar_expansion(
    [a0, a1, b0, b1], vars, targets,
    num_steps_=7, max_deg=2, minimize_mul=3
)
print(result)
```

### Single-variable solver (`compute_expansion`)

A simpler single-variable register solver that starts from zero and builds up to a target `OneVarPoly` using: `+1`, `-1`, `+X`, `-X`, double, square, multiply-by-X. Minimizes the number of active steps.

```python
from src import compute_expansion, OneVarPoly

target = OneVarPoly([0, 0, 1])   # X^2
status, result = compute_expansion(target, num_steps=3, max_deg=2)
print(result)
```

## API Reference

| Symbol | Description |
|--------|-------------|
| `OneVarPoly(coefs)` | Single-variable polynomial |
| `MultiVarPoly(vars, coefs)` | Multi-variable polynomial |
| `ConcreteMultiVarPoly(vars, deg, coefs)` | Multi-variable polynomial with explicit degree (required by solvers) |
| `PolyOptimizer(vars, deg, n_regs, n_steps)` | Register-file solver |
| `ChainPolyOptimizer(vars, deg, n_steps)` | Chain/DAG solver |
| `compute_multivar_expansion(init_polys, vars, targets, num_steps_, max_deg)` | Convenience wrapper for `ChainPolyOptimizer` |
| `compute_expansion(target, num_steps, max_deg)` | Single-variable solver |
| `AddOp, SubOp, MulOp, IdentityOp, IncOp, DecOp` | Operation classes (pass as `ops=` to override defaults) |

---

## TODO / Known Issues

These are code quality observations for future cleanup:

1. **DRY: inconsistent use of `add_keys`** — `MultiVarPoly.__mul__` ([poly.py:163](src/poly.py#L163)) and `mul_multivar_polys_enforce_no_ovf` ([poly.py:383](src/poly.py#L383)) inline `tuple([i0+i1 for i0,i1 in zip(k0,k1)])`. `ConcreteMultiVarPoly.__mul__` ([poly.py:245](src/poly.py#L245)) already uses the `add_keys` utility for the same thing.

2. **Debug prints in production** — `print(self.model.Validate())` is called on every solve in `PolyOptimizer.find_code` ([multi_var.py:244](src/multi_var.py#L244)), `ChainPolyOptimizer.find_code` ([multi_var.py:432](src/multi_var.py#L432)), and `compute_expansion` ([single_var.py:64](src/single_var.py#L64)). Should use `logging` or be gated behind a `verbose` flag.

3. **Dead commented-out code** — `_BaseOptimizer._new_poly` ([multi_var.py:50-53](src/multi_var.py#L50)) contains a 4-line commented-out block left over from a refactor.

4. **`issubclass(type(other), ...)` anti-pattern** — `OneVarPoly.__mul__` ([poly.py:31](src/poly.py#L31)) uses `issubclass(type(other), OneVarPoly)` instead of `isinstance(other, OneVarPoly)`.

5. **`SymbolicMultiVarPoly.__eq__` adds model constraints** — overloading `==` to add OR-Tools constraints ([poly.py:292](src/poly.py#L292)) is a trap: any accidental comparison (pytest assert, `in` check, conditional) silently corrupts the model. The explicit `.equate_to()` method is the safe API and should be the only way to do this.

6. **Misleading swap in `MulOp.apply_constraints`** ([ops.py:144](src/ops.py#L144)) — appends `(k_rhs, k_lhs)` instead of `(k_lhs, k_rhs)` to `target_coef_map`. Functionally harmless (the sum over all contributing pairs is symmetric), but confusing to read.

7. **Symmetry-breaking applied to unary ops in `ChainPolyOptimizer`** ([multi_var.py:381](src/multi_var.py#L381)) — `lhs_idx <= rhs_idx` is enforced for all operations including `IdentityOp` (unary). Since unary ops ignore `rhs`, this unnecessarily constrains the solver. `PolyOptimizer` correctly applies symmetry-breaking only for `op.is_commutative`.

8. **`compute_expansion` is architecturally isolated** — [single_var.py](src/single_var.py) uses an ad-hoc lambda-based operation list instead of the `Op`/`UnaryOp`/`BinaryOp` framework in [ops.py](src/ops.py). Makes it hard to share operation definitions or extend both solvers together.
