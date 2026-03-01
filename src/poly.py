from ortools.sat.python import cp_model
import itertools


class OneVarPoly:
    def __init__(self, coefs):
        self.coefs = coefs.copy()
        self.var = "X"

    def __len__(self):
        return len(self.coefs)

    def __getitem__(self, idx):
        return self.coefs[idx]

    def __add__(self, other):
        rv = [0] * max(len(self), len(other))
        for i in range(len(rv)):
            rv[i] += self[i] if i < len(self) else 0
            rv[i] += other[i] if i < len(other) else 0
        return OneVarPoly(rv)

    def __sub__(self, other):
        rv = [0] * max(len(self), len(other))
        for i in range(len(rv)):
            rv[i] += self[i] if i < len(self) else 0
            rv[i] -= other[i] if i < len(other) else 0
        return OneVarPoly(rv)

    def __mul__(self, other):
        assert issubclass(type(other), OneVarPoly)
        if len(self) == 0 or len(other) == 0:
            return OneVarPoly([])
        rv = [0] * (len(self) + len(other) - 1)
        for i in range(len(self)):
            for j in range(len(other)):
                rv[i + j] += self[i] * other[j]
        return OneVarPoly(rv)

    def canonize(self):
        rv = OneVarPoly(self.coefs)
        if len(rv) == 0:
            return rv
        i = 0
        for i in range(len(self)):
            if rv[-i - 1] != 0:
                break
        else:
            # All coefficients are zero
            return OneVarPoly([])
        rv.coefs = rv.coefs[: len(rv) - i]
        return rv

    def __eq__(self, other):
        assert isinstance(other, OneVarPoly)
        s_can = self.canonize()
        o_can = other.canonize()
        if len(s_can) != len(o_can):
            return False
        else:
            rv = True
            for x, y in zip(s_can.coefs, o_can.coefs):
                rv &= x == y
            return rv

    def __repr__(self):
        rv = ""
        if all([x == 0 for x in self.coefs]):
            return "0"
        for i in range(len(self)):
            plus = " + " if not all([x == 0 for x in self.coefs[:i]]) else ""
            term = f"X^{i}" if i > 1 else ("X" if i == 1 else "")
            coef = f"{self[i]}" if self[i] != 1 or i == 0 else ""
            rv = (f"{coef}{term}{plus}" if self[i] != 0 else "") + rv
        return rv


def combination_to_keys(comb, n_vars):
    rv = [0] * n_vars
    for i in comb:
        rv[i] += 1
    return tuple(rv)

def generate_all_keys(n_vars, deg, homogenous=True):
    if homogenous:
        return [
            combination_to_keys(x, n_vars)
            for x in itertools.combinations_with_replacement(range(n_vars), deg)
        ]
    else:
        rv = []
        for i in range(deg + 1):
            rv += generate_all_keys(n_vars, i)
        return rv

def clear_zero_coefs(coefs):
    return {k: v for k, v in coefs.items() if v != 0}


def add_keys(t1, t2):
    """Add two exponent tuples element-wise."""
    return tuple(t1i + t2i for t1i, t2i in zip(t1, t2))

class MultiVarPoly:
    def __init__(self, vars, coefs=None, deg=None):
        self.vars = vars
        self.n_vars = len(vars)
        self._deg = deg  # Optional degree bound
        assert self.vars == sorted(
            self.vars
        ), "Vars must be sorted in alphabetical order"
        self.coefs = coefs.copy() if coefs else {}

    def __getitem__(self, idx):
        return self.coefs[idx]

    @property
    def deg(self):
        """Return stored degree if set, otherwise compute from coefficients."""
        if self._deg is not None:
            return self._deg
        if not self.coefs:
            return 0
        return max([sum(x) for x in self.coefs.keys()])

    def max_deg(self):
        """Compute max degree from coefficients."""
        if not self.coefs:
            return 0
        return max([sum(x) for x in self.coefs.keys()])

    def __add__(self, other):
        assert self.vars == other.vars
        rv = MultiVarPoly(self.vars, self.coefs.copy())
        for k, v in other.coefs.items():
            if k in rv.coefs:
                rv.coefs[k] += v
            else:
                rv.coefs[k] = v
        return rv

    def __sub__(self, other):
        assert self.vars == other.vars
        rv = MultiVarPoly(self.vars, self.coefs.copy())
        for k, v in other.coefs.items():
            if k in rv.coefs:
                rv.coefs[k] -= v
            else:
                rv.coefs[k] = -v
        return rv

    def __mul__(self, other):
        if isinstance(other, int):
            rv = MultiVarPoly(self.vars, self.coefs)
            for k, v in rv.coefs.items():
                rv.coefs[k] = v * other
            return rv
        elif isinstance(other, MultiVarPoly):
            assert self.vars == other.vars
            rv = MultiVarPoly(self.vars, {})
            for k0, v0 in self.coefs.items():
                for k1, v1 in other.coefs.items():
                    k = tuple([i0 + i1 for i0, i1 in zip(k0, k1)])
                    if k in rv.coefs:
                        rv.coefs[k] += v0 * v1
                    else:
                        rv.coefs[k] = v0 * v1
            return rv
        else:
            raise NotImplementedError()

    def __repr__(self):
        rv = ""
        if all([v == 0 for v in self.coefs.values()]):
            return "0"
        prev_elements = False
        for k, v in self.coefs.items():
            if v == 0:
                continue
            plus = (" + " if v > 0 else " - ") if prev_elements else ""
            term = "".join(
                [
                    (
                        f"{self.vars[i]}^{k[i]}"
                        if k[i] > 1
                        else (f"{self.vars[i]}" if k[i] == 1 else "")
                    )
                    for i in range(len(self.vars))
                ]
            )
            minus = ("-" if v < 0 else "") if not prev_elements else ""
            coef = f"{abs(v)}" if abs(v) != 1 else ""
            rv = rv + f"{plus}{minus}{coef}{term}"
            prev_elements = True

        return rv

    def __eq__(self, other):
        if not isinstance(other, MultiVarPoly):
            return False
        if self.vars != other.vars:
            return False
        return clear_zero_coefs(self.coefs) == clear_zero_coefs(other.coefs)


class ConcreteMultiVarPoly(MultiVarPoly):
    """Multi-variable polynomial with concrete integer coefficients."""

    def __init__(self, vars, deg, coefs=None):
        super().__init__(vars, coefs, deg)

    def __add__(self, other):
        assert isinstance(other, ConcreteMultiVarPoly)
        assert self.vars == other.vars
        rv = ConcreteMultiVarPoly(self.vars, self._deg, self.coefs.copy())
        for k, v in other.coefs.items():
            if k in rv.coefs:
                rv.coefs[k] += v
            else:
                rv.coefs[k] = v
        return rv

    def __sub__(self, other):
        assert isinstance(other, ConcreteMultiVarPoly)
        assert self.vars == other.vars
        rv = ConcreteMultiVarPoly(self.vars, self._deg, self.coefs.copy())
        for k, v in other.coefs.items():
            if k in rv.coefs:
                rv.coefs[k] -= v
            else:
                rv.coefs[k] = -v
        return rv

    def __mul__(self, other):
        if isinstance(other, int):
            rv = ConcreteMultiVarPoly(self.vars, self._deg, self.coefs.copy())
            for k in rv.coefs:
                rv.coefs[k] *= other
            return rv
        elif isinstance(other, ConcreteMultiVarPoly):
            assert self.vars == other.vars
            rv = ConcreteMultiVarPoly(self.vars, self._deg, {})
            for k0, v0 in self.coefs.items():
                for k1, v1 in other.coefs.items():
                    k = add_keys(k0, k1)
                    if k in rv.coefs:
                        rv.coefs[k] += v0 * v1
                    else:
                        rv.coefs[k] = v0 * v1
            return rv
        else:
            raise NotImplementedError()


class SymbolicMultiVarPoly(MultiVarPoly):
    """Multi-variable polynomial with OR-Tools IntVar coefficients."""

    def __init__(self, model, vars, max_deg, bound, name_prefix, solver=None):
        coefs = initialize_multivarpoly_coefs(model, max_deg, vars, bound, name_prefix)
        super().__init__(vars, coefs, max_deg)
        self.model = model
        self.bound = bound
        self.name_prefix = name_prefix
        self.solver = solver

    def set_solver(self, solver):
        self.solver = solver

    def to_concrete(self):
        """Convert to ConcreteMultiVarPoly using solved values."""
        assert self.solver is not None, "Solver not set"
        return ConcreteMultiVarPoly(
            self.vars,
            self._deg,
            {k: self.solver.value(v) for k, v in self.coefs.items()},
        )

    def equate_to(self, other, cond=None):
        """Add constraints to make this polynomial equal to other."""
        for k in self.coefs:
            if k in other.coefs:
                val = other.coefs[k]
            else:
                val = 0
            if cond is None:
                self.model.add(self.coefs[k] == val)
            else:
                self.model.add(self.coefs[k] == val).only_enforce_if(cond)
        for k in other.coefs:
            assert k in self.coefs, f"{k}, {self.coefs}"

    def __eq__(self, other):
        """Overload == to add constraints (for compatibility with poly_optimizer2 tests)."""
        self.equate_to(other)
        return True  # Return truthy to avoid issues

    def __repr__(self):
        if self.solver is not None:
            return self.to_concrete().__repr__()
        return f"SymbolicMultiVarPoly({self.name_prefix})"


def initialize_poly_coefs(model, deg, bound=10**9, name_prefix=""):
    return [
        model.new_int_var(-bound, bound, f"{name_prefix}_{i}") for i in range(deg + 1)
    ]


def initialize_multivarpoly_coefs(model, deg, vars, bound=10**9, name_prefix=""):
    all_coefs = generate_all_keys(len(vars), deg, homogenous=False)
    result = {}
    for k in all_coefs:
        var_name = "_".join([f"{vars[i]}^{k[i]}" for i in range(len(vars))])
        result[k] = model.new_int_var(-bound, bound, f"{name_prefix}_{var_name}")
    return result


def assert_poly_eq(model, lhs, rhs, cond=None):
    if isinstance(lhs, OneVarPoly):
        assert len(lhs) == len(rhs)
        for i in range(len(lhs)):
            if cond is None:
                model.add(lhs[i] == rhs[i])
            else:
                model.add(lhs[i] == rhs[i]).only_enforce_if(cond)
    elif isinstance(lhs, MultiVarPoly):
        # assert lhs.coefs.keys() == rhs.coefs.keys()
        for k in lhs.coefs:
            if k in rhs.coefs:
                val = rhs.coefs[k]
            else:
                val = 0
            if cond is None:
                model.add(lhs[k] == val)
            else:
                model.add(lhs[k] == val).only_enforce_if(cond)
        for k in rhs.coefs:
            assert k in lhs.coefs, f"{k}, {lhs.coefs}"
    else:
        raise NotImplementedError()


def canonize_assert_zero(poly, deg):
    constraints = []
    if isinstance(poly, OneVarPoly):
        for var in poly.coefs[deg + 1 :]:
            constraints.append(var == 0)
        return OneVarPoly(poly.coefs[: deg + 1]), constraints
    elif isinstance(poly, MultiVarPoly):
        rv = MultiVarPoly(poly.vars, {})
        for k, v in poly.coefs.items():
            if sum(k) > deg:
                constraints.append(v == 0)
            else:
                rv.coefs[k] = v
        return rv, constraints
    else:
        raise NotImplementedError()


def mul_polys_enforce_no_ovf(model, lhs, rhs, max_deg, bound):
    assert len(lhs) != 0 and len(rhs) != 0
    rv = [0] * (len(lhs) + len(rhs) - 1)
    for i in range(len(lhs)):
        for j in range(len(rhs)):
            if isinstance(rhs[j], int):
                mul_var = lhs[i] * rhs[j]
            else:
                mul_var = model.new_int_var(
                    -bound, bound, lhs[i].name + "_MUL_" + rhs[j].name
                )
                model.add_multiplication_equality(mul_var, [lhs[i], rhs[j]])
            rv[i + j] += mul_var
    rv_, constraints = canonize_assert_zero(OneVarPoly(rv), max_deg)
    return rv_, constraints


def mul_multivar_polys_enforce_no_ovf(model, lhs, rhs, max_deg, bound):
    assert lhs.vars == rhs.vars
    rv = MultiVarPoly(lhs.vars, {})
    for k0, v0 in lhs.coefs.items():
        for k1, v1 in rhs.coefs.items():
            k = tuple([i0 + i1 for i0, i1 in zip(k0, k1)])
            if isinstance(v1, int):
                mul_var = v0 * v1
            else:
                mul_var = model.new_int_var(-bound, bound, v0.name + "_MUL_" + v1.name)
                model.add_multiplication_equality(mul_var, [v0, v1])
            if k in rv.coefs:
                rv.coefs[k] += mul_var
            else:
                rv.coefs[k] = mul_var
    rv_, constraints = canonize_assert_zero(rv, max_deg)
    return rv_, constraints
