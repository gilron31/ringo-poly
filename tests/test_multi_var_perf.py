import pytest
from ortools.sat.python import cp_model
from src import (
    compute_multivar_expansion,
    MultiVarPoly,
)
import time


def _test_karatsuba_multiplication(bound):

    start_time = time.time()
    vars = ["a0", "a1", "b0", "b1"]
    a0 = MultiVarPoly(vars, {(1, 0, 0, 0): 1})
    a1 = MultiVarPoly(vars, {(0, 1, 0, 0): 1})
    b0 = MultiVarPoly(vars, {(0, 0, 1, 0): 1})
    b1 = MultiVarPoly(vars, {(0, 0, 0, 1): 1})

    # Targets: cross term, low product, high product
    targets = [a1 * b0 + b1 * a0, a0 * b0, a1 * b1]

    status, res = compute_multivar_expansion(
        [a0, a1, b0, b1],
        vars,
        targets,
        num_steps_=7,
        max_deg=2,
        minimize_mul=3,
        bound=bound,
    )
    assert status == cp_model.OPTIMAL
    elapsed_s = time.time() - start_time
    print(f"{bound=} {elapsed_s=:0.3f}")
    print(res)


@pytest.mark.manual
def test_karatsuba_multiple_bounds():
    """
    All runs exhibit similar runtime (20s-35s).
    There is also an element of randomness in all this. So we should rely on a few runs
    """
    for i in range(1, 4):
        _test_karatsuba_multiplication(10**i)
