from .poly import (
    OneVarPoly,
    initialize_poly_coefs,
    mul_polys_enforce_no_ovf,
    assert_poly_eq,
)
from ortools.sat.python import cp_model


def compute_expansion(target, num_steps, max_deg, bound=10**10):

    model = cp_model.CpModel()

    reg_polys = [
        OneVarPoly(initialize_poly_coefs(model, max_deg, bound, f"r_{i}"))
        for i in range(num_steps + 1)
    ]

    zero = OneVarPoly([0] * (max_deg + 1))
    one = OneVarPoly([1] + [0] * max_deg)
    m_one = OneVarPoly([-1] + [0] * max_deg)
    x = OneVarPoly([0, 1] + [0] * (max_deg - 1))
    m_x = OneVarPoly([0, -1] + [0] * (max_deg - 1))

    NOP_IDX = 0
    ops = [
        lambda poly: (poly, []),  # nop
        lambda poly: (poly + one, []),
        lambda poly: (poly + m_one, []),
        lambda poly: (poly + x, []),
        lambda poly: (poly + m_x, []),
        lambda poly: (poly + poly, []),  # MUL2
        lambda poly: mul_polys_enforce_no_ovf(model, poly, poly, max_deg, bound),  # SQR
        lambda poly: mul_polys_enforce_no_ovf(
            model, poly, x, max_deg, bound
        ),  # MULX (SHL)
    ]

    op_selectors = [
        model.new_int_var(0, len(ops) - 1, f"op_selector_{i}") for i in range(num_steps)
    ]

    is_ops = [model.new_bool_var(f"is_op_{i}") for i in range(num_steps)]
    assert_poly_eq(model, reg_polys[0], zero)

    for i in range(num_steps):
        model.add(op_selectors[i] != NOP_IDX).only_enforce_if(is_ops[i])
        model.add(op_selectors[i] == NOP_IDX).only_enforce_if(~is_ops[i])
        for op_idx, op in enumerate(ops):
            res, constraints = op(reg_polys[i])
            op_one_hot = model.new_bool_var(f"step_{i}_op_{op_idx}")
            model.add(op_selectors[i] == op_idx).only_enforce_if(op_one_hot)
            model.add(op_selectors[i] != op_idx).only_enforce_if(~op_one_hot)
            for constraint in constraints:
                model.add(constraint).only_enforce_if(op_one_hot)
            assert len(res) == (max_deg + 1), f"{i=}, {op_idx=}, {len(res)=}"
            for j in range(max_deg + 1):
                model.add(reg_polys[i + 1][j] == res[j]).only_enforce_if(op_one_hot)

    assert_poly_eq(model, reg_polys[-1], target)

    model.minimize(sum(is_ops))

    print(model.Validate())
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.OPTIMAL:
        rv = (
            "".join(
                [
                    (
                        f"{OneVarPoly([solver.value(reg_) for reg_ in reg])} =={solver.value(op)}=> \n"
                        if solver.value(op) != NOP_IDX
                        else ""
                    )
                    for op, reg in zip(op_selectors, reg_polys)
                ]
            )
            + f"{OneVarPoly([solver.value(reg_) for reg_ in reg_polys[-1]])}\n"
            + f"Op count: {solver.objective_value}"
        )
    else:
        rv = f"UNSAT {status}"
    return status, rv
