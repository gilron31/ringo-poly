from ortools.sat.python import cp_model


def compute_binary_expansion(target, num_steps, upper_bound=10**10):

    model = cp_model.CpModel()

    reg_values = [model.new_int_var(0, upper_bound, f"r_{i}") for i in range(num_steps)]
    # 0: inc, 1: double
    ops = [model.new_bool_var(f"op_{i}") for i in range(num_steps)]

    model.add(reg_values[0] == 1)
    for i in range(num_steps - 1):
        model.add(reg_values[i + 1] == reg_values[i] + 1).only_enforce_if(ops[i])
        model.add(reg_values[i + 1] == 2 * reg_values[i]).only_enforce_if(~ops[i])

    model.add(reg_values[-1] == target)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.OPTIMAL:
        return [
            (solver.value(op), solver.value(reg)) for op, reg in zip(ops, reg_values)
        ]
    else:
        return None
