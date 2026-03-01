import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import z3
    return (z3,)


@app.cell
def _(z3):
    x = z3.Int('x')
    y = z3.Int('y')
    z3.solve(x > 2, y < 10, x+2*y == 7)
    return (x,)


@app.cell
def _(x, z3):
    z3.simplify(x + x + x + 1.2 * x)
    return


if __name__ == "__main__":
    app.run()
