import brainpy.math as bm
def f(x):
    return bm.sum(x**2), x
gv = bm.TrainVar(bm.array([1.0, 2.0]))
gfn = bm.grad(f, grad_vars=[gv], has_aux=True, return_value=True)
res = gfn(bm.array([1.0, 2.0]))
print(f"Length of result: {len(res)}")
for i, r in enumerate(res):
    print(f"Part {i}: {type(r)}")
