import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mpc_loop
import mpc_fast
import bench_cycle as bc

ALPHA, BETA, SIGMA_S, GBAR, HORIZON = bc.ALPHA, bc.BETA, bc.SIGMA_S, bc.GBAR, bc.HORIZON
theta = np.array([2.0e-4, 1.0e-4])
h = 0.0137

for label, install in (("NO install", False), ("WITH install", True)):
    m = mpc_loop.BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=4242)
    if install:
        mpc_fast.install(m)
    r_ref = m.step(theta.copy(), h_meas=h)
    run = bc.CycleRunner(4242, anytime=False)
    _, _, diag = run.cycle(theta.copy(), h)
    print(f"[{label}] step best_f={r_ref.best_f!r}  cycle best_f={diag[3]!r}  "
          f"identical={r_ref.best_f == diag[3]}  evals {r_ref.evaluations}/{diag[1]}  iters {r_ref.iterations}/{diag[0]}")

# objective-level check: same swarm, fast vs loop objective
m1 = mpc_loop.BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=4242)
m2 = mpc_loop.BeamSteeringMPC(ALPHA, BETA, SIGMA_S, GBAR, horizon=HORIZON, seed=4242)
mpc_fast.install(m2)
X = np.array([[0.05, 2e-4, 1e-4], [0.12, -1e-4, 2e-4], [0.2, 3e-4, -2e-4]])
st = np.zeros(3)
try:
    f1, _ = m1._objective(X, st, np.array([h, h, h]))
    f2, _ = m2._objective(X, st, np.array([h, h, h]))
    print("objective loop vs fast:", f1, f2, "identical:", np.array_equal(f1, f2))
except Exception as e:
    print("objective direct check failed:", e)
