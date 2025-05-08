#!/usr/bin/env python3
import sys, os

# — add src/ to PYTHONPATH (so we can import your build_model)
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
src_path  = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
from simulation        import run_ode
from topology_search   import build_model   # ← make sure this is your patched version
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# build_model must now have the signature:
#   build_model(A_flat,
#               pulse_schedule,
#               pulse_params,
#               input_node=0)

def validate_candidate(
    params,           # your optimized parameter dict
    A_flat,           # flat adjacency list, e.g. [0,0,0,1,0,0,...]
    y0,
    t_span,
    pulse_schedule,
    pulse_params,
    measurement_delay,
    measurement_window,
    reporter_index,
    input_node=0      # which species/node receives the pulse
):
    """
    1) Build the ODE with pulsing injected,
    2) Simulate it,
    3) Print & plot the pulse responses & increments.
    """
    # --- 1) build the actual pulsed model function ---
    model_fn = build_model(
        A_flat,
        pulse_schedule=pulse_schedule,
        pulse_params=pulse_params,
        input_node=input_node
    )

    # --- 2) simulate ---
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol    = run_ode(model_fn, y0, t_span, params, t_eval=t_eval)
    times  = sol.t
    y      = sol.y[reporter_index, :]

    # --- 3) measure responses ---
    responses = []
    for tp in pulse_schedule:
        window_start = tp + measurement_delay
        window_end   = window_start + measurement_window
        idx = np.where((times >= window_start) & (times < window_end))[0]
        if idx.size:
            responses.append(y[idx].mean())
        else:
            responses.append(np.nan)

    increments = np.diff(responses)

    # --- 4) print & plot ---
    print("Pulse responses:", responses)
    print("Increments    :", increments)

    plt.figure(figsize=(6,3))
    plt.plot(times, y, label=f"node {reporter_index}")
    for tp in pulse_schedule:
        plt.axvline(tp,   color='k', linestyle='--', alpha=0.3)
        plt.axvline(tp + measurement_delay,   color='r', linestyle='--', alpha=0.5)
        plt.axvline(tp + measurement_delay + measurement_window, color='g', linestyle='--', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel(f"State[{reporter_index}]")
    plt.legend()
    plt.tight_layout()
    plt.show()