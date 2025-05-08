# src/fitness.py

import numpy as np
import logging
from simulation import run_ode
from models import pulse_input  # or wherever your pulse_input lives

logger = logging.getLogger(__name__)

def fitness_pulse_count(
    candidate_params,
    model_fn,
    y0,
    t_span,
    pulse_schedule,
    pulse_params,
    measurement_delay,
    measurement_window,
    reporter_index,
    target_increment=1.0,
    *,
    t_eval=None
):
    """
    Compute sum of squared errors in pulse‐counting for one candidate.
    
    Parameters
    ----------
    candidate_params : dict
        kinetic parameters including 'k_pulse', etc.
    model_fn : callable
        f_base(t, y, params) defining the adjacency‐only ODE.
    y0 : sequence
        initial conditions.
    t_span : tuple
        (t0, tf) integration window.
    pulse_schedule : list of floats
        times at which pulses occur.
    pulse_params : dict
        {'width':…, 'amplitude':…} for pulse_input.
    measurement_delay : float
        time after pulse to start measuring.
    measurement_window : float
        duration over which to average the reporter.
    reporter_index : int
        which state variable to read out.
    target_increment : float, optional
        desired increment per pulse (default 1.0).
    t_eval : array_like, optional
        times at which to record the solution (passed to solve_ivp).
    """
    # 1) wrap the base model to include u(t)*k_pulse in node 0's basal rate
    def model_with_pulse(t, y, params):
        u = pulse_input(
            t,
            pulse_schedule,
            pulse_params['width'],
            pulse_params['amplitude'],
        )
        p2 = params.copy()
        p2['k_base_0'] = p2.get('k_base_0', 0.0) + p2.get('k_pulse', 0.0) * u
        return model_fn(t, y, p2)

    # 2) integrate
    try:
        sol = run_ode(
            model_with_pulse,
            y0,
            t_span,
            candidate_params,
            t_eval=t_eval
        )
    except Exception:
        logger.exception("Simulation failed for %s", candidate_params)
        return 1e6

    t = sol.t
    Y = sol.y[reporter_index]

    # 3) compute mean response in each window
    responses = []
    for i, pt in enumerate(pulse_schedule):
        mask = (t >= pt + measurement_delay) & (t < pt + measurement_delay + measurement_window)
        if np.any(mask):
            responses.append(Y[mask].mean())
        else:
            responses.append(0.0)

    # 4) squared‐error vs. desired ramp [0, 1*inc, 2*inc, …]
    desired = np.arange(len(responses)) * target_increment
    err = np.sum((np.array(responses) - desired) ** 2)
    return err