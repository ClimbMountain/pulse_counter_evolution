# src/simulation.py

import numpy as np
from scipy.integrate import solve_ivp
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def pulse_input(t, schedule, width, amplitude):
    """
    A simple on/off pulse train:
      – schedule : list of pulse start times
      – width    : duration of each pulse
      – amplitude: height when “on”
    Returns amplitude if t ∈ [pt, pt+width), else 0.
    """
    for pt in schedule:
        if pt <= t < pt + width:
            return amplitude
    return 0.0

def run_ode(model, y0, t_span, params, method='RK45', atol=1e-6, rtol=1e-3, t_eval=None, return_events=False):
    """
    Solves a system of ODEs for a given model.
    """
    if not isinstance(y0, (list, np.ndarray)):
        logger.error("y0 must be a list or numpy array, got: %s", type(y0))
        raise TypeError("y0 must be a list or numpy array")
    
    if not (isinstance(t_span, (list, tuple)) and len(t_span) == 2):
        logger.error("t_span must be a list or tuple with two elements (t_start, t_end)")
        raise ValueError("t_span must be a list or tuple with two elements")
    
    t_start, t_end = t_span

    def wrapped_model(t, y):
        return model(t, y, params)
    
    try:
        sol = solve_ivp(fun=wrapped_model, t_span=(t_start, t_end), y0=y0,
                        method=method, atol=atol, rtol=rtol, t_eval=t_eval)
    except Exception as e:
        logger.exception("Error during ODE integration")
        raise e

    if not sol.success:
        logger.error("ODE solver failed: %s", sol.message)
        raise RuntimeError("ODE solver did not converge: " + sol.message)
    
    if return_events:
        return sol, sol.t_events
    return sol