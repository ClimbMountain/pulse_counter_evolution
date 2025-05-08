# src/models.py

import numpy as np
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)

def pulse_counter_model(t, y, params):
    """
    Candidate pulse-counter model: a pulse-triggered integrator with positive feedback.
    
    Variables:
      y[0]: mRNA concentration (M)
      y[1]: Protein concentration (P) [also serves as reporter]
    
    Parameters in params:
      - k_base: Basal transcription rate.
      - k_pulse: Pulse-induced transcription rate.
      - k_positive: Positive feedback strength.
      - n: Hill coefficient for self-activation.
      - K: Activation threshold for the Hill function.
      - d_M: mRNA degradation rate.
      - k_translation: Protein translation rate.
      - d_P: Protein degradation rate.
      - pulse: External pulse input (passed dynamically).
    
    Returns:
      List containing [dM/dt, dP/dt]
    """
    # Check required parameters
    for param in ['k_base', 'k_pulse', 'k_positive', 'n', 'K', 'd_M', 'k_translation', 'd_P']:
        if param not in params:
            logger.error("Missing parameter '%s' in params.", param)
            raise KeyError(f"Parameter '{param}' is required in params.")
    
    k_base = params['k_base']
    k_pulse = params['k_pulse']
    k_positive = params['k_positive']
    n = params['n']
    K_val = params['K']
    d_M = params['d_M']
    k_translation = params['k_translation']
    d_P = params['d_P']
    pulse = params.get('pulse', 0.0)  # Default to 0 if not provided

    M, P = y[0], y[1]
    
    # Compute positive feedback using a Hill function.
    try:
        P_pos = np.maximum(P, 0.0)
        activation = (P_pos**n) / (K_val**n + P_pos**n)
    except Exception as e:
        logger.exception("Error computing Hill function for P=%s with n=%s and K=%s", P, n, K_val)
        raise e

    dM_dt = k_base + k_pulse * pulse + k_positive * activation - d_M * M
    dP_dt = k_translation * M - d_P * P

    return [dM_dt, dP_dt]


def pulse_input(t, pulse_schedule, pulse_width, amplitude):
    """
    Return the pulse input value at time t based on a pulse schedule.

    Args:
      t (float): Current time.
      pulse_schedule (list): List of pulse start times.
      pulse_width (float): Duration of each pulse.
      amplitude (float): Amplitude of the pulse.

    Returns:
      float: Pulse value (amplitude if t is within a pulse period, else 0).
    """
    for start in pulse_schedule:
        if start <= t < start + pulse_width:
            return amplitude
    return 0.0

def repressilator_model(t, y, params):
    """
    Defines the ODE model for the repressilator circuit.
    
    The repressilator is a synthetic gene circuit consisting of three genes (A, B, and C)
    that inhibit each other cyclically. The ODEs are defined as:
    
        dA/dt = -A + alpha / (1 + C^n)
        dB/dt = -B + alpha / (1 + A^n)
        dC/dt = -C + alpha / (1 + B^n)
    
    Args:
        t (float): Current time (the equations are autonomous and do not explicitly depend on t).
        y (list or np.ndarray): Current concentrations of [A, B, C].
        params (dict): Dictionary of parameters containing:
                       - 'alpha': Maximum production rate (float, > 0).
                       - 'n': Hill coefficient (float, >= 1).
    
    Returns:
        list: Derivatives [dA/dt, dB/dt, dC/dt]
    
    Raises:
        KeyError: If 'alpha' or 'n' are missing in the params dictionary.
        ValueError: If y does not contain exactly three elements.
    """
    # Check for required parameters
    if 'alpha' not in params or 'n' not in params:
        logger.error("Parameters 'alpha' and 'n' must be provided in params.")
        raise KeyError("params must include 'alpha' and 'n'")
    
    alpha = params['alpha']
    n = params['n']
    
    # Check that y has the proper length
    if len(y) != 3:
        logger.error("Expected 3 state variables for [A, B, C], but got %d.", len(y))
        raise ValueError("y must have three elements representing [A, B, C]")
    
    A, B, C = y

    try:
        # Compute derivatives with proper handling of exponentiation
        dA_dt = -A + alpha / (1 + C**n)
        dB_dt = -B + alpha / (1 + A**n)
        dC_dt = -C + alpha / (1 + B**n)
    except Exception as e:
        logger.exception("Error calculating derivatives. Check your parameters and input values.")
        raise e

    return [dA_dt, dB_dt, dC_dt]