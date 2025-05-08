# src/optimization.py

import random
import numpy as np
from deap import base, creator, tools, algorithms

from fitness import fitness_pulse_count

def run_optimization(
    population_size: int,
    generations: int,
    cxpb: float,
    mutpb: float,
    tournsize: int,
    hof_size: int,
    target_increment: float,
    *,
    model_fn,
    y0,
    t_span,
    pulse_schedule,
    pulse_params,
    measurement_delay,
    measurement_window,
    reporter_index
):
    """
    Inner‑loop evolutionary search over kinetic parameters for one fixed topology.

    Required keyword args:
      model_fn          – ODE f(t,y,params) already wrapped with pulses  
      y0                – initial state list/array  
      t_span            – (t0, t1) integration interval  
      pulse_schedule    – list of pulse times  
      pulse_params      – dict with 'width' & 'amplitude'  
      measurement_delay – seconds after each pulse to start measuring  
      measurement_window– length of measurement window  
      reporter_index    – which ODE state is the readout  

    Returns (best_params_dict, logbook)
    """

    # — DEAP setup —
    creator.create("ParamFitness", base.Fitness, weights=(-1.0,))  
    creator.create("Params", dict, fitness=creator.ParamFitness)

    toolbox = base.Toolbox()

    # 1) parameters to evolve
    PARAM_KEYS = [
        'k_base', 'k_pulse', 'k_positive',
        'n', 'K',
        'd_M', 'k_translation', 'd_P'
    ]
    BOUNDS = {
        'k_base':         (0.0, 1.0),
        'k_pulse':        (0.0, 5.0),
        'k_positive':     (0.0, 5.0),
        'n':              (1.0, 5.0),
        'K':              (0.1, 20.0),
        'd_M':            (0.01, 1.0),
        'k_translation':  (0.1, 10.0),
        'd_P':            (0.001, 0.1),
    }

    def random_candidate():
        """Draw a random dict of kinetic params."""
        d = {k: random.uniform(*BOUNDS[k]) for k in PARAM_KEYS}
        return creator.Params(d)

    def mate(a, b):
        # uniform crossover on scalar dict values
        for k in PARAM_KEYS:
            if random.random() < 0.5:
                a[k], b[k] = b[k], a[k]
        return a, b

    def mutate(individual, indpb=0.2):
        # gaussian perturbation, clipped
        for k in PARAM_KEYS:
            if random.random() < indpb:
                low, hi = BOUNDS[k]
                sigma = (hi - low) * 0.1
                individual[k] = float(np.clip(
                    random.gauss(individual[k], sigma),
                    low, hi
                ))
        return (individual,)

    def _evaluate(candidate):
        """Wrap the fitness_pulse_count call with all of our args."""
        err = fitness_pulse_count(
            candidate_params     = candidate,
            model_fn             = model_fn,
            y0                   = y0,
            t_span               = t_span,
            t_eval               = np.linspace(t_span[0], t_span[1], 2000),
            pulse_schedule       = pulse_schedule,
            pulse_params         = pulse_params,
            target_increment     = target_increment,
            measurement_delay    = measurement_delay,
            measurement_window   = measurement_window,
            reporter_index       = reporter_index
        )
        return (err,)

    # register
    toolbox.register("individual", random_candidate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("evaluate", _evaluate)

    # stats & hall‑of‑fame
    pop = toolbox.population(population_size)
    hof = tools.HallOfFame(hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # run GA
    pop, logbook = algorithms.eaSimple(
        population=pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    # best is in hall‑of‑fame[0]
    best = hof[0]
    return best, logbook