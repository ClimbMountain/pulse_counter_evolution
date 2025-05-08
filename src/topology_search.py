# src/topology_search.py

import random
import os
import numpy as np
from deap import base, creator, tools, algorithms
from models import pulse_input

# make sure you can import from src/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in os.sys.path:
    os.sys.path.append(repo_root)

from models import pulse_input
from optimization import run_optimization
from fitness import fitness_pulse_count

# --- Global settings ---
N = 3
y0 = [0.0] * N
t_span = (0, 130)

pulse_schedule   = [10, 30, 50, 70, 90]
pulse_params     = {'width': 1.0, 'amplitude': 1.0}

target_increment   = 1.0
measurement_delay  = 2.0
measurement_window = 1.0
reporter_index     = N - 1

lambda_complexity = 0.1      # complexity penalty per non‑zero edge
topo_size         = N * N   # flattened adjacency length

# --- DEAP setup for topology search ---
creator.create("TopoFitness", base.Fitness, weights=(-1.0,))
creator.create("Topology", list, fitness=creator.TopoFitness)

toolbox = base.Toolbox()

def random_topology():
    """Random adjacency (self‑regulation allowed) in {-1,0,+1}."""
    return creator.Topology([random.choice([-1,0,1]) for _ in range(topo_size)])

def mutate_topology(ind, indpb=0.1):
    """Flip each entry with probability indpb."""
    for i in range(topo_size):
        if random.random() < indpb:
            ind[i] = random.choice([-1,0,1])
    return (ind,)

def cx_topology(a, b):
    return tools.cxTwoPoint(a, b)

toolbox.register("individual", random_topology)
toolbox.register("population", lambda n: [toolbox.individual() for _ in range(n)])
toolbox.register("mate", cx_topology)
toolbox.register("mutate", mutate_topology, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Build an ODE function for a given adjacency matrix ---
def build_model(A_flat,
                pulse_schedule,
                pulse_params,
                input_node=0):
    A = np.array(A_flat).reshape((N, N))

    def f(t, y, params):
        # 1) get the external pulse
        u = pulse_input(t,
                        pulse_schedule,
                        pulse_params['amplitude'],
                        pulse_params['width'])

        dydt = np.zeros_like(y)
        # 2) inject into your chosen “input” species:
        dydt[input_node] += params.get('k_pulse', 0.0) * u

        # 3) now your usual basal + interactions + degradation
        for i in range(N):
            dydt[i] += params.get(f'k_base_{i}', 0.0)
            for j in range(N):
                if A[i,j] != 0:
                    sign   = A[i,j]
                    k      = params.get(f'k_{i}_{j}', 0)
                    n      = params.get(f'n_{i}_{j}', 1)
                    K      = params.get(f'K_{i}_{j}', 1)
                    hill   = y[j]**n / (K**n + y[j]**n)
                    dydt[i]+= sign * k * hill
            dydt[i] -= params.get(f'd_{i}', 0) * y[i]
        return dydt

    return f

# --- Inner‐loop: optimize kinetics for one topology ---
def inner_parameter_opt(A_flat):
    model_fn = build_model(A_flat)

    best_params, _ = run_optimization(
        population_size     = 50,
        generations         = 50,
        cxpb                = 0.5,
        mutpb               = 0.2,
        tournsize           = 3,
        hof_size            = 1,
        target_increment    = target_increment,
        model_fn            = model_fn,
        y0                  = y0,
        t_span              = t_span,
        pulse_schedule      = pulse_schedule,
        pulse_params        = pulse_params,
        measurement_delay   = measurement_delay,
        measurement_window  = measurement_window,
        reporter_index      = reporter_index
        # ← no t_eval here
    )

    # re‐evaluate SSE for penalty calculation
    sse = fitness_pulse_count(
        candidate_params    = best_params,
        model_fn            = model_fn,
        y0                  = y0,
        t_span              = t_span,
        pulse_schedule      = pulse_schedule,
        pulse_params        = pulse_params,
        target_increment    = target_increment,
        measurement_delay   = measurement_delay,
        measurement_window  = measurement_window,
        reporter_index      = reporter_index
    )
    return best_params, sse

# --- Evaluate a topology by best‐case SSE + complexity penalty ---
def evaluate_topology(ind):
    _, sse = inner_parameter_opt(ind)
    complexity = sum(1 for e in ind if e != 0)
    return (sse + lambda_complexity * complexity,)

toolbox.register("evaluate", evaluate_topology)

def run_topology_search(pop_size=20, generations=10, cxpb=0.5, mutpb=0.2):
    pop = toolbox.population(pop_size)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

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
    return pop, logbook, hof

if __name__ == "__main__":
    pop, logbook, hof = run_topology_search()
    print("Top 5 topologies and fitness:")
    for topo in hof:
        print(list(topo), topo.fitness.values)