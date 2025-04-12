# Pulse Counting Gene Circuit Evolution

This repository contains code and documentation for the computational evolution of a synthetic gene circuit that exhibits pulse-counting behavior. The project leverages Python-based optimization techniques—including evolutionary algorithms and gradient-based methods—to design and analyze gene network architectures with the capacity to "count" discrete stimulation events.

## Project Overview

The primary objective of this project is to develop a minimal yet robust gene circuit that can reliably count input pulses. This circuit is expected to integrate features of oscillation, memory, and decision-making, enabling potential applications in biosensing, therapeutic delivery, and developmental biology.

### Key Components:
- **Simulation Framework:** A Python-based environment for modeling gene circuits using systems of ordinary differential equations (ODEs).
- **Fitness Function:** Metrics to evaluate the accuracy and robustness of the pulse-counting behavior.
- **Optimization Algorithms:** Implementation of evolutionary algorithms (EAs) and gradient descent (or hybrid methods) to search for optimal circuit topologies and parameter configurations.
- **Analysis Tools:** Scripts for dynamic behavior analysis including time-series plots, phase portraits, and robustness testing.

## Repository Structure
```
pulse_counter_evolution/
├── notebooks/
│   └── validation.ipynb         # Jupyter notebook for simulation validation and data analysis
├── src/
│   ├── simulation.py            # ODE simulation framework and numerical integration routines
│   ├── fitness.py               # Fitness function definitions for pulse counting
│   ├── optimization.py          # Evolutionary and gradient-based optimization algorithms
│   └── models.py                # Gene circuit model definitions (e.g., repressilator, pulse counter)
├── requirements.txt             # List of Python dependencies
├── README.md                    # Project overview and documentation
└── LICENSE                      # License information (if applicable)
```


## Getting Started

### Prerequisites
- Python 3.7 or later
- [pip](https://pip.pypa.io/) for package management

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pulse_counter_evolution.git
   cd pulse_counter_evolution
   ```
2. **Set up Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate    # On Windows, use `env\Scripts\activate`
   ```
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running Simulations
- **Simulation Framework:**
  The core simulation code is in src/simulation.py. You can start by running the example models, such as the repressilator or toggle      switch, to verify the ODE solver functionality.
- **Validation Notebook**
  Open the notebooks/validation.ipynb in Jupyter Notebook or JupyterLab to run through example simulations and view the generated plots:
  ```bash
  jupyter notebook notebooks/validation.ipynb
   ```



