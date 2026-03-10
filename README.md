# Solver-in-the-Loop Closure for Shell Models of Turbulence

Code accompanying the paper  
"A posteriori closure of turbulence models: are symmetries preserved?"  
André Freitas, Kiwon Um, Mathieu Desbrun, Michele Buzzicotti, Luca Biferale (2026)

---

## Overview

This repository contains the code used to train and evaluate a data-driven closure model for the Sabra shell model of turbulence using a solver-in-the-loop (a posteriori) training strategy.

Instead of learning instantaneous subgrid terms (a priori), the neural network is embedded inside the numerical solver during training. The model therefore learns how its predictions affect the time evolution of the system.

The closure predicts the two unresolved shells

(u_{Nc+1}, u_{Nc+2})

from the last three resolved shells

(u_{Nc−2}, u_{Nc−1}, u_{Nc})

while the resolved shells are advanced with a fourth-order Runge–Kutta scheme.

---

## Repository structure

solver.py  
    Generates training data by integrating the fully resolved Sabra shell model.

train.py  
    Trains the neural closure using the solver-in-the-loop approach.

inf.py  
    Runs inference with a trained model and produces long LES-NN trajectories.

requirements.txt  
    Python dependencies.

---

## Installation

Create a Python environment and install the required packages:

pip install -r requirements.txt

---

## 1. Generate training data

Run the Sabra shell model solver to generate DNS trajectories:

python solver.py

The output is a dataset containing the shell velocities:

u[shell, initial_condition, time]

---

## 2. Train the neural closure

Training uses the solver-in-the-loop strategy, where the reduced system is unrolled for several time steps before computing the loss.

python train.py path_to_dataset.npz

Example:

python train.py u_40_2.npz

The script saves the trained model as a `.keras` file.

---

## 3. Run inference

Once a model is trained, inference can be performed with:

python inf.py

The script loads the trained model, evolves the reduced system forward in time, and saves the predicted trajectories.

---

## Numerical parameters

Typical parameters used in the experiments:

Number of shells (DNS): 40  
Cutoff shell Nc: 14  
Viscosity: 1e-12  
DNS timestep: 1e-8  
LES timestep: 1e-5  

---

## Citation

If you use this code in your research, please cite:

Freitas, A. et al.  
A posteriori closure of turbulence models: are symmetries preserved?  
2026

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.
