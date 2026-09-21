# Why NEAT Doesn't Learn to Drive

**MSc Dissertation — Data Science & Artificial Intelligence**  
---

## Overview

This repository contains the complete source code, experimental configuration,
run logs, and analysis scripts for the dissertation:

> *Why NEAT Doesn't Learn to Drive: A Systematic Empirical Investigation into
> Specification Gaming in Neuroevolutionary Autonomous Racing*

The project investigates whether NEAT (Neuroevolution of Augmenting Topologies)
can learn genuine driving competence in Gymnasium's CarRacing-v3 simulation.
The answer, established across 16 experimental runs, is that it cannot under
the configurations tested. The dissertation documents why — through a
seven-instance reward hacking taxonomy — and systematically characterises how
NEAT's speciation mechanism and population size interact with sparse,
exploitable reward signals.

---

## Repository Structure

f1-neat-dissertation/
├── src/
│ ├── environment.py # CarRacing-v3 wrapper (96x96 → 24x24 greyscale, 576 inputs)
│ ├── neat_runner.py # 5-component fitness function + eval_genome()
│ ├── parallel_eval.py # 12-worker multiprocessing evaluator
│ ├── track_distance.py # Corridor penalty — corridor_fraction(), corridor_penalty()
│ └── logger.py # Per-run JSON logging with cross-seed evaluation
├── config/
│ └── neat_config.txt # NEAT hyperparameters (pop=100, inputs=576, outputs=3)
├── results/
│ ├── runs/ # JSON logs + saved genome .pkl files for all 16 runs
│ └── figures/ # Generated dissertation figures (PNG)
├── run_experiment.py # Systematic experiment runner with CLI args
├── analyse_results.py # Loads all JSON run logs, prints summary tables
├── generate_figures.py # Generates all 6 dissertation figures from JSON data
├── replay.py # Visual replay of saved genome in CarRacing-v3 window
└── verify_setup.py # Environment verification script


---

## Setup

**Requirements:** Python 3.11+, Windows/Linux, GPU not required

```bash
git clone https://github.com/RehmanSaqib96/f1-neat-dissertation.git
cd f1-neat-dissertation
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
python verify_setup.py
```

---

## Running Experiments

```bash
# Run a single experiment (example: spec 1.5, 600 generations, rep 1)
python run_experiment.py --group D --spec 1.5 --gens 600 --rep 1

# Run population size comparison (Group C)
python run_experiment.py --group C --spec 1.5 --gens 600 --pop 50 --rep 1

# View all results summary
python analyse_results.py
```

Available CLI arguments:

| Argument | Options | Description |
|----------|---------|-------------|
| `--group` | A, B, C, D | Experimental group |
| `--spec` | 1.5, 3.0, 5.0 | Speciation compatibility threshold |
| `--gens` | any int | Number of generations |
| `--pop` | any int | Population size (default: 100) |
| `--rep` | any int | Repetition number (for logging) |

---

## Replaying a Saved Genome

```bash
# Watch the best genome drive (or stall) on seed 42
python replay.py --run D_spec1.5_gens600_rep2 --seed 42 --slow

# Watch on seed 555 (positive score — drives along straight then stalls at corner)
python replay.py --run D_spec1.5_gens600_rep2 --seed 555 --slow

# Watch on seed 7 (catastrophic failure — stalls immediately)
python replay.py --run D_spec1.5_gens600_rep2 --seed 7
```

---

## Generating Figures

```bash
python generate_figures.py
# Output: results/figures/fig1_group_a_300gen.png through fig6_species_diversity.png
```

---

## Experimental Results Summary

All 16 run JSON logs are in `results/runs/`. Key findings:

| Condition | +Seeds / Total | Mean Score | Species at End |
|-----------|---------------|------------|----------------|
| Spec 1.5, 300 gens (2 reps) | 3/20 | −47.2 | varies (1–10) |
| Spec 3.0, 300 gens (2 reps) | 3/20 | −5.1 | not fully recorded |
| Spec 5.0, 300 gens (3 reps) | 5/30 | −17.3 | not recorded |
| Spec 1.5, 600 gens (2 reps) | 4/20 | −7.42 | 10–11 (active) |
| Spec 3.0, 600 gens (2 reps) | 2/20 | −20.20 | 2 (stagnated) |
| Spec 5.0, 600 gens (2 reps) | 3/20 | −4.92 | 1 (stagnated) |
| Pop 50, spec 1.5, 600 gens | 2/10 | −4.0 | 4 (active) |
| Pop 200, spec 1.5, 600 gens | 2/10 | −2.6 | 21 (active) |

**Key finding:** Speciation threshold 3.0 consistently underperforms both lower
and higher values at 600 generations. Population size does not change which
seeds score positively but substantially improves per-seed scores at four times
the compute cost.

---

## The Seven Specification Gaming Instances

| # | Strategy | Peak Fitness | Resolution |
|---|----------|-------------|------------|
| 1 | Steer-lock grass crawling | 592.39 | Steer-lock penalty |
| 2 | Exploit-prevention over-correction | −100 all | 100-frame warmup |
| 3 | Sprint-and-crash | 573.03 | Crash discount |
| 4 | Brake-and-idle | 370–383 | Corridor-distance penalty |
| 5 | Throttle-brake cancellation | 383.84 | Net throttle bonus |
| 6 | Fluctuating-brake stall | 373.69 | Multi-seed training |
| 7 | Universal throttle-brake stall | ~210/8 seeds | Displacement check |

---

## Licence

This repository is made available for academic review purposes in conjunction
with the MSc dissertation submission. All rights reserved.
