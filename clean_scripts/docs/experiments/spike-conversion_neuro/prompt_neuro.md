# Role & Objective

You are an expert Computational Neuroscience & Dynamical Systems Research Engineer. You have access to our repository containing electrophysiology spike data (500 units sampled at 40 Hz, $\Delta t = 25\text{ ms}$), simultaneously recorded mouse $(x, y)$ maze coordinates, and our Manifold Dimension Expansion (MDE / Generative Manifold Network) pipeline.

**Your Goal:** Replace our current acausal sawtooth spike-to-continuous preprocessing with biophysically plausible, strictly causal, $C^2$-smooth single-neuron state models, and update the MDE candidate evaluation criteria so that the reconstructed network can run **autonomously (free-running without external inputs)** and replicate both neural population dynamics and $(x, y)$ behavioral trajectories.

---

# Working Copy: Run This Experiment in Its Own Fork (Do This First)

Build this experiment as a **fork** of the project, not as a switchable feature of the main pipeline. Work only in a separate git worktree on a new branch. Never edit the main checkout.

**This overrides two things:**
- The standing rule in the repository's `CLAUDE.md`, "Every change must be a feature you can switch off", **does not apply on the experiment branch.** Change the preprocessing and evaluation code directly. You do not need `TSLS_MDE_GMN_<FEATURE>` toggles, an "OFF equals before" guarantee or toggle-equivalence tests. The parent commit is the reference for the pre-change behaviour.
- In Deliverables, read "preserving backwards compatibility via config flags" as **one method selector inside the fork** (sawtooth baseline plus each new method). This lets the ablation run every method, the baseline included, under the same updated evaluation. The sawtooth path does not have to stay bit-identical to the parent branch.

**Setup**
1. **Base commit.** Start from the tip of `no-kEDM-gpu-optimization` in `/home/j/j-tominaga/pao_rotation`. A worktree contains only committed code, so if `git status` there shows uncommitted changes, stop and ask the user whether to commit them first.
2. **Worktree.** Create it with `git -C /home/j/j-tominaga/pao_rotation worktree add -b exp/spike-conversion-neuro /home/j/j-tominaga/pao_rotation-exp-neuro no-kEDM-gpu-optimization` and do all work there.
3. **Repoint the hard-coded checkout path.** The submit scripts and sbatch files copy code from the literal path `/home/j/j-tominaga/pao_rotation` (`PROJ_DIR`, `$HOME/pao_rotation`). Left unchanged, a job submitted from the worktree silently runs the **main checkout's** code. Run `grep -rn '/home/j/j-tominaga/pao_rotation\b\|\$HOME/pao_rotation' scripts/`. Repoint every hit in the scripts you use to submit or run jobs, at least `scripts/batch/submit.sh`, `scripts/saion/saion_env.sh`, `scripts/saion/submit_saion.sh` and the sbatch you submit, to `/home/j/j-tominaga/pao_rotation-exp-neuro`. Leave the `/bucket/...` paths, `WORK_BASE` and `SLURM_LOG_DIR` unchanged.
4. **Virtual environments.** Venvs are not in git and must not be rebuilt. Symlink them from the main checkout: `ln -s /home/j/j-tominaga/pao_rotation/.venv-gpu .venv-gpu`, plus `.venv` if you use the Deigo pipeline. Do not install into them or modify them.
5. **clean_scripts.** Turn off clean_scripts syncing on this branch. Remove the two `clean_sync.py` hooks from the worktree's `.claude/settings.json`, and do not port anything to `clean_scripts/`.
6. **Check isolation before any long job.** Each job prints the commit it staged (`commit=...` in its Slurm output), and that commit must be on `exp/spike-conversion-neuro`. Confirm this with a short test job first.

**Still binding on the branch**
- Every cluster rule in `CLAUDE.md` and `docs/cluster-operations.md`: no computation on login nodes, `srun` for tests, jobs submitted through the submit scripts, `/bucket` read-only from Saion compute nodes, and the cache read from `/bucket`.
- The cache needs no separation. Its keys include a hash of the code, so this branch never reuses the main pipeline's cache entries.
- Keep `tests/test_lagged_leaf_nodes.py` (the algorithm contract) passing. If the experiment truly has to change the contract, say so in the report.
- You may update tests that pin sawtooth-specific behaviour. Note each one in the report.

**Never** do any of the following unless the user asks:
- edit files in `/home/j/j-tominaga/pao_rotation`
- merge into `main` or `no-kEDM-gpu-optimization`
- push the branch

The other spike-conversion experiment runs in its own worktree (`/home/j/j-tominaga/pao_rotation-exp-general`, branch `exp/spike-conversion-general`). Do not touch it.

---

# Strict Architectural & Scientific Constraints (Non-Negotiable)

1. **1-to-1 Physical Interpretability:** Every neural node in the MDE network MUST correspond directly to an individual recorded unit. **Do NOT use** dimensionality reduction, latent factor models (PCA, GPFA, autoencoders), or black-box neural networks that merge units into abstract nodes.
2. **Causal Inference Preservation:** All transformations must be strictly causal ($x_i[n]$ may only depend on spikes at $t_k \le n\Delta t$). Zero future lookahead is permitted.
3. **Low-Data Regime:** Avoid introducing trainable parameters per neuron that require large datasets to fit. Use fixed, biophysically grounded differential equations with global hyperparameter sweeps.

---

# Diagnosis of Current Failure

1. **Acausality & Discontinuity:** The current sawtooth representation requires knowing the next spike time $t_{i+1}$ to compute the rising slope and introduces instantaneous vertical resets ($C^0$ discontinuities). This violates Takens' theorem ($C^2$ smoothness requirement) and makes autonomous forward rollout impossible without external spike timing inputs.
2. **Spike-Window Evaluation Bias:** Evaluating MDE candidate embeddings only on time windows near spikes prevents the model from learning the smooth manifold flow during inter-spike intervals (ISIs), causing free-running trajectories to diverge immediately between spikes.
3. **Silent Neuron Singularity:** Units that are silent for long periods collapse to $(0, 0, \dots, 0)$, creating degenerate nearest-neighbor lookups during simplex/S-map projections.

---

# Implementation & Experimentation Plan

### Step 1: Audit the Existing Pipeline

- Locate the spike preprocessing module, the MDE causal network construction module, and the autonomous rollout / evaluation scripts.
- Verify the exact data shapes, sampling alignment ($40\text{ Hz}$), and how initial conditions are seeded for free-running simulation.

### Step 2: Implement Biophysically Plausible Spike-to-Continuous Transformations

Create a modular preprocessing module (`biophysical_spike_filters.py` or equivalent) implementing the following three single-neuron transformations at $\Delta t = 25\text{ ms}$ ($40\text{ Hz}$):

1. **Model A: Dual-Compartment Synaptic Conductance & Somatic Calcium ODE ($C^2$Smooth)**
Integrate the two-stage linear ODE per neuron $i$ driven by spike train $S_i(t) = \sum \delta(t - t_k)$:
    
    $$
    \frac{dg_i}{dt} = -\frac{g_i}{\tau_{\text{syn}}} + S_i(t), \qquad \frac{dc_i}{dt} = -\frac{c_i}{\tau_{\text{Ca}}} + g_i(t)
    $$
    
    *Test parameter grid:* $\tau_{\text{syn}} \in \{50, 75\}\text{ ms}$, $\tau_{\text{Ca}} \in \{300, 500, 800\}\text{ ms}$.
    
2. **Model B: Synaptic Calcium + Slow Spike-Frequency Adaptation (AdEx-Inspired) with Naka-Rushton Saturation**
Add a slow adaptation current $w_i(t)$ ($\tau_w \in \{1500, 2500\}\text{ ms}$) to prevent silent intervals from collapsing to zero, followed by a biophysical Naka-Rushton nonlinearity to prevent high-frequency bursts from dominating Euclidean distances in delay space:
    
    $$
    \tau_w \frac{dw_i}{dt} = a \cdot c_i(t) - w_i(t) + b S_i(t)
    $$
    
    $$
    r_i(t) = \frac{\left(c_i(t) + \lambda w_i(t)\right)^n}{\sigma_i^n + \left(c_i(t) + \lambda w_i(t)\right)^n}
    $$
    
    *(Set $\sigma_i$ to each unit's median or 75th-percentile non-zero activity, $n \in \{1.5, 2.0\}$).*
    
3. **Model C: Damped Theta-Band Biophysical Oscillator (Optional / Region-Dependent)**
Model each neuron as an underdamped oscillator kicked by spikes to maintain non-zero phase velocity during brief silences:
    
    $$
    \ddot{x}_i + 2\zeta \omega_0 \dot{x}_i + \omega_0^2 x_i = S_i(t)
    $$
    
    *Test parameter grid:* $\omega_0 = 2\pi \cdot 8\text{ Hz}$, damping ratio $\zeta \in \{0.7, 0.9\}$.
    

### Step 3: Update MDE Embedding & Evaluation Criteria

1. **Full-Trajectory Evaluation:** Modify the MDE candidate scoring metric so it evaluates Pearson correlation and Normalized RMSE across the **entire continuous time series** (or weighted by animal movement velocity $\sqrt{\dot{x}^2 + \dot{y}^2}$), rather than restricting evaluation to windows around spikes.
2. **Takens Lag ($\tau$) Calibration:** Because smoothing at $40\text{ Hz}$ increases autocorrelation, run a sweep over embedding lag $\tau \in \{2, 4, 6, 8\}\text{ bins}$ ($50\text{–}200\text{ ms}$) using first minimum of mutual information / false nearest neighbors as a sanity check.
3. **Ultra-Sparse Unit Pruning:** Add an optional preprocessing filter to exclude dead or ultra-sparse units (e.g., firing rate $< 0.1\text{ Hz}$) that lack sufficient attractor coverage in low-data regimes, logging which physical unit IDs are retained.

### Step 4: Run Controlled Experiments & Report Metrics

Execute a structured ablation study comparing the baseline sawtooth method against Models A, B, and C. For each experiment, log and report:

1. **1-Step Prediction Accuracy:** Correlation & RMSE across all time bins for both neural traces and $(x, y)$ coordinates.
2. **Autonomous Free-Running Rollout Stability:** Seed the reconstructed network with initial delay vectors from a held-out test segment and run **completely free without external inputs** for $T = 2\text{ s}, 5\text{ s}, \text{and } 10\text{ s}$. Measure:
    - $(x, y)$ trajectory correlation and Euclidean path error over time.
    - Neural population state-space divergence (whether trajectories stay on the empirical manifold or blow up/collapse to a fixed point).
    - Spike-raster reconstruction (optional: thresholding/deconvolving generated $c_i(t)$ traces to compare firing rate maps against true place/task fields).
3. **Inferred Causal Graph Inspection:** Verify that the inferred directed edges between neural units and $(x, y)$ remain sparse, interpretable, and biologically plausible.

---

# Deliverables

1. Clean, modular code implementing the biophysical spike conversions and updated MDE evaluation loop (preserving backwards compatibility via config flags).
2. A summary experiment report (Markdown table + diagnostic plots saved to disk) comparing free-running rollout fidelity across models and time constants, along with your recommendation for the optimal configuration.