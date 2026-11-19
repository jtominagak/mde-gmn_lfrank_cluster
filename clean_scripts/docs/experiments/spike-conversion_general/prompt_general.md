# Role & Objective

You are an expert Applied Mathematician and Dynamical Systems Research Engineer. You have access to our repository containing sparse event time series (point processes, currently instantiated with 500 neural spike channels at 40 Hz, $\Delta t = 25\text{ ms}$), simultaneously recorded continuous state variables ($(x, y)$ coordinates), and our Manifold Dimension Expansion (MDE / Generative Manifold Network) pipeline.

**Your Goal:** Replace our current acausal sawtooth event-to-continuous preprocessing with **universal, domain-agnostic, strictly causal, $C^2$-smooth mathematical operators** grounded in LTI System Theory, Orthogonal Polynomial Projections, and Rough Path Theory. Update the MDE candidate evaluation criteria so that the reconstructed dynamical network can run **autonomously (free-running without external inputs)** and generalize to *any* sparse point process dataset (neuroscience, seismology, finance, server logs, etc.).

---

# Working Copy: Run This Experiment in Its Own Fork (Do This First)

Build this experiment as a **fork** of the project, not as a switchable feature of the main pipeline. Work only in a separate git worktree on a new branch. Never edit the main checkout.

**This overrides two things:**
- The standing rule in the repository's `CLAUDE.md`, "Every change must be a feature you can switch off", **does not apply on the experiment branch.** Change the preprocessing and evaluation code directly. You do not need `TSLS_MDE_GMN_<FEATURE>` toggles, an "OFF equals before" guarantee or toggle-equivalence tests. The parent commit is the reference for the pre-change behaviour.
- In Deliverables, read "preserving backwards compatibility via config flags" as **one method selector inside the fork** (sawtooth baseline plus each new method). This lets the ablation run every method, the baseline included, under the same updated evaluation. The sawtooth path does not have to stay bit-identical to the parent branch.

**Setup**
1. **Base commit.** Start from the tip of `no-kEDM-gpu-optimization` in `/home/j/j-tominaga/pao_rotation`. A worktree contains only committed code, so if `git status` there shows uncommitted changes, stop and ask the user whether to commit them first.
2. **Worktree.** Create it with `git -C /home/j/j-tominaga/pao_rotation worktree add -b exp/spike-conversion-general /home/j/j-tominaga/pao_rotation-exp-general no-kEDM-gpu-optimization` and do all work there.
3. **Repoint the hard-coded checkout path.** The submit scripts and sbatch files copy code from the literal path `/home/j/j-tominaga/pao_rotation` (`PROJ_DIR`, `$HOME/pao_rotation`). Left unchanged, a job submitted from the worktree silently runs the **main checkout's** code. Run `grep -rn '/home/j/j-tominaga/pao_rotation\b\|\$HOME/pao_rotation' scripts/`. Repoint every hit in the scripts you use to submit or run jobs, at least `scripts/batch/submit.sh`, `scripts/saion/saion_env.sh`, `scripts/saion/submit_saion.sh` and the sbatch you submit, to `/home/j/j-tominaga/pao_rotation-exp-general`. Leave the `/bucket/...` paths, `WORK_BASE` and `SLURM_LOG_DIR` unchanged.
4. **Virtual environments.** Venvs are not in git and must not be rebuilt. Symlink them from the main checkout: `ln -s /home/j/j-tominaga/pao_rotation/.venv-gpu .venv-gpu`, plus `.venv` if you use the Deigo pipeline. Do not install into them or modify them.
5. **clean_scripts.** Turn off clean_scripts syncing on this branch. Remove the two `clean_sync.py` hooks from the worktree's `.claude/settings.json`, and do not port anything to `clean_scripts/`.
6. **Check isolation before any long job.** Each job prints the commit it staged (`commit=...` in its Slurm output), and that commit must be on `exp/spike-conversion-general`. Confirm this with a short test job first.

**Still binding on the branch**
- Every cluster rule in `CLAUDE.md` and `docs/cluster-operations.md`: no computation on login nodes, `srun` for tests, jobs submitted through the submit scripts, `/bucket` read-only from Saion compute nodes, and the cache read from `/bucket`.
- The cache needs no separation. Its keys include a hash of the code, so this branch never reuses the main pipeline's cache entries.
- Keep `tests/test_lagged_leaf_nodes.py` (the algorithm contract) passing. If the experiment truly has to change the contract, say so in the report.
- You may update tests that pin sawtooth-specific behaviour. Note each one in the report.

**Never** do any of the following unless the user asks:
- edit files in `/home/j/j-tominaga/pao_rotation`
- merge into `main` or `no-kEDM-gpu-optimization`
- push the branch

The other spike-conversion experiment runs in its own worktree (`/home/j/j-tominaga/pao_rotation-exp-neuro`, branch `exp/spike-conversion-neuro`). Do not touch it.

---

# Strict Architectural & Mathematical Constraints (Non-Negotiable)

1. **Domain Agnosticism:** Do NOT rely on biology-specific assumptions (e.g., ion channels, refractory periods, or theta rhythms). All transformations must be purely mathematical operators valid for arbitrary unmarked or marked point processes $\mu_i(t) = \sum_k w_k \delta(t - t_k)$.
2. **1-to-1 Variable Interpretability:** Every node in the MDE causal graph MUST correspond strictly to an individual input channel $i$. **Do NOT use** cross-channel dimensionality reduction (PCA, GPFA, autoencoders) or black-box neural networks that mix variables.
3. **Strict Causality:** The transformed continuous signal $x_i[n]$ at time step $n$ may only depend on events at $t_k \le n\Delta t$. Zero future lookahead is permitted.
4. **Zero Trainable Parameters (Low-Data Regime):** Use closed-form, fixed linear time-invariant (LTI) state-space operators or integral transforms requiring zero per-channel parameter fitting.

---

# Diagnosis of Current Failure

1. **Acausality & Manifold Tearing:** The current sawtooth representation requires knowing the next event timestamp $t_{k+1}$ to compute the rising slope and introduces instantaneous vertical resets ($C^0$ discontinuities). This violates Takens' / Sauer-Yorke-Casdagli Embedology smoothness ($C^2$) requirements and prevents autonomous forward rollout without external event timing inputs.
2. **Event-Window Evaluation Bias:** Evaluating MDE candidate embeddings only on time windows near discrete events prevents the model from learning the smooth manifold flow during inter-event intervals, causing autonomous rollouts to diverge immediately between events.
3. **Inter-Event Singularity:** Channels with long silent intervals collapse to $(0, 0, \dots, 0)$, creating degenerate nearest-neighbor lookups during simplex/S-map projections.

---

# Implementation & Experimentation Plan

### Step 1: Audit the Existing Pipeline

- Locate the event preprocessing module, the MDE causal network construction module, and the autonomous rollout / evaluation scripts.
- Check whether MDE nodes accept strictly **1 scalar time series per channel** or can accept a **low-dimensional state vector per channel**, and verify how initial delay vectors are seeded for free-running simulation.

### Step 2: Implement Universal Sparse-to-Continuous Mathematical Operators

Create a modular, domain-agnostic preprocessing library (`universal_point_process_embeddings.py`) implementing the following three closed-form operators at sampling step $\Delta t$:

1. **Method 1: Cascaded Causal Gamma Filter ($C^m$Smooth LTI Observation)**
Convolve each channel's Dirac measure $\mu_i(t)$ with a causal Gamma kernel of integer order $m \ge 2$:
    
    $$
    g_m(t; \tau) = \frac{t^m e^{-t/\tau}}{\tau^{m+1} m!} \, u(t)
    $$
    
    Implemented via an exact discrete cascade of $m+1$ identical first-order leaky integrators ($\alpha = e^{-\Delta t / \tau}$).
    *Why:* Guarantees $C^m$ smoothness ($g_m(0) = g'_m(0) = 0$ for $m \ge 2$) with zero overshoot or ringing, strictly preserving topological invariants without injecting artificial cycles.
    *Test grid:* Order $m \in \{2, 3\}$, characteristic timescale $\tau \in \{100, 250, 500\}\text{ ms}$ (or scaled automatically relative to the median inter-event interval of the dataset), plus an optional multi-scale sum ($\tau_{\text{fast}} + \tau_{\text{slow}}$) and smooth compression $\log(1 + c \cdot x_i[n])$ to prevent silent-interval collapse.
    
2. **Method 2: Causal Orthogonal Legendre Memory Projection (LMU State-Space ODE)**
Project the sliding causal event history over lookback horizon $\theta$ onto shifted Legendre polynomials $P_d(s/\theta)$ of degree $d \in \{0, 1, \dots, D-1\}$ using the closed-form LTI ODE:
    
    $$
    \theta \dot{\mathbf{m}}_i(t) = A \mathbf{m}_i(t) + B \mu_i(t)
    $$
    
    Discretized via Zero-Order Hold (ZOH) or Padé approximation at $\Delta t$.
    *Why:* Optimally compresses exact inter-event timing geometry over window $\theta$ into $D$ smooth orthogonal coordinates per channel, preventing different inter-event intervals from collapsing onto degenerate points.
    *Test grid:* Horizon $\theta \in \{500, 1000, 2000\}\text{ ms}$, polynomial order $D \in \{2, 3, 4\}$ (either keeping $D$ coordinates per channel if MDE supports vector nodes, or taking the Euclidean norm / dominant projection if scalar nodes are required).
    
3. **Method 3: Causal Smoothly-Windowed Path Log-Signatures (Rough Path Theory)**
Construct the 2D time-augmented cumulative event path $X_i(t) = (t, N_i(t))$ per channel, weighted by a causal Gamma/exponential decay envelope over lookback window $W$, and compute the truncated **Log-Signature** up to depth $K \in \{2, 3\}$.
*Why:* Uniquely characterizes event ordering, clustering, and nonlinear timing interactions up to tree-like equivalences with zero domain assumptions.

### Step 3: Update MDE Embedding & Evaluation Criteria

1. **Full-Trajectory Evaluation:** Modify the MDE candidate scoring metric to evaluate Pearson correlation and Normalized RMSE across the **entire continuous time series** (or weighted by continuous target velocity $\|\dot{\mathbf{y}}\|$), rather than restricting evaluation to windows around discrete events.
2. **Scale-Normalized Takens Lag ($\tau_{\text{embed}}$) Selection:** Automatically select the embedding lag $\tau_{\text{embed}}$ from the autocorrelation decay / first minimum of mutual information of the *transformed continuous signals* rather than raw event timestamps.
3. **Observability Filtering:** Add a domain-agnostic minimum event-density threshold (e.g., excluding channels with fewer than $N_{\min}$ total events across the training split) so unobserved variables do not inject spurious edges in low-data regimes.

### Step 4: Run Controlled Experiments & Report Metrics

Execute a structured ablation study comparing the baseline sawtooth method against Methods 1, 2, and 3 on the current benchmark dataset. Log and report:

1. **1-Step Prediction Accuracy:** Correlation & NRMSE across all time steps for both event-derived channels and continuous target variables $(x, y)$.
2. **Autonomous Free-Running Rollout Stability:** Seed the reconstructed network with initial delay vectors from a held-out test segment and run **completely free without external inputs** for $T = 2\text{ s}, 5\text{ s}, \text{and } 10\text{ s}$. Measure:
    - Target $(x, y)$ trajectory correlation and Euclidean path error over time.
    - Autonomous attractor stability (verifying trajectories remain on the empirical state-space manifold without exploding or collapsing to a trivial fixed point).
3. **Topological Preservation Check (Persistent Homology / Betti Numbers):** Compare the persistence diagrams ($H_0, H_1$) of the delay-embedded trajectories generated during autonomous free runs against the ground-truth empirical manifold to verify topological fidelity.
4. **Causal Graph Sparsity & Interpretability:** Verify that inferred directed edges remain sparse and physically/structurally consistent.

---

# Deliverables

1. Clean, self-contained, domain-agnostic code implementing the mathematical operators (`universal_point_process_embeddings.py`) and updated MDE evaluation pipeline, preserving backwards compatibility via config flags.
2. A comprehensive experiment report (Markdown summary tables + diagnostic trajectory & persistence plots saved to disk) comparing all methods across timescales, along with clear guidelines on how to auto-calibrate parameters ($\tau, \theta, W$) for any new sparse point process dataset.