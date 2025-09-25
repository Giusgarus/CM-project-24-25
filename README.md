# CM Project 24-25

## Goal
This project compares two numerical methods for solving a ridge-regularized linear regression problem:

- **L-BFGS** with a strong Wolfe line search.
- **QR factorization with Householder reflections** for rectangular matrices.

We work on a real dataset provided in `csv` format, converted into `.mat` matrices, and we measure execution times and stability as the regularization parameter `\lambda` varies.

## Repository Structure
- `src/` — MATLAB/Octave implementation of the numerical methods.
  - `l_bfgs.m`, `two_loop.m`, `lineSearchStrongWolfe.m`, `compute_gradient.m` — core building blocks of the L-BFGS solver.
  - `householder.m`, `solve_ln_qr.m` — routines for QR factorization via Householder reflectors.
  - `Dataset CM.csv` — raw features and targets separated by `;`.
  - `tests/` — scripts, `.mat` data (`X.mat`, `y1.mat` … `y10.mat`), and log files used in the experiments.
- `Books/`, `Papers/` — reference material consulted during the study.
- `CM_report.pdf` — final report describing the project in detail.

## Requirements
- MATLAB (R2021a or newer recommended) or GNU Octave ≥ 7.0.
- Keep the `.m` and `.mat` files in the original directory structure.
- No external dependencies beyond the standard MATLAB/Octave toolboxes.

## Dataset
1. `src/Dataset CM.csv` contains the raw observations. Load it with `readmatrix("Dataset CM.csv", 'Delimiter', ';')`.
2. `src/tests/X.mat` and `src/tests/y*.mat` provide precomputed matrices for the experiments:
   - `X` is the feature matrix.
   - `y1` … `y10` represent different targets (e.g., measurements or time instances) to test robustness.

The experiments can be reproduced directly from the provided `.mat` files.

## Implemented Algorithms
- **L-BFGS (`l_bfgs.m`)**
  - Initializes the correction matrices `S` and `Y`, applies the two-loop recursion (`two_loop.m`), and updates the parameters with a line search that satisfies the strong Wolfe conditions (`lineSearchStrongWolfe.m`).
  - `compute_gradient.m` evaluates the gradient of the ridge-regularized least-squares problem.
- **QR with Householder reflections (`householder.m`, `solve_ln_qr.m`)**
  - `householder.m` builds the reflector vectors and the upper-triangular matrix.
  - `solve_ln_qr.m` applies the reflectors to the right-hand side to obtain `Q^T b` and solves the triangular system by back substitution.

Both approaches operate on the augmented system

```
[ X                 ]        [ y ]
[ sqrt(lambda) * I ] w =  [ 0 ]
```

which turns the ridge penalty into a standard least-squares problem.

## Running the Scripts
1. Open MATLAB/Octave and switch to the `src/tests` directory.
2. Add `src/` to the path (test scripts already call `addpath('../')`).
3. To run a solver manually:

```matlab
% Example: L-BFGS with the first configuration
load('X.mat');
load('y1.mat');
lambda = 1e-3;
[m,n] = size(X);
X_hat = [X'; sqrt(lambda) * eye(m, m)];
y_hat = [y1; zeros(m, 1)];
epsilon = 1e-5;
mem = m;       % maximum number of stored corrections
limit = 50;    % limit for columns in S/Y
w0 = zeros(m, 1);
w = l_bfgs(X_hat, w0, y_hat, mem, epsilon, limit);
```

```matlab
% Example: QR with Householder reflections
load('X.mat');
load('y1.mat');
lambda = 1e-3;
[m,n] = size(X);
X_hat = [X'; sqrt(lambda) * eye(m, m)];
y_hat = [y1; zeros(m, 1)];
[V,R] = householder(X_hat);
R_upper = triu(R(1:m, :));
qTb = solve_ln_qr(y_hat, V);
qTb = qTb(1:m);
w = R_upper \ qTb;
```

## Reproducing the Experiments
The scripts `condition_lambda_lbfgs.m` and `condition_lambda_qr.m` sweep nine orders of magnitude of `lambda` and repeat each configuration ten times:

```matlab
% Run inside src/tests
condition_lambda_lbfgs;
condition_lambda_qr;
```

For every configuration you will obtain:
- `execution_time_LBFGS.txt`, `execution_time_QR.txt` — mean execution time and variance for each `lambda`.
- `execution_time_LBFGS_*.txt`, `execution_time_QR_*.txt` — detailed logs for the ten repetitions identified by `s`.
- `results_lbfgs.txt`, `results_qr.txt` — first coefficients of the solution vector `w` for stability checks.

## Expected Output
- L-BFGS runtimes decrease significantly as `lambda` grows, while QR runtimes stay near 0.2 s with low variance.
- The recovered parameter vectors are stored in `results_*.txt` to monitor how the first coefficients evolve.

## Documentation and References
- `Books/` and `Papers/` collect the main theoretical sources.
- `CM_report.pdf` summarizes the motivation, mathematical background, results, and conclusions.

## Notes
- Scripts target MATLAB with Octave compatibility; minor numerical differences may appear in the factorization routines and in the `var` function.
- Ensure the output `.txt` files are not opened elsewhere while running the scripts to avoid write conflicts.
