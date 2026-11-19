### Free-run accuracy (median over starts)

| setting | r 2 s | r 5 s | r 10 s | speed-wtd r 10 s | NRMSE 10 s | path err 2 s | hold 2 s | path err 10 s | hold 10 s |
|---|---|---|---|---|---|---|---|---|---|
| sawtooth | 0.267 | 0.119 | 0.132 | 0.134 | 0.817 | 5.54 | 7.89 | 27.12 | 33.51 |
| gamma_m3_t100 | 0.045 | 0.082 | 0.015 | 0.052 | 1.092 | 7.08 | 7.98 | 32.30 | 33.84 |
| gamma_m3_t250 | 0.118 | 0.013 | -0.013 | -0.020 | 1.132 | 7.01 | 7.89 | 36.17 | 33.51 |
| gamma_m3_t500 | 0.208 | 0.330 | 0.166 | 0.184 | 1.219 | 7.62 | 7.89 | 37.56 | 33.51 |
| gamma_m3_t500_log100 | 0.210 | 0.129 | 0.017 | -0.090 | 0.918 | 2.39 | 7.89 | 26.88 | 33.51 |
| lmu_th1000_D3 | 0.176 | -0.002 | -0.002 | 0.021 | 1.094 | 5.34 | 7.02 | 31.80 | 33.25 |
| lmu_th2000_D2_pre2 | 0.097 | 0.142 | 0.208 | 0.229 | 1.150 | 7.23 | 7.89 | 37.74 | 33.51 |
| logsig_W1000_K3 | 0.082 | -0.231 | -0.102 | -0.133 | 0.981 | 4.50 | 7.89 | 29.05 | 33.51 |
| logsig_W2000_K2_pre3 | -0.001 | 0.246 | 0.001 | 0.007 | 1.069 | 6.00 | 7.89 | 30.33 | 33.51 |


### Attractor stability (median over starts; fractions over starts)

| setting | manifold 2 s | manifold 10 s | motion 2 s | motion 10 s | collapsed 10 s | blew up 10 s | unit r 2 s |
|---|---|---|---|---|---|---|---|
| sawtooth | 0.919 | 0.851 | 0.453 | 0.054 | 0.458 | 0.000 | 0.203 |
| gamma_m3_t100 | 3.197 | 4.887 | 0.095 | 0.069 | 0.458 | 0.000 | 0.081 |
| gamma_m3_t250 | 1.001 | 1.437 | 0.027 | 0.029 | 0.917 | 0.000 | -0.049 |
| gamma_m3_t500 | 0.999 | 1.278 | 0.013 | 0.010 | 1.000 | 0.000 | 0.056 |
| gamma_m3_t500_log100 | 1.015 | 1.257 | 0.018 | 0.015 | 1.000 | 0.000 | 0.157 |
| lmu_th1000_D3 | 0.757 | 0.724 | 0.196 | 0.001 | 1.000 | 0.000 | 0.057 |
| lmu_th2000_D2_pre2 | 1.013 | 1.132 | 0.023 | 0.019 | 0.875 | 0.167 | -0.157 |
| logsig_W1000_K3 | 0.850 | 0.787 | 0.268 | 0.000 | 1.000 | 0.000 | -0.046 |
| logsig_W2000_K2_pre3 | 0.975 | 0.996 | 0.023 | 0.019 | 0.958 | 0.000 | -0.112 |


### One-step accuracy over the whole held-out segment (median over nodes)

| setting | r (x, y) | skill vs persistence (x, y) | driven units | r (driven units) | skill vs persistence (driven units) | rescored |
|---|---|---|---|---|---|---|
| sawtooth | 1.000 | 0.090 | 9 | 0.991 | 0.002 | True |
| gamma_m3_t100 | 1.000 | -0.191 | 3 | 0.998 | 0.003 | True |
| gamma_m3_t250 | 1.000 | 0.010 | 5 | 1.000 | 0.108 | True |
| gamma_m3_t500 | 1.000 | 0.048 | 4 | 1.000 | 0.002 | True |
| gamma_m3_t500_log100 | 1.000 | 0.041 | 7 | 1.000 | 0.069 | False |
| lmu_th1000_D3 | 1.000 | 0.051 | 6 | 0.994 | 0.010 | False |
| lmu_th2000_D2_pre2 | 1.000 | -0.013 | 5 | 1.000 | 0.166 | False |
| logsig_W1000_K3 | 1.000 | 0.085 | 4 | 0.978 | 0.007 | False |
| logsig_W2000_K2_pre3 | 1.000 | 0.067 | 6 | 1.000 | 0.067 | False |


### Topology of the (x, y) delay embedding: Wasserstein-1 distances

| setting | H0 gen-real | H1 gen-real | H0 lib-real (floor) | H1 lib-real (floor) | H1 bars real | H1 bars gen | pop H1 gen-real | pop H1 lib-real (floor) |
|---|---|---|---|---|---|---|---|---|
| sawtooth | 1.623 | 2.181 | 2.023 | 2.090 | 10 | 9 | 12.701 | 13.929 |
| gamma_m3_t100 | 23.991 | 2.569 | 2.023 | 2.090 | 10 | 2 | 9.467 | 11.391 |
| gamma_m3_t250 | 37.269 | 2.801 | 2.023 | 2.090 | 10 | 11 | 12.941 | 12.298 |
| gamma_m3_t500 | 13.115 | 2.569 | 2.023 | 2.090 | 10 | 6 | 10.200 | 8.068 |
| gamma_m3_t500_log100 | 13.307 | 2.529 | 2.023 | 2.090 | 10 | 5 | 14.220 | 12.128 |
| lmu_th1000_D3 | 6.173 | 2.416 | 2.023 | 2.090 | 10 | 3 | 11.162 | 7.905 |
| lmu_th2000_D2_pre2 | 69.004 | 2.882 | 2.023 | 2.090 | 10 | 6 | 12.565 | 8.628 |
| logsig_W1000_K3 | 19.618 | 2.800 | 2.023 | 2.090 | 10 | 5 | 8.891 | 9.376 |
| logsig_W2000_K2_pre3 | 22.130 | 3.186 | 2.023 | 2.090 | 10 | 2 | 9.042 | 11.395 |


### Graph (one combined network per run)

| setting | Takens lag (rows) | channels kept | real nodes | lagged leaves | edges | density | edges/node | zero-driver | within-region | target drivers |
|---|---|---|---|---|---|---|---|---|---|---|
| sawtooth | 40 | 83 | 27 | 40 | 78 | 0.048 | 1.164 | 56 | 0.256 | x: CA1_29_tsls::delay-40, CA1_2_tsls, CA1_4_tsls::delay-80, CA1_6_tsls, OFC_12_tsls::delay-80; y: CA1_11_tsls::delay-80, CA1_23_tsls::delay-40, CA1_2_tsls, CA1_6_tsls, OFC_19_tsls::delay-40, mPFC_21_tsls::delay-40, mPFC_24_tsls::delay-80 |
| gamma_m3_t100 | 23 | 83 | 8 | 13 | 21 | 0.125 | 1.000 | 16 | 0.333 | x: CA1_6_gam, CA1_6_gam::delay-23, CA1_6_gam::delay-46; y: CA1_6_gam, OFC_19_gam::delay-23 |
| gamma_m3_t250 | 40 | 83 | 23 | 22 | 53 | 0.047 | 1.178 | 38 | 0.340 | x: CA1_29_gam::delay-40, CA1_6_gam; y: CA1_23_gam::delay-40, CA1_23_gam::delay-80, CA1_6_gam, CA1_6_gam::delay-80, OFC_22_gam::delay-40, OFC_22_gam::delay-80 |
| gamma_m3_t500 | 40 | 83 | 17 | 29 | 48 | 0.062 | 1.043 | 40 | 0.354 | x: CA1_29_gam::delay-40, CA1_29_gam::delay-80, CA1_6_gam; y: CA1_23_gam::delay-40, CA1_24_gam::delay-40, CA1_26_gam::delay-80, CA1_6_gam, OFC_19_gam::delay-40, OFC_19_gam::delay-80, OFC_22_gam::delay-40 |
| gamma_m3_t500_log100 | 40 | 83 | 29 | 33 | 71 | 0.038 | 1.145 | 53 | 0.408 | x: CA1_29_gam::delay-40, CA1_29_gam::delay-80, CA1_6_gam; y: CA1_24_gam::delay-40, CA1_25_gam::delay-80, CA1_27_gam::delay-80, CA1_6_gam, OFC_19_gam::delay-40, mPFC_15_gam::delay-80, mPFC_9_gam::delay-40 |
| lmu_th1000_D3 | 36 | 83 | 22 | 25 | 54 | 0.052 | 1.149 | 39 | 0.259 | x: CA1_29_lmu::delay-36, CA1_29_lmu::delay-72, CA1_6_lmu, CA1_6_lmu::delay-36, CA1_6_lmu::delay-72, OFC_17_lmu::delay-36; y: CA1_6_lmu, OFC_19_lmu::delay-72 |
| lmu_th2000_D2_pre2 | 40 | 83 | 19 | 23 | 46 | 0.058 | 1.095 | 35 | 0.217 | x: CA1_29_lmu::delay-40, CA1_6_lmu; y: CA1_23_lmu::delay-40, CA1_25_lmu::delay-40, CA1_6_lmu, CA1_6_lmu::delay-80, OFC_19_lmu::delay-40, OFC_22_lmu::delay-40, OFC_22_lmu::delay-80 |
| logsig_W1000_K3 | 40 | 83 | 20 | 24 | 54 | 0.061 | 1.227 | 38 | 0.259 | x: CA1_6_lsig, CA1_6_lsig::delay-40, CA1_6_lsig::delay-80, OFC_17_lsig::delay-40; y: CA1_25_lsig::delay-40, CA1_26_lsig::delay-40, CA1_26_lsig::delay-80, CA1_6_lsig, OFC_19_lsig::delay-40, OFC_19_lsig::delay-80 |
| logsig_W2000_K2_pre3 | 40 | 83 | 22 | 29 | 57 | 0.045 | 1.118 | 43 | 0.281 | x: CA1_6_lsig, CA1_6_lsig::delay-40, CA1_6_lsig::delay-80; y: CA1_14_lsig::delay-40, CA1_23_lsig::delay-40, CA1_24_lsig::delay-40, CA1_6_lsig, CA1_6_lsig::delay-80, CA1_9_lsig::delay-40, mPFC_20_lsig::delay-40 |


### Paired against the sawtooth, start by start (median difference; Wilcoxon p)

Δr > 0: better (x, y) correlation than the sawtooth. Δlog path < 0: smaller path error than the sawtooth.

| setting | pairs | Δr 2 s | p 2 s | Δlog path 2 s | p 2 s | Δr 5 s | p 5 s | Δlog path 5 s | p 5 s | Δr 10 s | p 10 s | Δlog path 10 s | p 10 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gamma_m3_t100 | 24 | -0.041 | 0.565 | 0.060 | 0.684 | -0.141 | 0.422 | -0.020 | 0.922 | -0.024 | 0.422 | -0.005 | 0.643 |
| gamma_m3_t250 | 24 | 0.037 | 0.989 | 0.123 | 0.345 | -0.097 | 0.303 | 0.126 | 0.143 | -0.100 | 0.025 | 0.056 | 0.084 |
| gamma_m3_t500 | 24 | 0.024 | 0.922 | 0.134 | 0.345 | 0.052 | 0.390 | 0.032 | 0.643 | -0.076 | 0.303 | -0.027 | 0.944 |
| gamma_m3_t500_log100 | 24 | -0.004 | 0.663 | -0.187 | 0.015 | -0.013 | 1.000 | -0.118 | 0.039 | -0.080 | 0.197 | -0.119 | 0.004 |
| lmu_th1000_D3 | 24 | -0.008 | 0.944 | -0.012 | 0.790 | -0.155 | 0.095 | -0.155 | 0.390 | -0.127 | 0.074 | -0.037 | 0.643 |
| lmu_th2000_D2_pre2 | 24 | 0.030 | 0.989 | 0.053 | 0.509 | 0.069 | 0.966 | -0.072 | 0.546 | -0.107 | 0.439 | 0.034 | 0.218 |
| logsig_W1000_K3 | 24 | -0.048 | 0.527 | 0.098 | 0.726 | -0.194 | 0.027 | -0.027 | 0.623 | -0.175 | 0.107 | 0.022 | 0.290 |
| logsig_W2000_K2_pre3 | 24 | -0.035 | 0.527 | 0.114 | 0.643 | 0.001 | 0.922 | -0.006 | 0.989 | -0.019 | 0.264 | -0.078 | 0.375 |

