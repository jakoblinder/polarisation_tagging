# Ensemble tests on the common test sets

**conservative band: 10 replicas x ~470k events (1/10 of the nominal training data).** The replica spread overestimates the uncertainty of the production networks.

Test MSE: unweighted MSE of log(1 + r_LL) on the common test set (seed-42 test split of the production files). At NLOPS it contains the 4 clamped targets (r_LL <= -1), which add ~2.2e-3 to every network.

Flags: check 1 if the mean replica test MSE exceeds 1.5 x the control; check 2 if |pull| of sigma_LL > 2.0, pull = (replica mean - control) / (replica std / sqrt(n)).

| arch | train | test | control MSE | replica MSE (mean) | ratio | check 1 | sigma control/true | sigma replica mean/true | sigma rel. spread | sigma pull | check 2 | bins: median abs pull | bins: frac abs pull > 2 |
|:--|:--|:--|--:|--:|--:|:--:|--:|--:|--:|--:|:--:|--:|--:|
| ffnn | LO | LO | 1.970e-04 | 3.219e-04 | 1.63 | FLAG | 0.9971 | 0.9939 | 0.41% | -2.5 | FLAG | 3.6 | 60% |
| autoencoder | LO | LO | 1.820e-04 | 2.908e-04 | 1.60 | FLAG | 0.9987 | 0.9964 | 0.41% | -1.8 | ok | 2.6 | 49% |
| pn | LO | LO | 2.490e-04 | 8.777e-04 | 3.52 | FLAG | 0.9993 | 0.9967 | 0.21% | -3.9 | FLAG | 4.3 | 71% |
| ffnn | LO | LOwS | 4.460e-04 | 6.394e-04 | 1.43 | ok | 0.9867 | 0.9886 | 2.23% | +0.3 | ok | 0.6 | 25% |
| ffnn | LOwS | LOwS | 2.280e-04 | 4.200e-04 | 1.84 | FLAG | 0.9998 | 0.9912 | 0.45% | -6.1 | FLAG | 5.7 | 90% |
| autoencoder | LO | LOwS | 4.530e-04 | 7.016e-04 | 1.55 | FLAG | 1.0044 | 1.0430 | 1.89% | +6.2 | FLAG | 5.9 | 96% |
| autoencoder | LOwS | LOwS | 2.010e-04 | 4.232e-04 | 2.11 | FLAG | 0.9979 | 0.9919 | 0.66% | -2.9 | FLAG | 3.5 | 70% |
| pn | LO | LOwS | 5.470e-04 | 1.003e-03 | 1.83 | FLAG | 0.9973 | 0.9913 | 0.32% | -6.0 | FLAG | 3.6 | 84% |
| pn | LOwS | LOwS | 3.260e-04 | 1.003e-03 | 3.08 | FLAG | 0.9994 | 0.9967 | 0.16% | -5.4 | FLAG | 5.6 | 91% |
| ffnn | LO | NLOPS | 3.075e-03 | 3.234e-03 | 1.05 | ok | 0.9611 | 0.9617 | 2.23% | +0.1 | ok | 0.6 | 25% |
| ffnn | LOwS | NLOPS | 2.885e-03 | 3.041e-03 | 1.05 | ok | 0.9752 | 0.9662 | 0.46% | -6.5 | FLAG | 6.2 | 90% |
| ffnn | NLOPS | NLOPS | 2.895e-03 | 3.140e-03 | 1.08 | ok | 0.9861 | 0.9826 | 1.12% | -1.0 | ok | 1.4 | 30% |
| autoencoder | LO | NLOPS | 3.085e-03 | 3.303e-03 | 1.07 | ok | 0.9779 | 1.0140 | 1.90% | +5.9 | FLAG | 5.7 | 97% |
| autoencoder | LOwS | NLOPS | 2.860e-03 | 3.048e-03 | 1.07 | ok | 0.9730 | 0.9660 | 0.66% | -3.5 | FLAG | 4.4 | 80% |
| autoencoder | NLOPS | NLOPS | 2.878e-03 | 3.100e-03 | 1.08 | ok | 0.9885 | 0.9889 | 1.62% | +0.1 | ok | 0.9 | 12% |
| pn | LO | NLOPS | 3.155e-03 | 3.552e-03 | 1.13 | ok | 0.9716 | 0.9659 | 0.33% | -5.6 | FLAG | 3.3 | 73% |
| pn | LOwS | NLOPS | 2.970e-03 | 3.547e-03 | 1.19 | ok | 0.9744 | 0.9721 | 0.15% | -4.7 | FLAG | 5.6 | 91% |
| pn | NLOPS | NLOPS | nan | 3.509e-03 | nan | n/a | 0.9974 | 0.9875 | 1.28% | -2.5 | FLAG | 4.5 | 85% |
