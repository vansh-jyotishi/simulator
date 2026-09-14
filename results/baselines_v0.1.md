# Comparison table

- git 98bffbf | python 3.12.10 | numpy 2.5.3 | pandas 3.0.5
- train seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | heldout seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
- schedulers: round_robin=baselines.round_robin.RoundRobin, weighted_priority=baselines.weighted_priority.WeightedPriority
- scenario train_default.json: sha256[:8]=11652ffb
- scenario heldout_mix.json: sha256[:8]=7f297c1c

| scheduler | scenario | scenario_split | seed_split | n | POI | POI_time | Pd_sensor | Pd_eff | Pfa | TTI_mean | TTI_med | int/s | pred% | pred_err | reward | blind% | switches | us_mean | us_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | train_default | train | train | 10 | 0.028 ± 0.003 | 0.004 ± 0.000 | 0.712 ± 0.085 | 0.317 ± 0.036 | 0.0009 ± 0.0006 | 4.05 ± 0.72 | 2.8 ± 1.1 | 5.2 ± 0.6 | nan | nan | -6140 ± 8 | 50.0 | 2499 | 0.7 ± 0.1 | 1.4 ± 0.2 |
| round_robin | train_default | train | heldout | 10 | 0.027 ± 0.004 | 0.004 ± 0.001 | 0.697 ± 0.048 | 0.316 ± 0.026 | 0.0009 ± 0.0008 | 4.23 ± 0.89 | 2.4 ± 0.6 | 5.1 ± 0.7 | nan | nan | -6140 ± 11 | 50.0 | 2499 | 0.7 ± 0.1 | 1.7 ± 0.8 |
| weighted_priority | train_default | train | train | 10 | 0.028 ± 0.003 | 0.004 ± 0.000 | 0.712 ± 0.085 | 0.317 ± 0.036 | 0.0009 ± 0.0006 | 4.05 ± 0.72 | 2.8 ± 1.1 | 5.2 ± 0.6 | nan | nan | -6140 ± 8 | 50.0 | 2499 | 3.7 ± 0.8 | 15.2 ± 9.8 |
| weighted_priority | train_default | train | heldout | 10 | 0.027 ± 0.004 | 0.004 ± 0.001 | 0.697 ± 0.048 | 0.316 ± 0.026 | 0.0009 ± 0.0008 | 4.23 ± 0.89 | 2.4 ± 0.6 | 5.1 ± 0.7 | nan | nan | -6140 ± 11 | 50.0 | 2499 | 3.8 ± 0.7 | 16.2 ± 9.7 |
| round_robin | heldout_mix | heldout | train | 10 | 0.057 ± 0.005 | 0.007 ± 0.001 | 0.626 ± 0.035 | 0.315 ± 0.020 | 0.0008 ± 0.0007 | 5.32 ± 0.47 | 4.1 ± 0.9 | 12.8 ± 1.1 | nan | nan | -6038 ± 17 | 50.0 | 2499 | 0.7 ± 0.1 | 1.7 ± 0.7 |
| round_robin | heldout_mix | heldout | heldout | 10 | 0.059 ± 0.005 | 0.007 ± 0.001 | 0.665 ± 0.049 | 0.333 ± 0.024 | 0.0009 ± 0.0009 | 5.57 ± 0.46 | 4.4 ± 0.9 | 13.2 ± 1.2 | nan | nan | -6034 ± 16 | 50.0 | 2499 | 0.7 ± 0.0 | 1.3 ± 0.1 |
| weighted_priority | heldout_mix | heldout | train | 10 | 0.057 ± 0.005 | 0.007 ± 0.001 | 0.626 ± 0.035 | 0.315 ± 0.020 | 0.0008 ± 0.0007 | 5.32 ± 0.47 | 4.1 ± 0.9 | 12.8 ± 1.1 | nan | nan | -6038 ± 17 | 50.0 | 2499 | 3.8 ± 0.3 | 15.8 ± 4.7 |
| weighted_priority | heldout_mix | heldout | heldout | 10 | 0.059 ± 0.005 | 0.007 ± 0.001 | 0.665 ± 0.049 | 0.333 ± 0.024 | 0.0009 ± 0.0009 | 5.57 ± 0.46 | 4.4 ± 0.9 | 13.2 ± 1.2 | nan | nan | -6034 ± 16 | 50.0 | 2499 | 4.2 ± 1.0 | 15.2 ± 3.3 |

Cells are mean ± std over seeds. `clairvoyant` reads the truth (upper bound, CHEATING). Pd_sensor excludes blind dwells, Pd_eff counts them as misses. Channels/slots are 0-based.