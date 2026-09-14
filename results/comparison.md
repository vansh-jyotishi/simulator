# Comparison table

- git 98bffbf | python 3.12.10 | numpy 2.5.3 | pandas 3.0.5
- train seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | heldout seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
- schedulers: round_robin=baselines.round_robin.RoundRobin, weighted_priority=baselines.weighted_priority.WeightedPriority, clairvoyant=baselines.clairvoyant.Clairvoyant [CHEATING]
- scenario train_default.json: sha256[:8]=11652ffb
- scenario heldout_mix.json: sha256[:8]=7f297c1c
- scenario heldout_stale_intel.json: sha256[:8]=0aca9c57
- scenario blind_zone.json: sha256[:8]=0fdadbdd

| scheduler | scenario | scenario_split | seed_split | n | POI | POI_time | Pd_sensor | Pd_eff | Pfa | TTI_mean | TTI_med | int/s | pred% | pred_err | reward | blind% | switches | us_mean | us_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | train_default | train | train | 10 | 0.028 ± 0.003 | 0.004 ± 0.000 | 0.712 ± 0.085 | 0.317 ± 0.036 | 0.0009 ± 0.0006 | 4.05 ± 0.72 | 2.8 ± 1.1 | 5.2 ± 0.6 | nan | nan | -6140 ± 8 | 50.0 | 2499 | 0.8 ± 0.2 | 2.0 ± 1.0 |
| round_robin | train_default | train | heldout | 10 | 0.027 ± 0.004 | 0.004 ± 0.001 | 0.697 ± 0.048 | 0.316 ± 0.026 | 0.0009 ± 0.0008 | 4.23 ± 0.89 | 2.4 ± 0.6 | 5.1 ± 0.7 | nan | nan | -6140 ± 11 | 50.0 | 2499 | 0.8 ± 0.1 | 1.6 ± 0.4 |
| weighted_priority | train_default | train | train | 10 | 0.028 ± 0.003 | 0.004 ± 0.000 | 0.712 ± 0.085 | 0.317 ± 0.036 | 0.0009 ± 0.0006 | 4.05 ± 0.72 | 2.8 ± 1.1 | 5.2 ± 0.6 | nan | nan | -6140 ± 8 | 50.0 | 2499 | 4.0 ± 0.3 | 19.0 ± 6.2 |
| weighted_priority | train_default | train | heldout | 10 | 0.027 ± 0.004 | 0.004 ± 0.001 | 0.697 ± 0.048 | 0.316 ± 0.026 | 0.0009 ± 0.0008 | 4.23 ± 0.89 | 2.4 ± 0.6 | 5.1 ± 0.7 | nan | nan | -6140 ± 11 | 50.0 | 2499 | 3.7 ± 0.2 | 13.2 ± 2.0 |
| clairvoyant | train_default | train | train | 10 | 0.921 ± 0.005 | 0.150 ± 0.001 | 0.640 ± 0.008 | 0.553 ± 0.006 | 0.0009 ± 0.0005 | 0.81 ± 0.03 | 0.0 | 173.5 ± 2.0 | nan | nan | -1603 ± 15 | 18.0 ± 0.3 | 901 ± 13 | 2.3 ± 0.2 | 8.3 ± 1.0 |
| clairvoyant | train_default | train | heldout | 10 | 0.921 ± 0.010 | 0.149 ± 0.003 | 0.639 ± 0.013 | 0.551 ± 0.012 | 0.0010 ± 0.0006 | 0.83 ± 0.04 | 0.0 | 172.2 ± 3.1 | nan | nan | -1613 ± 27 | 17.9 ± 0.3 | 896 ± 13 | 2.3 ± 0.5 | 8.9 ± 3.3 |
| round_robin | heldout_mix | heldout | train | 10 | 0.057 ± 0.005 | 0.007 ± 0.001 | 0.626 ± 0.035 | 0.315 ± 0.020 | 0.0008 ± 0.0007 | 5.32 ± 0.47 | 4.1 ± 0.9 | 12.8 ± 1.1 | nan | nan | -6038 ± 17 | 50.0 | 2499 | 0.7 ± 0.1 | 1.4 ± 0.1 |
| round_robin | heldout_mix | heldout | heldout | 10 | 0.059 ± 0.005 | 0.007 ± 0.001 | 0.665 ± 0.049 | 0.333 ± 0.024 | 0.0009 ± 0.0009 | 5.57 ± 0.46 | 4.4 ± 0.9 | 13.2 ± 1.2 | nan | nan | -6034 ± 16 | 50.0 | 2499 | 0.7 ± 0.1 | 1.4 ± 0.2 |
| weighted_priority | heldout_mix | heldout | train | 10 | 0.057 ± 0.005 | 0.007 ± 0.001 | 0.626 ± 0.035 | 0.315 ± 0.020 | 0.0008 ± 0.0007 | 5.32 ± 0.47 | 4.1 ± 0.9 | 12.8 ± 1.1 | nan | nan | -6038 ± 17 | 50.0 | 2499 | 3.5 ± 0.3 | 12.3 ± 2.5 |
| weighted_priority | heldout_mix | heldout | heldout | 10 | 0.059 ± 0.005 | 0.007 ± 0.001 | 0.665 ± 0.049 | 0.333 ± 0.024 | 0.0009 ± 0.0009 | 5.57 ± 0.46 | 4.4 ± 0.9 | 13.2 ± 1.2 | nan | nan | -6034 ± 16 | 50.0 | 2499 | 3.7 ± 0.4 | 13.4 ± 3.5 |
| clairvoyant | heldout_mix | heldout | train | 10 | 0.909 ± 0.011 | 0.109 ± 0.002 | 0.542 ± 0.016 | 0.439 ± 0.011 | 0.0008 ± 0.0004 | 1.66 ± 0.06 | 1.0 | 203.5 ± 3.2 | nan | nan | -1516 ± 31 | 21.6 ± 0.2 | 1078 ± 9 | 2.5 ± 0.2 | 9.8 ± 1.1 |
| clairvoyant | heldout_mix | heldout | heldout | 10 | 0.910 ± 0.007 | 0.108 ± 0.001 | 0.543 ± 0.011 | 0.441 ± 0.009 | 0.0007 ± 0.0005 | 1.64 ± 0.06 | 1.0 | 202.4 ± 2.6 | nan | nan | -1520 ± 20 | 21.4 ± 0.2 | 1071 ± 11 | 2.3 ± 0.2 | 8.4 ± 0.8 |
| round_robin | heldout_stale_intel | heldout | train | 10 | 0.028 ± 0.003 | 0.004 ± 0.000 | 0.712 ± 0.085 | 0.317 ± 0.036 | 0.0009 ± 0.0006 | 4.05 ± 0.72 | 2.8 ± 1.1 | 5.2 ± 0.6 | nan | nan | -6140 ± 8 | 50.0 | 2499 | 0.7 ± 0.1 | 1.4 ± 0.1 |
| round_robin | heldout_stale_intel | heldout | heldout | 10 | 0.027 ± 0.004 | 0.004 ± 0.001 | 0.697 ± 0.048 | 0.316 ± 0.026 | 0.0009 ± 0.0008 | 4.23 ± 0.89 | 2.4 ± 0.6 | 5.1 ± 0.7 | nan | nan | -6140 ± 11 | 50.0 | 2499 | 0.7 ± 0.1 | 1.4 ± 0.2 |
| weighted_priority | heldout_stale_intel | heldout | train | 10 | 0.020 ± 0.003 | 0.003 ± 0.000 | 0.724 ± 0.067 | 0.366 ± 0.034 | 0.0005 ± 0.0004 | 4.02 ± 0.45 | 3.0 ± 0.8 | 3.8 ± 0.5 | nan | nan | -6166 ± 7 | 50.0 | 2499 | 4.0 ± 0.4 | 18.0 ± 5.7 |
| weighted_priority | heldout_stale_intel | heldout | heldout | 10 | 0.018 ± 0.004 | 0.003 ± 0.001 | 0.676 ± 0.103 | 0.343 ± 0.053 | 0.0006 ± 0.0004 | 3.99 ± 1.03 | 3.0 ± 1.5 | 3.4 ± 0.8 | nan | nan | -6170 ± 16 | 50.0 | 2499 | 3.7 ± 0.3 | 13.8 ± 2.9 |
| clairvoyant | heldout_stale_intel | heldout | train | 10 | 0.921 ± 0.005 | 0.150 ± 0.001 | 0.640 ± 0.008 | 0.553 ± 0.006 | 0.0009 ± 0.0005 | 0.81 ± 0.03 | 0.0 | 173.5 ± 2.0 | nan | nan | -1603 ± 15 | 18.0 ± 0.3 | 901 ± 13 | 2.1 ± 0.3 | 7.0 ± 1.3 |
| clairvoyant | heldout_stale_intel | heldout | heldout | 10 | 0.921 ± 0.010 | 0.149 ± 0.003 | 0.639 ± 0.013 | 0.551 ± 0.012 | 0.0010 ± 0.0006 | 0.83 ± 0.04 | 0.0 | 172.2 ± 3.1 | nan | nan | -1613 ± 27 | 17.9 ± 0.3 | 896 ± 13 | 2.3 ± 0.6 | 7.7 ± 0.8 |
| round_robin | blind_zone | train | train | 10 | 0.000 | 0.000 | nan | nan | 0.0510 ± 0.0046 | nan | nan | 0.0 | nan | nan | -2371 ± 11 | 50.0 | 999 | 0.7 ± 0.1 | 1.4 ± 0.3 |
| round_robin | blind_zone | train | heldout | 10 | 0.000 | 0.000 | nan | nan | 0.0526 ± 0.0043 | nan | nan | 0.0 | nan | nan | -2367 ± 11 | 50.0 | 999 | 0.8 ± 0.2 | 1.8 ± 0.6 |
| weighted_priority | blind_zone | train | train | 10 | 0.451 ± 0.024 | 0.451 ± 0.024 | 0.851 ± 0.045 | 0.851 ± 0.045 | 0.0487 ± 0.0040 | 0.00 | 0.0 | 22.6 ± 1.2 | nan | nan | -2178 ± 12 | 47.4 ± 0.0 | 948 | 3.8 ± 0.2 | 18.1 ± 4.1 |
| weighted_priority | blind_zone | train | heldout | 10 | 0.438 ± 0.034 | 0.438 ± 0.034 | 0.826 ± 0.065 | 0.826 ± 0.065 | 0.0525 ± 0.0053 | 0.00 | 0.0 | 21.9 ± 1.7 | nan | nan | -2172 ± 18 | 47.4 ± 0.0 | 948 | 3.6 ± 0.4 | 15.6 ± 6.0 |
| clairvoyant | blind_zone | train | train | 10 | 0.850 ± 0.047 | 0.850 ± 0.047 | 0.850 ± 0.047 | 0.850 ± 0.047 | 0.0494 ± 0.0048 | 0.00 | 0.0 | 42.5 ± 2.4 | nan | nan | -544 ± 22 | 0.1 | 1 | 1.7 ± 0.3 | 5.6 ± 1.1 |
| clairvoyant | blind_zone | train | heldout | 10 | 0.833 ± 0.040 | 0.833 ± 0.040 | 0.833 ± 0.040 | 0.833 ± 0.040 | 0.0500 ± 0.0030 | 0.00 | 0.0 | 41.6 ± 2.0 | nan | nan | -546 ± 13 | 0.1 | 1 | 1.5 ± 0.1 | 5.2 ± 0.9 |

Cells are mean ± std over seeds. `clairvoyant` reads the truth (upper bound, CHEATING). Pd_sensor excludes blind dwells, Pd_eff counts them as misses. Channels/slots are 0-based.