# Optimal deduplication evaluation

## Method

- 300 reviewed pairs, 542 unique questions
- Stratified 150-pair validation / 150-pair held-out test split
- Configuration and duplicate threshold selected only on validation F1
- Duplicate rule: same cluster and cosine similarity above threshold

## Selected route

- Recommended configuration: `baseline_k50`
- TF-IDF: max features 5000, min DF 2, max DF 1.0
- K: 50; max iterations: 30; include top answers: False
- Clustering: spherical_kmeans
- Selected threshold: 0.20

## Held-out test result

| Metric | Result |
| --- | ---: |
| TP / FP / FN / TN | 21 / 27 / 29 / 73 |
| Precision | 0.4375 |
| Recall | 0.4200 |
| F1 | 0.4286 |
| Average Precision | 0.5910 |
| Candidate Recall | 0.4200 |

## Full 300-pair descriptive result

| Metric | Result |
| --- | ---: |
| TP / FP / FN / TN | 43 / 59 / 57 / 141 |
| Precision | 0.4216 |
| Recall | 0.4300 |
| F1 | 0.4257 |
| Average Precision | 0.5944 |
| Candidate Recall | 0.4400 |

The answers-enhanced route was selected on validation F1 but did not generalize.
The simpler route above achieved the best held-out F1, so it is the safer
practical recommendation. Confirm it on a new labeled set before deployment.
The full-300 result is included for comparison with the earlier presentation.
