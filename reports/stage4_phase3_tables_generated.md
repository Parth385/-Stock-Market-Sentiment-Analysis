
# Setup A (Strict Pre-Open: 10 Market Features Base)

### TEST SET RESULTS

| Experiment | Model | Feats | Accuracy (Delta / %Delta) | ROC-AUC (Delta / %Delta) | Brier Score (Delta / %Delta) | F1-Pos | F1-Macro | Precision | Recall | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Market-Only (Reference Baseline) | Logistic Regression | 10 | 50.37% <br><small>(Base)</small> | 0.5132 <br><small>(Base)</small> | 0.2495 <br><small>(Base)</small> | 0.5404 | 0.5005 | 48.34% | 61.25% | 0.475 |
| Market-Only (Reference Baseline) | Random Forest | 10 | 48.54% <br><small>(Base)</small> | 0.4909 <br><small>(Base)</small> | 0.2500 <br><small>(Base)</small> | 0.5289 | 0.4810 | 46.89% | 60.66% | 0.470 |
| Market-Only (Reference Baseline) | XGBoost | 10 | 48.56% <br><small>(Base)</small> | 0.4882 <br><small>(Base)</small> | 0.2498 <br><small>(Base)</small> | 0.5101 | 0.4843 | 46.68% | 56.22% | 0.470 |
| Market + News Volume Control | Logistic Regression | 18 | 51.09% <br><small>(+0.72% (+1.43%))</small> | 0.5177 <br><small>(+0.0045 (+0.87%))</small> | 0.2495 <br><small>(-0.0000 (-0.01%))</small> | 0.5243 | 0.5105 | 48.84% | 56.59% | 0.475 |
| Market + News Volume Control | Random Forest | 18 | 48.52% <br><small>(-0.02% (-0.04%))</small> | 0.4931 <br><small>(+0.0022 (+0.45%))</small> | 0.2499 <br><small>(-0.0001 (-0.06%))</small> | 0.5448 | 0.4762 | 47.05% | 64.68% | 0.470 |
| Market + News Volume Control | XGBoost | 18 | 48.10% <br><small>(-0.46% (-0.94%))</small> | 0.4924 <br><small>(+0.0042 (+0.86%))</small> | 0.2497 <br><small>(-0.0001 (-0.02%))</small> | 0.5373 | 0.4733 | 46.69% | 63.27% | 0.470 |
| Market + LM Sentiment + Volume | Logistic Regression | 36 | 50.61% <br><small>(+0.24% (+0.48%))</small> | 0.5116 <br><small>(-0.0016 (-0.32%))</small> | 0.2499 <br><small>(+0.0004 (+0.15%))</small> | 0.5508 | 0.5012 | 48.58% | 63.59% | 0.470 |
| Market + LM Sentiment + Volume | Random Forest | 36 | 50.31% <br><small>(+1.76% (+3.64%))</small> | 0.5033 <br><small>(+0.0124 (+2.52%))</small> | 0.2497 <br><small>(-0.0003 (-0.12%))</small> | 0.5398 | 0.4999 | 48.29% | 61.21% | 0.470 |
| Market + LM Sentiment + Volume | XGBoost | 36 | 51.33% <br><small>(+2.77% (+5.70%))</small> | 0.5103 <br><small>(+0.0221 (+4.52%))</small> | 0.2495 <br><small>(-0.0003 (-0.12%))</small> | 0.3319 | 0.4746 | 47.93% | 25.39% | 0.470 |
| Market + FinBERT Sentiment + Volume | Logistic Regression | 40 | 51.83% <br><small>(+1.46% (+2.90%))</small> | 0.5124 <br><small>(-0.0008 (-0.16%))</small> | 0.2499 <br><small>(+0.0004 (+0.16%))</small> | 0.4852 | 0.5163 | 49.41% | 47.67% | 0.475 |
| Market + FinBERT Sentiment + Volume | Random Forest | 40 | 50.63% <br><small>(+2.09% (+4.31%))</small> | 0.4961 <br><small>(+0.0052 (+1.05%))</small> | 0.2508 <br><small>(+0.0008 (+0.32%))</small> | 0.5085 | 0.5063 | 48.35% | 53.61% | 0.475 |
| Market + FinBERT Sentiment + Volume | XGBoost | 40 | 51.98% <br><small>(+3.42% (+7.04%))</small> | 0.5005 <br><small>(+0.0123 (+2.52%))</small> | 0.2495 <br><small>(-0.0003 (-0.10%))</small> | 0.1302 | 0.3993 | 47.41% | 7.55% | 0.470 |
| Market + RoBERTa Sentiment + Volume | Logistic Regression | 40 | 50.98% <br><small>(+0.61% (+1.21%))</small> | 0.5092 <br><small>(-0.0040 (-0.78%))</small> | 0.2501 <br><small>(+0.0006 (+0.24%))</small> | 0.4882 | 0.5089 | 48.55% | 49.09% | 0.475 |
| Market + RoBERTa Sentiment + Volume | Random Forest | 40 | 49.04% <br><small>(+0.50% (+1.03%))</small> | 0.4863 <br><small>(-0.0047 (-0.95%))</small> | 0.2509 <br><small>(+0.0009 (+0.37%))</small> | 0.4960 | 0.4904 | 46.88% | 52.65% | 0.475 |
| Market + RoBERTa Sentiment + Volume | XGBoost | 40 | 49.26% <br><small>(+0.70% (+1.44%))</small> | 0.4860 <br><small>(-0.0022 (-0.45%))</small> | 0.2497 <br><small>(-0.0001 (-0.04%))</small> | 0.4480 | 0.4893 | 46.48% | 43.23% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Logistic Regression | 80 | 50.70% <br><small>(+0.33% (+0.65%))</small> | 0.5054 <br><small>(-0.0079 (-1.53%))</small> | 0.2508 <br><small>(+0.0013 (+0.50%))</small> | 0.4084 | 0.4929 | 47.65% | 35.73% | 0.480 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Random Forest | 80 | 49.06% <br><small>(+0.52% (+1.08%))</small> | 0.4858 <br><small>(-0.0051 (-1.04%))</small> | 0.2509 <br><small>(+0.0009 (+0.37%))</small> | 0.5354 | 0.4859 | 47.33% | 61.62% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | XGBoost | 80 | 47.63% <br><small>(-0.94% (-1.93%))</small> | 0.4699 <br><small>(-0.0183 (-3.74%))</small> | 0.2502 <br><small>(+0.0004 (+0.18%))</small> | 0.5060 | 0.4744 | 45.93% | 56.31% | 0.470 |

### VAL SET RESULTS

| Experiment | Model | Feats | Accuracy (Delta / %Delta) | ROC-AUC (Delta / %Delta) | Brier Score (Delta / %Delta) | F1-Pos | F1-Macro | Precision | Recall | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Market-Only (Reference Baseline) | Logistic Regression | 10 | 52.22% <br><small>(Base)</small> | 0.5123 <br><small>(Base)</small> | 0.2492 <br><small>(Base)</small> | 0.4701 | 0.5176 | 49.34% | 44.89% | 0.475 |
| Market-Only (Reference Baseline) | Random Forest | 10 | 50.22% <br><small>(Base)</small> | 0.5058 <br><small>(Base)</small> | 0.2493 <br><small>(Base)</small> | 0.4586 | 0.4989 | 47.12% | 44.66% | 0.470 |
| Market-Only (Reference Baseline) | XGBoost | 10 | 50.62% <br><small>(Base)</small> | 0.5110 <br><small>(Base)</small> | 0.2490 <br><small>(Base)</small> | 0.4545 | 0.5018 | 47.50% | 43.57% | 0.470 |
| Market + News Volume Control | Logistic Regression | 18 | 51.14% <br><small>(-1.08% (-2.08%))</small> | 0.5136 <br><small>(+0.0012 (+0.24%))</small> | 0.2491 <br><small>(-0.0001 (-0.03%))</small> | 0.4610 | 0.5071 | 48.10% | 44.26% | 0.475 |
| Market + News Volume Control | Random Forest | 18 | 50.57% <br><small>(+0.35% (+0.70%))</small> | 0.5105 <br><small>(+0.0047 (+0.93%))</small> | 0.2492 <br><small>(-0.0001 (-0.05%))</small> | 0.4818 | 0.5046 | 47.69% | 48.68% | 0.470 |
| Market + News Volume Control | XGBoost | 18 | 51.17% <br><small>(+0.54% (+1.07%))</small> | 0.5160 <br><small>(+0.0050 (+0.98%))</small> | 0.2490 <br><small>(-0.0000 (-0.00%))</small> | 0.4837 | 0.5102 | 48.28% | 48.45% | 0.470 |
| Market + LM Sentiment + Volume | Logistic Regression | 36 | 51.65% <br><small>(-0.57% (-1.09%))</small> | 0.5152 <br><small>(+0.0029 (+0.57%))</small> | 0.2496 <br><small>(+0.0004 (+0.16%))</small> | 0.5227 | 0.5164 | 48.95% | 56.08% | 0.470 |
| Market + LM Sentiment + Volume | Random Forest | 36 | 49.00% <br><small>(-1.22% (-2.43%))</small> | 0.4937 <br><small>(-0.0120 (-2.38%))</small> | 0.2496 <br><small>(+0.0002 (+0.09%))</small> | 0.4755 | 0.4896 | 46.21% | 48.97% | 0.470 |
| Market + LM Sentiment + Volume | XGBoost | 36 | 51.82% <br><small>(+1.19% (+2.36%))</small> | 0.5131 <br><small>(+0.0021 (+0.41%))</small> | 0.2492 <br><small>(+0.0002 (+0.09%))</small> | 0.2983 | 0.4657 | 47.73% | 21.70% | 0.470 |
| Market + FinBERT Sentiment + Volume | Logistic Regression | 40 | 49.32% <br><small>(-2.90% (-5.55%))</small> | 0.4896 <br><small>(-0.0227 (-4.43%))</small> | 0.2505 <br><small>(+0.0013 (+0.52%))</small> | 0.4551 | 0.4907 | 46.21% | 44.83% | 0.475 |
| Market + FinBERT Sentiment + Volume | Random Forest | 40 | 51.03% <br><small>(+0.81% (+1.62%))</small> | 0.4947 <br><small>(-0.0111 (-2.19%))</small> | 0.2501 <br><small>(+0.0008 (+0.32%))</small> | 0.4572 | 0.5056 | 47.95% | 43.69% | 0.475 |
| Market + FinBERT Sentiment + Volume | XGBoost | 40 | 52.25% <br><small>(+1.63% (+3.21%))</small> | 0.4920 <br><small>(-0.0190 (-3.72%))</small> | 0.2493 <br><small>(+0.0003 (+0.11%))</small> | 0.1496 | 0.4088 | 46.97% | 8.90% | 0.470 |
| Market + RoBERTa Sentiment + Volume | Logistic Regression | 40 | 49.84% <br><small>(-2.38% (-4.57%))</small> | 0.4955 <br><small>(-0.0168 (-3.28%))</small> | 0.2503 <br><small>(+0.0011 (+0.45%))</small> | 0.4683 | 0.4968 | 46.87% | 46.79% | 0.475 |
| Market + RoBERTa Sentiment + Volume | Random Forest | 40 | 50.30% <br><small>(+0.08% (+0.16%))</small> | 0.4990 <br><small>(-0.0068 (-1.34%))</small> | 0.2500 <br><small>(+0.0007 (+0.26%))</small> | 0.4606 | 0.4999 | 47.23% | 44.95% | 0.475 |
| Market + RoBERTa Sentiment + Volume | XGBoost | 40 | 52.38% <br><small>(+1.76% (+3.48%))</small> | 0.5208 <br><small>(+0.0098 (+1.91%))</small> | 0.2492 <br><small>(+0.0002 (+0.08%))</small> | 0.3576 | 0.4897 | 49.24% | 28.07% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Logistic Regression | 80 | 51.38% <br><small>(-0.84% (-1.61%))</small> | 0.5060 <br><small>(-0.0063 (-1.23%))</small> | 0.2502 <br><small>(+0.0011 (+0.43%))</small> | 0.4497 | 0.5071 | 48.29% | 42.08% | 0.480 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Random Forest | 80 | 49.65% <br><small>(-0.57% (-1.13%))</small> | 0.4857 <br><small>(-0.0201 (-3.97%))</small> | 0.2503 <br><small>(+0.0010 (+0.38%))</small> | 0.5105 | 0.4961 | 47.18% | 55.63% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | XGBoost | 80 | 51.49% <br><small>(+0.87% (+1.71%))</small> | 0.5161 <br><small>(+0.0051 (+0.99%))</small> | 0.2491 <br><small>(+0.0002 (+0.06%))</small> | 0.5107 | 0.5149 | 48.75% | 53.62% | 0.470 |


# Setup B (Opening Bell: 11 Market Features Base with open_gap_pct)

### TEST SET RESULTS

| Experiment | Model | Feats | Accuracy (Delta / %Delta) | ROC-AUC (Delta / %Delta) | Brier Score (Delta / %Delta) | F1-Pos | F1-Macro | Precision | Recall | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Market-Only (Reference Baseline) | Logistic Regression | 11 | 51.90% <br><small>(Base)</small> | 0.5445 <br><small>(Base)</small> | 0.2480 <br><small>(Base)</small> | 0.5419 | 0.5177 | 49.58% | 59.74% | 0.475 |
| Market-Only (Reference Baseline) | Random Forest | 11 | 52.22% <br><small>(Base)</small> | 0.5279 <br><small>(Base)</small> | 0.2488 <br><small>(Base)</small> | 0.5653 | 0.5175 | 49.88% | 65.23% | 0.475 |
| Market-Only (Reference Baseline) | XGBoost | 11 | 51.74% <br><small>(Base)</small> | 0.5211 <br><small>(Base)</small> | 0.2491 <br><small>(Base)</small> | 0.5415 | 0.5161 | 49.45% | 59.84% | 0.475 |
| Market + News Volume Control | Logistic Regression | 19 | 52.46% <br><small>(+0.57% (+1.09%))</small> | 0.5433 <br><small>(-0.0012 (-0.22%))</small> | 0.2480 <br><small>(-0.0000 (-0.02%))</small> | 0.5408 | 0.5240 | 50.08% | 58.78% | 0.475 |
| Market + News Volume Control | Random Forest | 19 | 52.75% <br><small>(+0.52% (+1.00%))</small> | 0.5322 <br><small>(+0.0043 (+0.81%))</small> | 0.2486 <br><small>(-0.0002 (-0.09%))</small> | 0.5733 | 0.5219 | 50.29% | 66.65% | 0.475 |
| Market + News Volume Control | XGBoost | 19 | 50.94% <br><small>(-0.81% (-1.56%))</small> | 0.5169 <br><small>(-0.0042 (-0.81%))</small> | 0.2497 <br><small>(+0.0006 (+0.26%))</small> | 0.5273 | 0.5087 | 48.72% | 57.46% | 0.480 |
| Market + LM Sentiment + Volume | Logistic Regression | 37 | 52.22% <br><small>(+0.33% (+0.63%))</small> | 0.5362 <br><small>(-0.0083 (-1.53%))</small> | 0.2484 <br><small>(+0.0004 (+0.15%))</small> | 0.5419 | 0.5213 | 49.87% | 59.33% | 0.475 |
| Market + LM Sentiment + Volume | Random Forest | 37 | 51.94% <br><small>(-0.28% (-0.54%))</small> | 0.5353 <br><small>(+0.0074 (+1.40%))</small> | 0.2485 <br><small>(-0.0003 (-0.12%))</small> | 0.5406 | 0.5184 | 49.62% | 59.38% | 0.475 |
| Market + LM Sentiment + Volume | XGBoost | 37 | 52.51% <br><small>(+0.76% (+1.47%))</small> | 0.5320 <br><small>(+0.0109 (+2.09%))</small> | 0.2486 <br><small>(-0.0005 (-0.19%))</small> | 0.5136 | 0.5248 | 50.13% | 52.65% | 0.480 |
| Market + FinBERT Sentiment + Volume | Logistic Regression | 41 | 53.53% <br><small>(+1.63% (+3.15%))</small> | 0.5377 <br><small>(-0.0068 (-1.26%))</small> | 0.2484 <br><small>(+0.0004 (+0.15%))</small> | 0.5153 | 0.5345 | 51.20% | 51.88% | 0.475 |
| Market + FinBERT Sentiment + Volume | Random Forest | 41 | 51.22% <br><small>(-1.00% (-1.92%))</small> | 0.5113 <br><small>(-0.0166 (-3.14%))</small> | 0.2498 <br><small>(+0.0010 (+0.40%))</small> | 0.5287 | 0.5116 | 48.97% | 57.46% | 0.475 |
| Market + FinBERT Sentiment + Volume | XGBoost | 41 | 51.76% <br><small>(+0.02% (+0.04%))</small> | 0.5141 <br><small>(-0.0070 (-1.34%))</small> | 0.2499 <br><small>(+0.0008 (+0.32%))</small> | 0.5493 | 0.5153 | 49.49% | 61.71% | 0.470 |
| Market + RoBERTa Sentiment + Volume | Logistic Regression | 41 | 51.79% <br><small>(-0.11% (-0.21%))</small> | 0.5346 <br><small>(-0.0099 (-1.82%))</small> | 0.2486 <br><small>(+0.0006 (+0.22%))</small> | 0.5150 | 0.5178 | 49.43% | 53.75% | 0.475 |
| Market + RoBERTa Sentiment + Volume | Random Forest | 41 | 51.66% <br><small>(-0.57% (-1.08%))</small> | 0.5206 <br><small>(-0.0073 (-1.38%))</small> | 0.2494 <br><small>(+0.0006 (+0.25%))</small> | 0.5081 | 0.5164 | 49.29% | 52.42% | 0.480 |
| Market + RoBERTa Sentiment + Volume | XGBoost | 41 | 52.16% <br><small>(+0.41% (+0.80%))</small> | 0.5183 <br><small>(-0.0028 (-0.54%))</small> | 0.2499 <br><small>(+0.0008 (+0.32%))</small> | 0.4811 | 0.5186 | 49.76% | 46.57% | 0.485 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Logistic Regression | 81 | 52.51% <br><small>(+0.61% (+1.18%))</small> | 0.5280 <br><small>(-0.0166 (-3.04%))</small> | 0.2491 <br><small>(+0.0011 (+0.45%))</small> | 0.4800 | 0.5215 | 50.15% | 46.02% | 0.475 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Random Forest | 81 | 50.87% <br><small>(-1.35% (-2.59%))</small> | 0.5092 <br><small>(-0.0188 (-3.55%))</small> | 0.2501 <br><small>(+0.0013 (+0.53%))</small> | 0.5473 | 0.5051 | 48.77% | 62.35% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | XGBoost | 81 | 52.48% <br><small>(+0.74% (+1.43%))</small> | 0.5228 <br><small>(+0.0017 (+0.33%))</small> | 0.2493 <br><small>(+0.0002 (+0.09%))</small> | 0.5249 | 0.5248 | 50.10% | 55.12% | 0.475 |

### VAL SET RESULTS

| Experiment | Model | Feats | Accuracy (Delta / %Delta) | ROC-AUC (Delta / %Delta) | Brier Score (Delta / %Delta) | F1-Pos | F1-Macro | Precision | Recall | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Market-Only (Reference Baseline) | Logistic Regression | 11 | 54.44% <br><small>(Base)</small> | 0.5436 <br><small>(Base)</small> | 0.2483 <br><small>(Base)</small> | 0.4895 | 0.5391 | 51.97% | 46.27% | 0.475 |
| Market-Only (Reference Baseline) | Random Forest | 11 | 53.50% <br><small>(Base)</small> | 0.5424 <br><small>(Base)</small> | 0.2480 <br><small>(Base)</small> | 0.5281 | 0.5349 | 50.69% | 55.11% | 0.475 |
| Market-Only (Reference Baseline) | XGBoost | 11 | 53.69% <br><small>(Base)</small> | 0.5456 <br><small>(Base)</small> | 0.2479 <br><small>(Base)</small> | 0.5116 | 0.5356 | 50.94% | 51.38% | 0.475 |
| Market + News Volume Control | Logistic Regression | 19 | 53.47% <br><small>(-0.98% (-1.79%))</small> | 0.5395 <br><small>(-0.0042 (-0.76%))</small> | 0.2483 <br><small>(-0.0001 (-0.03%))</small> | 0.4957 | 0.5319 | 50.75% | 48.45% | 0.475 |
| Market + News Volume Control | Random Forest | 19 | 53.60% <br><small>(+0.11% (+0.20%))</small> | 0.5433 <br><small>(+0.0009 (+0.16%))</small> | 0.2480 <br><small>(-0.0000 (-0.01%))</small> | 0.5330 | 0.5360 | 50.78% | 56.08% | 0.475 |
| Market + News Volume Control | XGBoost | 19 | 54.63% <br><small>(+0.95% (+1.77%))</small> | 0.5508 <br><small>(+0.0052 (+0.95%))</small> | 0.2475 <br><small>(-0.0004 (-0.17%))</small> | 0.5033 | 0.5429 | 52.09% | 48.68% | 0.480 |
| Market + LM Sentiment + Volume | Logistic Regression | 37 | 52.55% <br><small>(-1.90% (-3.48%))</small> | 0.5317 <br><small>(-0.0119 (-2.19%))</small> | 0.2487 <br><small>(+0.0004 (+0.16%))</small> | 0.5129 | 0.5252 | 49.76% | 52.93% | 0.475 |
| Market + LM Sentiment + Volume | Random Forest | 37 | 52.71% <br><small>(-0.79% (-1.47%))</small> | 0.5326 <br><small>(-0.0098 (-1.82%))</small> | 0.2484 <br><small>(+0.0004 (+0.15%))</small> | 0.5089 | 0.5264 | 49.92% | 51.89% | 0.475 |
| Market + LM Sentiment + Volume | XGBoost | 37 | 52.95% <br><small>(-0.73% (-1.36%))</small> | 0.5352 <br><small>(-0.0104 (-1.90%))</small> | 0.2484 <br><small>(+0.0004 (+0.17%))</small> | 0.4882 | 0.5265 | 50.18% | 47.53% | 0.480 |
| Market + FinBERT Sentiment + Volume | Logistic Regression | 41 | 51.14% <br><small>(-3.31% (-6.07%))</small> | 0.5112 <br><small>(-0.0325 (-5.97%))</small> | 0.2496 <br><small>(+0.0013 (+0.53%))</small> | 0.4876 | 0.5103 | 48.28% | 49.25% | 0.475 |
| Market + FinBERT Sentiment + Volume | Random Forest | 41 | 52.71% <br><small>(-0.79% (-1.47%))</small> | 0.5288 <br><small>(-0.0136 (-2.51%))</small> | 0.2488 <br><small>(+0.0008 (+0.33%))</small> | 0.5234 | 0.5271 | 49.92% | 54.99% | 0.475 |
| Market + FinBERT Sentiment + Volume | XGBoost | 41 | 52.60% <br><small>(-1.08% (-2.02%))</small> | 0.5342 <br><small>(-0.0114 (-2.10%))</small> | 0.2484 <br><small>(+0.0004 (+0.18%))</small> | 0.5475 | 0.5249 | 49.84% | 60.73% | 0.470 |
| Market + RoBERTa Sentiment + Volume | Logistic Regression | 41 | 51.52% <br><small>(-2.93% (-5.38%))</small> | 0.5130 <br><small>(-0.0306 (-5.63%))</small> | 0.2495 <br><small>(+0.0011 (+0.46%))</small> | 0.4948 | 0.5144 | 48.69% | 50.29% | 0.475 |
| Market + RoBERTa Sentiment + Volume | Random Forest | 41 | 53.44% <br><small>(-0.05% (-0.10%))</small> | 0.5386 <br><small>(-0.0039 (-0.71%))</small> | 0.2483 <br><small>(+0.0003 (+0.11%))</small> | 0.4878 | 0.5305 | 50.74% | 46.96% | 0.480 |
| Market + RoBERTa Sentiment + Volume | XGBoost | 41 | 53.90% <br><small>(+0.22% (+0.40%))</small> | 0.5439 <br><small>(-0.0017 (-0.30%))</small> | 0.2479 <br><small>(-0.0000 (-0.01%))</small> | 0.4771 | 0.5325 | 51.36% | 44.55% | 0.485 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Logistic Regression | 81 | 51.49% <br><small>(-2.95% (-5.43%))</small> | 0.5212 <br><small>(-0.0224 (-4.13%))</small> | 0.2493 <br><small>(+0.0010 (+0.40%))</small> | 0.4932 | 0.5140 | 48.66% | 50.00% | 0.475 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | Random Forest | 81 | 51.98% <br><small>(-1.52% (-2.84%))</small> | 0.5241 <br><small>(-0.0183 (-3.37%))</small> | 0.2491 <br><small>(+0.0011 (+0.44%))</small> | 0.5378 | 0.5191 | 49.28% | 59.18% | 0.470 |
| Market + All Sentiment (LM+FinBERT+RoBERTa) + Volume | XGBoost | 81 | 53.06% <br><small>(-0.62% (-1.16%))</small> | 0.5341 <br><small>(-0.0115 (-2.12%))</small> | 0.2485 <br><small>(+0.0006 (+0.24%))</small> | 0.5197 | 0.5304 | 50.27% | 53.79% | 0.475 |

