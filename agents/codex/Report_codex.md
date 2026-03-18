# Arts Participation Under-Engagement Analysis Report

## Purpose
This analysis aimed to identify people who may be under-engaged with physical arts activities, so public arts policy can better reach groups that might face barriers to participation. The focus was practical: build a prediction tool that helps prioritise outreach, while recognising that prediction does not prove cause.

## Data and Approach
We used a UK participation survey dataset with 34,378 records and 16 variables. After removing records with missing target outcomes, 34,338 records remained. The target was whether someone had *not* physically engaged with the arts in the last 12 months.

The approach included:
- data checks and exploratory analysis
- variable-specific handling of coded missing values
- preparation of fixed train/validation/test splits
- model development using Logistic Regression and XGBoost
- tuning with validation data only, including probability-threshold tuning to improve recall of under-engaged respondents

After cleaning, the modelling dataset included 28,995 records.

## Main Findings
Under-engagement was a minority outcome, so detecting it required balancing recall and false positives. Three models were compared on the test set:

- Baseline Logistic Regression: very high overall accuracy but poor detection of under-engaged respondents (F2 = 0.072).
- Tuned XGBoost: much better minority detection (F2 = 0.451, recall = 0.620).
- Tuned Logistic Regression: best overall policy-weighted performance (F2 = 0.464, recall = 0.713).

The tuned Logistic Regression model identified more under-engaged people than tuned XGBoost, although with modest precision (0.194). This means many flagged individuals may not actually be under-engaged, but fewer truly under-engaged people are missed.

## Final Model Choice
The final model selected was **tuned Logistic Regression**. A multi-metric selection framework was used, weighting recall and F2 most heavily because missing under-engaged groups is the costlier error for public engagement planning. Under that framework, tuned Logistic Regression scored highest (selection score 0.488), ahead of tuned XGBoost (0.465).

## Practical Implications
This model can support targeted engagement planning, for example prioritising areas or demographic profiles for outreach pilots. It is best used as a screening and prioritisation tool, not as a decision-maker on individuals.

## Limitations and Cautions
Results depend on available survey variables and coded-response handling choices. The model is predictive, not causal: it indicates statistical patterns, not why people do or do not engage. Precision is limited, so any intervention should combine model outputs with local knowledge, service design testing, and fairness monitoring across groups.
