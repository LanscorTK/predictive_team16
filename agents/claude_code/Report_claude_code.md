# Predicting Arts Under-Engagement: Evidence for Cultural Policy

## Purpose

This analysis was conducted to support evidence-based cultural policy by identifying groups of people who are less likely to engage with the arts physically. Rather than treating non-participation as a purely personal choice, the work investigates whether patterns of under-engagement are shaped by demographic, socioeconomic, and geographic factors. The findings are intended to help government arts departments target outreach and reduce barriers to participation.

## Data and Approach

The analysis used the UK Participation Survey 2024-25, a nationally representative sample of 34,338 respondents. The outcome of interest was whether each respondent had attended or participated in the arts physically in the last 12 months. Fifteen predictor variables were used, covering education, employment, financial circumstances, internet use, housing tenure, loneliness, ethnicity, area deprivation, satisfaction with local cultural activities, age, and gender.

After cleaning the data to handle missing and non-applicable responses, approximately 29,000 records were available for modelling. Three models were trained and compared: a baseline logistic regression, a tuned logistic regression with optimised settings, and a gradient-boosted tree model (XGBoost). The data were split into training (70%), validation (15%), and test (15%) sets, with class balance preserved in each split. All models were evaluated using the same set of metrics, with particular emphasis on the ability to correctly identify under-engaged individuals.

## Main Findings

On the held-out test set, the tuned logistic regression achieved the strongest performance for the policy objective. It correctly identified 76.9% of under-engaged respondents (recall), with a balanced accuracy of 74.5% and an ROC-AUC of 0.83, indicating good overall discrimination. The baseline logistic regression detected only 11.6% of under-engaged individuals, while XGBoost detected 18.8% with higher precision but much lower recall.

## Model Choice

The tuned logistic regression was selected as the final model. It scored highest on a weighted evaluation framework that prioritised detection of under-engaged populations, while also offering the advantage of interpretability — its coefficients directly show which factors increase or decrease the likelihood of under-engagement.

## Practical Implications

The model suggests that arts under-engagement is not randomly distributed but is associated with identifiable social and demographic patterns. Policy teams could use these findings to prioritise outreach towards groups with higher predicted under-engagement, design targeted interventions, and monitor whether engagement gaps narrow over time.

## Limitations

This analysis identifies statistical associations, not causes. The model cannot determine why certain groups are less engaged, only that they are. Precision was relatively low (17%), meaning many individuals flagged as under-engaged may in fact participate — a trade-off accepted in favour of broad detection. The survey is cross-sectional, so results reflect a single time point. Missing data handling, while carefully documented, required judgement calls that could influence results. These findings should complement, not replace, qualitative research and community engagement.
