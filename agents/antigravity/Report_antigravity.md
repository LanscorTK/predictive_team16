# Arts Engagement Analysis for Cultural Policy Strategy

## Purpose of the Analysis
This report outlines the findings of a predictive modeling exercise targeting arts engagement. The objective is to identify demographic, socioeconomic, and geographic subgroups that are "under-engaged" with physical arts activities. In contrast to viewing participation as purely individual choice, diagnosing under-engagement structurally provides actionable evidence to guide targeted cultural policy, address significant social barriers, and promote a more inclusive arts strategy across the UK.

## Overview of Data and Approach
We examined survey data representing roughly 25,000 respondents from the UK Participation Survey (2024-25). The dataset contained varied indicators of individual background and lived experience, such as age, tenure status, internet connectivity, self-reported well-being, feelings of loneliness, and area deprivation levels.

To analyze this data, we built a machine learning pipeline that framed under-engagement as a complex pattern-recognition task. Recognizing that "under-engaged" individuals represent a smaller portion of the overall population (an "imbalanced" problem), our methodology specifically weighted our models to prioritize correctly identifying these individuals rather than prioritizing general accuracy. We tested standard baseline models against heavily-optimized Logistic Regression and XGBoost (advanced pattern-matching) algorithms, tuning the threshold limit to aggressively capture genuine cases of under-engagement.

## Main Findings
Our models successfully identified indicators associated with reduced arts engagement, utilizing performance metrics that strictly measure targeted subgroup identification (F2 Score). 

The baseline model proved largely ineffective at identifying under-engaged populations (F2 Score: ~0.06). However, an optimized Logistic Regression model correctly captured a far larger proportion of the under-engaged group, reaching an F2 score of 0.49 and an overall accuracy above 81%. This confirms that under-engagement is not purely random but follows structured societal predictors present in the data. Features surrounding socioeconomic hardship, neighborhood deprivation, and lower levels of education commonly appear as indicators of non-participation.

## Final Model Choice
We ultimately selected the Tuned Logistic Regression model over XGBoost. Both models performed comparably well at correctly categorizing under-engaged subgroups; however, the Logistic Regression model emerged as the preferred choice due to its superior F2 metric. Additionally, Logistic Regression offers tremendous interpretability. Because public policy design requires a transparent approach, utilizing a highly interpretable model ensures stakeholders can directly understand the exact driving factors and weight assigned to demographic traits when forecasting engagement levels.

## Practical Implications
These findings demonstrate that arts under-engagement follows highly predictable patterns tied to tangible life circumstances. For policymakers, this signals that future public arts strategies should shift towards structurally reducing barriers for these predictable demographic models, specifically addressing subgroups experiencing complex forms of deprivation, rather than solely funding broad community arts programs.

## Key Limitations and Cautions
While highly predictive, these models rely on observational correlations and do not imply direct causal relationships between demographic traits and arts participation. The results act as a diagnostic signpost, pointing researchers towards groups that need interventions but do not explain *why* the identified structural barriers prevent participation. Finally, missing responses for non-participants were removed to train the models cleanly, meaning the algorithm operates primarily on full respondent profiles and may struggle against radically incomplete future survey responses.
