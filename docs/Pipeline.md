**数据**

Participation
Survey是一项面向英格兰16岁及以上成人的持续性调查，旨在为英国数字、文化、媒体和体育部（DCMS）提供关于公众参与文化、数字和体育活动的权威代表性数据与分析。

Link:
<https://datacatalogue.ukdataservice.ac.uk/studies/study/9459#details>

**任务**

从原始数据集中选取了目标变量 CARTS_NET
以及十个特征变量，构建了实验用sub-dataset

Variable = CARTS_NET Label = In the last 12 months, engaged (attended OR
participated) with the arts physically

Value = -3.0 Label = Not applicable

Value = 1.0 Label = Yes

Value = 2.0 Label = No

Value = 3.0 Label = No & Missing

剔除value=3及-3的行，构建binary classification任务

**意义**

我们将此任务定义为一个具有社会研究价值的参与不足识别问题。该任务并非将艺术参与视为纯粹的个人偏好，而是探究不参与是否在人口统计、社会经济、数字技术和地理因素等方面呈现出社会模式。其目的是识别可能面临结构性或环境性障碍而无法参与实体艺术活动的群体，并为制定更具包容性的文化政策和公众参与策略提供依据。

**实验设计**

包含5个模块的end-to-end机器学习数据分析流程，给不同的ai
agent相同的prompt指令，确保对比的公平性。依次按顺序输入prompt要求ai完成相应任务，除非出现报错否则不额外干预。

为了确保agent都能基本完成相应任务，避免表现差异太大无法横向比较（e.g.,
只给一个prompt：完成...预测任务，则notebook结构可能都完全不同，难以比较），因此prompt设计需要有一定的明确性。同时，prompt又不能过于细致，这样模型可能表现又过于相似，不具备区分度。Prompt设计尽可能在这两点上达到平衡。

**备注**

后文中的success
criteria都是相对客观、基础的criteria，而不涉及实现质量。比如data
cleaning，糟糕的结果可能是删去了所有的含有无效值的行，虽然数据仍然能训练，但这会损失大量数据，甚至造成潜在的selection
bias。EDA的图像及分析指标的选择，也可能存在是否合适、提供insight多少的区别。这些需要主观判断的内容，或许要在报告第三章Comparative
analysis of agent tooling中想办法比较。

**\**

**0 Setup**

**主要任务：**

明确当前目录中的文件（不导入数据------下一个section导入）：变量字典；数据文件

创建Jupyter Notebook：experiment\_\[agent name\].ipynb (e.g.
experiment_codex.ipynb)

固定随机种子=42，后续所有涉及随机random_state的地方都用这个，确保实验可重复

确保notebook从头到尾可按顺序执行；所有文件都从当前目录读取，输出到当前目录。

**Prompt: 红色部分请填入相应测试的agent名**

0 Setup (section 0/step 0)

Work in the current directory and assume the following files are
available:

1\. Variable dictionary:
participation_2024-25_data_dictionary_cleaned.txt

2\. Data file: participation_2024-25_experiment.tab

Read the variable dictionary to understand the meanings of the variables
and their coded values (just try to understand, no need creating note in
the notebook). Do not load the data file yet in this step.

Create a Jupyter Notebook named \`experiment\_\[agent_name\].ipynb\`.

Add a markdown cell at the top of the notebook containing the main
title: \# Experiment: \[agent_name\] and the name of this section, with
correct level.

Set the global random seed to 42. Use \`random_state=42\` in all later
steps whenever randomness is involved, to ensure reproducibility.

Use only relative paths. Read all inputs from the current directory and
save all outputs to the current directory.

Ensure the notebook can run sequentially from top to bottom without any
manual intervention.

Also set up the following experiment-tracking structure in the current
directory:

\- a run log file named \`run_log\_\[agent_name\].md\`

\- an evidence folder named \`evidence\_\[agent_name\]\`

Use the run log to record progress step by step throughout the
experiment. For each major step, record:

\- step name

\- completion status

\- key actions

\- key outputs

\- important warnings or errors, if any

Save all major output files generated during the experiment in the
evidence folder, unless a later step specifies a different required
location.

If any failure prevents reliable continuation of the pipeline, mark it
clearly in the run log as \`EARLY FAILURE\`, and state:

\- which step failed

\- the reason for failure

\- which later steps were affected

**Success Criteria**

**Objective criteria (basic function)**

1.  The notebook is created with the correct filename:
    experiment\_\[agent_name\].ipynb

2.  The first cell is a markdown cell containing the main title \#
    Experiment: \[agent_name\]

3.  The notebook includes the section title for Step 0 with an
    appropriate heading level

4.  The two required files are acknowledged in this step:

5.  The variable dictionary is consulted in this step, but the data file
    is not loaded

6.  A fixed seed of 42 is defined for later use

7.  Only relative paths are used

8.  The notebook is structured to run sequentially from top to bottom
    without manual intervention

9.  A run log file named run_log\_\[agent_name\].md is created

10. An evidence folder named evidence\_\[agent_name\] is created

11. The run log includes the required fields for step tracking

12. The run log specifies how EARLY FAILURE should be recorded

**Quality criteria**

**1.** Setup clarity and completeness

What is being judged: whether the setup is clear, complete, and usable
for the rest of the pipeline.

- 5 = Setup is complete, clear, and directly supports later steps with
  no ambiguity

- 4 = Setup is mostly complete and clear, with only minor omissions or
  awkwardness

- 3 = Setup is usable but somewhat incomplete, vague, or inconsistent

- 2 = Setup is only partially usable and leaves important points unclear

- 1 = Setup is confusing, incomplete, or not usable as a reliable
  starting point

**2.** Reproducibility readiness

What is being judged: whether the setup establishes strong foundations
for reproducible execution.

- 5 = Reproducibility measures are clearly and correctly established,
  including seed control, path discipline, and sequential execution
  readiness

- 4 = Reproducibility is mostly well established, with only minor
  weaknesses

- 3 = Some reproducibility measures are present, but important details
  are weak or incomplete

- 2 = Reproducibility is addressed superficially and would likely cause
  issues later

- 1 = Reproducibility is poorly addressed or effectively absent

**3.** Logging and evidence setup quality

What is being judged: whether the run log and evidence structure are
well prepared for tracking and auditing the experiment.

- 5 = Logging and evidence setup are clear, well-structured, and
  practical for later monitoring and comparison

- 4 = Logging and evidence setup are solid, with only minor limitations

- 3 = Basic logging and evidence structures exist, but they are not
  fully clear or robust

- 2 = Logging and evidence structures are weak, incomplete, or difficult
  to use

- 1 = Logging and evidence setup are missing, unclear, or unusable

**\**

**1 Dataset Ingestion + Schema Checks + Problem Definition**

**这一部分做的事：**

**1.1 Dataset ingestion & Schema checks**

用pandas读取数据文件存入dataframe，返回行数列数。

预期数据文件中包含变量CARTS_NET, AGEBAND, SEX, QWORK, EDUCAT3, FINHARD,
CINTOFT, gor, rur11cat, CHILDHH, COHAB。请做必要的**Schema
checks**并记录入markdown cell你做了哪些检查。

创建markdown cell，包含一级、二级标题及该部分具体做的工作。随后创建code
cell代码实现。

**1.2 Problem definition**

明确**预测任务**：binary classification
task（是否参加过艺术活动），目标变量，和特征变量。

明确任务背景

补充信息：对于目标变量，我们将drop掉值为-3和3的所有行（缺失值），使其成为二分类预测任务。这里先不drop，只是在markdown中说明。

创建markdown
cell，包含二级标题及预测任务的定义，以及一个表格，包含所有变量的介绍。

**Prompt:**

1 Dataset Ingestion + Schema Checks + Problem Definition (section 1/step
1)

1.1 Dataset ingestion and schema checks

Read the data file \`participation_2024-25_experiment.tab\` into a
pandas dataframe called participation_raw, and report its number of rows
and columns.

The dataset is expected to contain the following variables:

\`CARTS_NET, AGEBAND, SEX, QWORK, EDUCAT3, FINHARD, CINTOFT, gor,
rur11cat, CHILDHH, COHAB\`

Perform appropriate schema checks for these variables and document in a
markdown cell what checks were performed. At minimum, confirm that the
required variables are present and check for any obvious schema issues
relevant to this step.

Create a markdown cell for this subsection that includes:

\- a level 1 heading

\- a level 2 heading

\- a short description of the work completed in this subsection

Then create a code cell that implements this step.

1.2 Problem definition

Create a markdown cell with a level 2 heading defining the prediction
task.

The task is a binary classification task: predict whether a respondent
engaged with the arts physically in the last 12 months.

We frame this task as an under-engagement identification problem with
social research value. Rather than treating arts participation as a
purely individual preference, the task investigates whether
non-participation is socially patterned across demographic,
socioeconomic, digital, and geographic factors. The purpose is to
identify groups that may face structural or contextual barriers to
physical arts engagement, and to provide evidence for more inclusive
cultural policy and public engagement strategies.

Define:

\- target variable: \`CARTS_NET\`

\- feature variables: \`AGEBAND, SEX, QWORK, EDUCAT3, FINHARD, CINTOFT,
gor, rur11cat, CHILDHH, COHAB\`

Also state clearly in the markdown cell that for the target variable,
rows with values \`-3\` and \`3\` will later be dropped as missing
values so that the task becomes a binary classification problem. Do not
drop them yet in this step.

In the same markdown cell, include a table introducing all variables,
based on the variable dictionary file
\`participation_2024-25_data_dictionary_cleaned.txt\`.

**Success Criteria**

**Objective criteria (basic function)**

1.  The data file is read successfully into a pandas dataframe

2.  The notebook reports the dataset shape correctly: number of rows and
    columns

3.  All required variables are checked and confirmed to exist, or any
    missing variables are clearly reported

4.  Appropriate schema checks are performed for the specified variables
    and the checks are documented in a markdown cell

5.  A markdown cell is created for Section 1.1 with the required level 1
    heading, level 2 heading, and brief description

6.  A code cell is created that implements dataset ingestion and schema
    checks

7.  A markdown cell is created for Section 1.2 with a clear definition
    of the binary classification task

8.  The target variable and all ten feature variables are stated
    correctly

9.  The markdown clearly notes that rows with CARTS_NET values -3 and 3
    will be dropped later, but are not dropped in this step

10. A variable summary table is included and matches the variable
    dictionary file

**Quality criteria**

**1.** Quality of schema checks

What is being judged: whether the schema checks are sufficiently
comprehensive, correct, and relevant to this prediction task.

- 5 = Checks go well beyond basic variable presence. They cover dataset
  shape, required variable presence, and at least two of the following:
  if there is duplicate column, observed values, coding consistency, or
  obvious structural issues. The checks are clearly reported and useful
  for later modelling.

- 4 = Checks include dataset shape, required variables, and at least one
  additional useful check, but miss one important aspect.

- 3 = Checks cover the basics only: dataset shape, required variable
  presence, and one limited additional check. Coverage is adequate but
  narrow.

- 2 = Checks are minimal: only dataset shape and required variable
  presence are confirmed, with little or no further inspection.

- 1 = Checks are missing, very unclear, or incorrect.

**2.** Quality of problem definition

What is being judged: whether the prediction task is defined accurately,
clearly, and with meaningful substantive context.

- 5 = The task is defined clearly and accurately, including the
  prediction goal, target, features, later target cleaning rule, and a
  clear statement of the task's real-world or policy relevance.

- 4 = The task definition is clear and mostly complete, with some
  meaningful substantive context, but the practical significance is not
  fully developed.

- 3 = The task definition is broadly correct, but somewhat generic,
  incomplete, or weakly connected to the task's real-world significance.

- 2 = The task definition is partly correct but vague, with little
  meaningful context or practical motivation.

- 1 = The task definition is unclear, inaccurate, or poorly aligned with
  the task.

**3.** Quality of variable description table

What is being judged: whether the variable table is useful, accurate,
and grounded in the variable dictionary.

- 5 = The table is complete, accurate, and clearly based on the variable
  dictionary. It is easy to use for later analysis.

- 4 = The table is mostly accurate and complete, with only minor
  omissions or wording issues.

- 3 = The table is usable but somewhat incomplete, uneven, or imprecise.

- 2 = The table has noticeable omissions, inaccuracies, or limited
  usefulness.

- 1 = The table is missing or seriously inaccurate.

**4.** Quality of notebook communication and structure

What is being judged: whether the section is well organised and easy to
follow as part of a reproducible analysis notebook.

- 5 = Markdown and code are very well structured, with clear headings,
  logical flow, and concise explanations.

- 4 = The section is clear and readable overall, with only minor issues
  in flow or organisation.

- 3 = The section is understandable but somewhat uneven in structure or
  explanation.

- 2 = The section is hard to follow in places because of weak
  organisation or unclear markdown.

- 1 = The section is poorly organised or confusing.

**\**

**2 EDA and insight generation**

**主要任务：**

首先将目标变量转化为二分类变量：drop掉缺失值。先不用重新编码为0和1。

将删除掉原始变量，增加了新的二分类变量的数据copy到participation_eda供后续使用。

进行探索性数据分析，以提供必要的数据分析洞见，包含可视化。

**Prompt:**

2 EDA and Insight Generation (section 2/step 2)

First, convert the target into a binary classification target for later
use. Remove all rows where \`CARTS_NET\` takes values \`-3\` or \`3\`,
as these are treated as missing target values. Do not recode the
remaining target values to \`0\` and \`1\` yet.

Create a copy of the data for subsequent steps: drop the original
\`CARTS_NET\` column and add a new binary target variable to this new
dataframe called participation_eda.

Create a new folder named \`EDA\_\[agent_name\]\_Pics\` in the evidence
folder created in section 0. Save all figures generated in this step as
\`.png\` files in that folder.

Perform appropriate exploratory data analysis for the prediction task,
based on the variable definitions and roles. The EDA should generate
useful data insights and must include suitable visualisations using
appropriate graph types.

Organise this section clearly in the notebook using markdown and code
cells. Include correct heading levels and add subheadings where helpful.
Follow a readable flow in which markdown cells explain the work and code
cells implement it.

Do not perform any modelling, training, or model evaluation in this
step.

**Success Criteria**

**Objective criteria (basic function)**

1.  Rows with CARTS_NET values -3 and 3 are removed correctly

2.  A new dataframe is created for subsequent steps

3.  The original CARTS_NET column is dropped from the new dataframe

4.  A new binary target variable is added to the new dataframe

5.  The folder EDA\_\[model_name\]\_Pics is created in the current
    directory

6.  All EDA figures are saved as .png files in that folder

7.  The EDA is relevant to the prediction task and includes appropriate
    visualisations

8.  Markdown and code cells are used in a clear, readable sequence

9.  Headings and subheadings are structured correctly

10. No modelling, training, or model evaluation is performed in this
    step

**Quality criteria**

**1.** Quality of EDA coverage and task relevance

What is being judged: whether the EDA is sufficiently relevant to the
prediction task and covers the most important aspects of the data.

- 5 = The EDA is clearly focused on the prediction task and covers the
  key elements needed for later modelling, including target
  distribution, feature distributions, and feature-target relationships.

- 4 = The EDA is mostly relevant and covers several important aspects,
  but misses one useful area of analysis.

- 3 = The EDA is broadly relevant but somewhat limited, uneven, or too
  descriptive.

- 2 = The EDA has weak focus on the prediction task and covers only a
  narrow or superficial set of analyses.

- 1 = The EDA is poorly targeted, largely uninformative, or not
  meaningfully connected to the task.

**2.** Quality of visualisation choices

What is being judged: whether the visualisations are appropriate for the
variables, readable, and useful for interpretation.

- 5 = Visualisations are well chosen for the data types and task,
  clearly labelled, readable, and genuinely helpful for interpretation.

- 4 = Visualisations are mostly appropriate and readable, with only
  minor weaknesses in choice or presentation.

- 3 = Visualisations are usable but somewhat repetitive, suboptimal, or
  only moderately informative.

- 2 = Visualisations are poorly chosen for the variables, hard to read,
  or add limited analytical value.

- 1 = Visualisations are missing, seriously inappropriate, or very
  difficult to interpret.

**3.** Quality of insights and modelling awareness

What is being judged: whether the EDA produces meaningful insights and
identifies issues relevant to later modelling.

- 5 = Insights are accurate, useful, and well connected to the modelling
  task. The section identifies important risks or considerations such as
  class imbalance, sparse categories, skewed distributions, or other
  issues likely to affect preprocessing or model performance.

- 4 = Insights are mostly useful and accurate, and identify at least one
  important modelling consideration, but do not develop all relevant
  implications.

- 3 = Insights are broadly reasonable but fairly descriptive, with
  limited attention to modelling implications.

- 2 = Insights are weak, generic, or only loosely supported by the
  analysis, with little awareness of later modelling risks.

- 1 = Insights are missing, misleading, or not supported by the EDA.

**4.** Quality of notebook communication and structure

What is being judged: whether the section is well organised and easy to
follow as part of a reproducible analysis workflow.

- 5 = Markdown and code are very well structured, with clear headings,
  logical flow, and concise explanations that support interpretation.

- 4 = The section is clear and readable overall, with only minor issues
  in flow or organisation.

- 3 = The section is understandable but somewhat uneven in structure or
  explanation.

- 2 = The section is hard to follow in places because of weak
  organisation or unclear markdown.

- 1 = The section is poorly organised or confusing.

**3 Missingness handling**

**主要任务**

基于变量字典，建立缺失值处理规则。

将处理完缺失值的数据copy到participation_clean中

**Prompt:**

3 Missingness Handling (section 3/step 3)

In this dataset, missing values are not stored as \`NaN\`, but are often
represented by negative codes. For this step, focus only on the feature
variables, since the target variable has already been handled earlier.

Using the variable dictionary
\`participation_2024-25_data_dictionary_cleaned.txt\`, review the
meanings of the coded values for each feature variable and determine how
to handle values that do not carry useful information for modelling,
such as missing, not applicable, or similar non-informative responses.

You may need to define variable-specific missingness handling rules,
depending on the missing rate and meanings of different coded values.
Clearly explain these handling rules in markdown.

Organise this section clearly in the notebook using appropriate heading
levels. Follow a readable flow in which markdown cells explain the work
and code cells implement it.

Print necessary information, such as rows before and after cleaning.

Conduct the missingness handling job on the data frame
"participation_eda" created in the last section, and the expected output
is a cleaned dataset "participation_clean"with no missing value, ready
for encoding and training.

**Success Criteria**

**Objective criteria (basic function)**

1.  Missingness handling is performed on participation_eda

2.  Missing-value rules are defined using the variable dictionary

3.  Rules are explained clearly in markdown

4.  Variable-specific handling is applied where needed

5.  Rows before and after cleaning are reported

6.  The cleaned output is saved as participation_clean

7.  participation_clean contains no missing values in the feature
    variables

8.  The cleaned dataset is ready for encoding and model training

9.  Markdown and code cells are organised clearly with appropriate
    headings

**Quality criteria**

**1.** Quality of missingness handling strategy

What is being judged: whether missing and non-informative coded values
are handled appropriately for the prediction task, without unnecessary
data loss or obviously poor decisions.

- 5 = Handling is thoughtful and variable-specific. The approach
  considers both code meaning and missingness rate, avoids unnecessary
  row dropping, and makes sensible distinctions between dropping,
  recoding as missing, or retaining values where appropriate.

- 4 = Handling is generally sound and mostly variable-specific, with
  only minor weaknesses such as one slightly over-simplified or weakly
  justified decision.

- 3 = Handling is broadly workable but somewhat blunt. Some
  variable-specific reasoning is present, but the approach is limited,
  inconsistent, or causes avoidable data loss.

- 2 = Handling is simplistic or poorly suited to the task, for example
  relying heavily on row deletion or using the same treatment for most
  variables without enough justification.

- 1 = Handling is clearly poor, causing major unnecessary data loss,
  obvious bias risk, or serious misuse of coded values.

**2.** Quality of rule definition and justification

What is being judged: whether the missingness rules are clearly defined,
interpretable, and grounded in the variable dictionary.

- 5 = Rules are clear, well organised, and explicitly linked to the
  coded meanings in the variable dictionary. The rationale is easy to
  follow.

- 4 = Rules are mostly clear and reasonably justified, with only minor
  gaps in explanation or organisation.

- 3 = Rules are present and partly justified, but some decisions are
  vague, weakly explained, or not clearly tied to the dictionary.

- 2 = Rules are incomplete, weakly justified, or difficult to follow.

- 1 = Rules are missing, very unclear, or not meaningfully justified.

**3.** Quality of cleaned output for downstream modelling

What is being judged: whether participation_clean appears well prepared
for later encoding and model training.

- 5 = participation_clean is clearly usable for downstream modelling,
  with no unresolved feature missingness and no obvious cleaning
  inconsistencies.

- 4 = participation_clean is mostly ready for modelling, with only minor
  issues or weak spots.

- 3 = participation_clean is usable but somewhat uneven, with minor
  remaining concerns about consistency or preparation quality.

- 2 = participation_clean has noticeable weaknesses that may create
  problems in later encoding or modelling.

- 1 = participation_clean is not reliably ready for downstream use.

**4.** Quality of notebook communication and structure

What is being judged: whether the section is well organised and easy to
audit as part of a reproducible cleaning workflow.

- 5 = Markdown and code are very well structured, with clear headings,
  logical flow, and useful reporting of before/after cleaning results.

- 4 = The section is clear and readable overall, with only minor issues
  in flow or explanation.

- 3 = The section is understandable but somewhat uneven in organisation
  or reporting.

- 2 = The section is hard to follow in places because of weak structure
  or unclear explanations.

- 1 = The section is poorly organised or confusing.

**\**

**4 Baseline model training +** **evaluation harness**

**主要任务**

针对logistic regression和xgboost合适的preprocess
pipeline（例如：对数据进行one-hot-encoding等必要的预处理）。

明确X和Y，按0.7/0.15/0.15的比例划分为training/validation/test，考虑target的不平衡性。

确定适合logistic
regression和xgboost的，能反映、比较模型性能的一系列统一衡量指标。

训练基准模型logistic regression

基于之前确立的evaluation harness输出模型训练结果。

**Prompt:**

4 Baseline Model Training + Evaluation Harness (section 4/step 4)

4.1 Prepare modeling data

Using the cleaned dataframe "participation_clean" from the previous
steps, define \`X\` and \`y\`.

Create appropriate preprocessing pipelines for Logistic Regression and
XGBoost respectively, including any necessary preprocessing such as
one-hot encoding.

Split the data into training, validation, and test sets in a 0.7 / 0.15
/ 0.15 ratio. Account for target imbalance when performing the split.

4.2 Create evaluation harness

Define a common set of evaluation metrics and outputs suitable for both
Logistic Regression and XGBoost, so that model performance can be
assessed and compared consistently.

When designing, consider the real-world context of the task and the
characteristics of the data to determine the metrics.

Explain the rules for evaluation harness and the design rationale in the
markdown cell.

4.3 Baseline model: Logistic Regression

Train a baseline Logistic Regression model. Choose reasonable baseline
hyperparameters based on standard practice, without extensive tuning.

Evaluate the model on the validation set only, using the predefined
evaluation harness and present the results clearly.

Organise this section clearly in the notebook using appropriate heading
levels. Follow a readable flow in which markdown cells explain the work
and code cells implement it.

**Success Criteria**

**Objective criteria (basic function)**

1.  X and y are defined correctly from participation_clean

2.  Preprocessing pipelines are created for both Logistic Regression and
    XGBoost

3.  The data is split into training, validation, and test sets in a 0.7
    / 0.15 / 0.15 ratio

4.  The split accounts for target imbalance

5.  A common evaluation harness is defined for both models

6.  The evaluation metrics and outputs are stated clearly

7.  A baseline Logistic Regression model is trained successfully

8.  The baseline Logistic Regression model is evaluated on the
    validation set only

9.  Results are presented clearly using the predefined evaluation
    harness

10. Markdown and code cells are organised clearly with appropriate
    headings

**Quality criteria**

1\. Quality of data split and modelling setup

What is being judged: whether the train/validation/test setup is
methodologically sound for this prediction task.

- 5 = The split is correct, clearly implemented, and methodologically
  sound, including appropriate handling of class imbalance such as
  stratification. The setup supports fair later comparison across
  models.

- 4 = The split is mostly sound and usable, with only minor weakness in
  implementation or explanation.

- 3 = The split is broadly correct but lacks rigour in one important
  aspect, such as weak handling of imbalance or limited explanation.

- 2 = The split is simplistic or weakly implemented, with noticeable
  methodological concerns.

- 1 = The split is incorrect, poorly justified, or unsuitable for
  reliable model comparison.

**2.** Quality of preprocessing design

What is being judged: whether the preprocessing pipelines are
appropriate for Logistic Regression, XGBoost, and the structure of the
data.

- 5 = Preprocessing is well chosen for both models and clearly
  appropriate for the feature types and later modelling steps. It
  supports valid training without obvious unnecessary or unsuitable
  transformations.

- 4 = Preprocessing is mostly appropriate, with only minor weaknesses or
  inefficiencies.

- 3 = Preprocessing is broadly workable but somewhat generic, uneven, or
  only partly suited to the models or data.

- 2 = Preprocessing has clear weaknesses, questionable choices, or
  limited suitability for one or both models.

- 1 = Preprocessing is missing, clearly inappropriate, or likely to
  undermine valid modelling.

**3.** Quality of evaluation harness design

What is being judged: whether the evaluation metrics and outputs are
well matched to the task, the data, and the intended comparison between
models.

- 5 = The evaluation harness is well designed, clearly explained, and
  strongly aligned with the task context and data characteristics.
  Metric choice reflects the class imbalance and the substantive
  importance of identifying the minority class.

- 4 = The evaluation harness is solid and mostly well matched to the
  task, with only minor gaps in justification or metric selection.

- 3 = The evaluation harness is usable but somewhat generic, with
  limited attention to task context or imbalance.

- 2 = The evaluation harness is weakly justified or poorly matched to
  the task.

- 1 = The evaluation harness is missing, inappropriate, or not useful
  for fair comparison.

**4.** Quality of baseline Logistic Regression implementation

What is being judged: whether the baseline model is implemented in a
clean, standard, and methodologically fair way.

- 5 = The baseline Logistic Regression is implemented cleanly and
  appropriately, with reasonable untuned hyperparameters, correct use of
  the validation set, and no sign of hidden tuning or misuse of the test
  set.

- 4 = The baseline implementation is mostly sound, with only minor
  weakness in clarity, parameter choice, or reporting.

- 3 = The baseline model is workable but somewhat generic, weakly
  explained, or methodologically uneven.

- 2 = The baseline implementation shows clear weaknesses, such as
  questionable setup, weak reporting, or limited fairness as a baseline.

- 1 = The baseline implementation is incorrect, poorly controlled, or
  not a valid baseline.

**5.** Quality of notebook communication and structure

What is being judged: whether the section is well organised and easy to
follow as part of a reproducible modelling workflow.

- 5 = Markdown and code are very well structured, with clear headings,
  logical flow, and concise explanations of setup, evaluation logic, and
  results.

- 4 = The section is clear and readable overall, with only minor issues
  in organisation or explanation.

- 3 = The section is understandable but somewhat uneven in flow,
  explanation, or structure.

- 2 = The section is hard to follow in places because of weak
  organisation or unclear markdown.

- 1 = The section is poorly organised or confusing.

**5 Improving performance**

**主要任务**

Logistic regression，在validation set上进行调参，找到最优参数

训练xgboost模型，并在validation set上进行调参，找到最优参数

在test set上使用模型进行预测，比较模型性能。确立选择标准，选择最优模型

**Prompt:**

5 Improving Performance (section 5/step 5)

5.1 Improve Logistic Regression

Use exactly the same training, validation, and test split defined
earlier.

Tune the Logistic Regression model on the validation set only and
identify the best-performing hyperparameter setting.

5.2 Train and tune XGBoost

Use exactly the same training, validation, and test split defined
earlier.

Train an XGBoost model and tune it on the validation set only to
identify the best-performing hyperparameter setting.

5.3 Model comparison

Only at this stage, use the test set for prediction and final
comparison.

Compare the following models:

\- baseline Logistic Regression

\- tuned Logistic Regression

\- XGBoost

Use the evaluation harness defined in the previous section to assess and
compare model performance consistently.

5.4 Final model decision

Taking into account the purpose of the prediction task, define a
systematic, multi-dimensional, and quantitative model selection
framework in a markdown cell.

Clearly explain the selection criteria, then use this framework to
choose the final model.

For both tuned Logistic Regression and tuned XGBoost, print a structured
tuning summary in the notebook. For each model, report:

\- tuning method used

\- hyperparameters searched

\- search range or candidate values for each hyperparameter

\- total number of parameter configurations evaluated

\- iteration or trial count completed

\- best hyperparameter setting found

\- best validation-set performance achieved under the predefined
evaluation harness

Organise this section clearly in the notebook using appropriate heading
levels. Follow a readable flow in which markdown cells explain the work
and code cells implement it.

**Success Criteria**

**Objective criteria (basic function)**

1.  The same training, validation, and test split from the previous
    section is reused

2.  Logistic Regression is tuned using the validation set only

3.  XGBoost is trained and tuned using the validation set only

4.  No test-set information is used during tuning

5.  A structured tuning summary is printed for tuned Logistic Regression

6.  A structured tuning summary is printed for tuned XGBoost

7.  Each tuning summary includes specified information

8.  Test-set prediction is performed only after tuning is complete

9.  Baseline Logistic Regression, tuned Logistic Regression, and XGBoost
    are all evaluated on the test set

10. Model comparison uses the predefined evaluation harness consistently

11. A model selection framework is defined in markdown

12. A final model is selected and justified using the stated framework

13. Markdown and code cells are organised clearly with appropriate
    headings

**Quality criteria**

**1.** Quality of tuning design and efficiency

What is being judged: whether tuning is methodologically sound,
reasonably efficient, and clearly reported.

- 5 = Tuning is well designed, validation-only, and reasonably
  efficient. Search choices are sensible, clearly reported, and not
  unnecessarily wasteful. The structured tuning summary makes the
  process easy to audit and compare.

- 4 = Tuning is mostly sound and reasonably efficient, with only minor
  weakness in search design, reporting, or efficiency.

- 3 = Tuning is broadly workable but somewhat generic, inefficient, or
  only partly well reported.

- 2 = Tuning is weakly designed, poorly reported, or clearly inefficient
  relative to the task.

- 1 = Tuning is incorrect, poorly controlled, or not meaningfully
  interpretable.

**2.** Quality of XGBoost implementation and fairness of comparison

What is being judged: whether XGBoost is trained and compared in a fair
and methodologically consistent way.

- 5 = XGBoost is implemented appropriately and compared fairly using the
  same split and evaluation framework as the other models. The
  comparison is methodologically clean and directly interpretable.

- 4 = XGBoost is mostly implemented and compared fairly, with only minor
  inconsistency or weakness.

- 3 = XGBoost is usable but the comparison is somewhat generic, uneven,
  or weakly controlled.

- 2 = XGBoost implementation or comparison has noticeable fairness or
  consistency issues.

- 1 = XGBoost is implemented incorrectly or compared in a way that is
  not reliable.

**3.** Quality of model comparison and test-set discipline

What is being judged: whether the final comparison is rigorous and
respects the role of the test set.

- 5 = The test set is clearly reserved for final comparison only, and
  the three-model comparison is rigorous, consistent, and easy to
  interpret under the predefined evaluation harness.

- 4 = Test-set use is mostly disciplined and comparison is mostly clear,
  with only minor weakness.

- 3 = The comparison is broadly acceptable, but test-set discipline,
  consistency, or clarity is somewhat limited.

- 2 = The comparison is methodologically weak, unclear, or shows
  questionable handling of the test set.

- 1 = The comparison is unreliable because of misuse of the test set or
  inconsistent evaluation.

**4.** Quality of final model selection framework

What is being judged: whether the final model is chosen using a clear,
task-appropriate, and genuinely multi-dimensional decision framework.

- 5 = The framework is quantitative, uses reasonable scoring or
  weighting, prioritises metrics appropriately for this task, and
  considers more than predictive performance, such as interpretability
  or practical usability.

- 4 = The framework is clear and mostly well designed, with sensible
  criteria and some multi-dimensional thinking, but one element is
  underdeveloped.

- 3 = The framework is usable but fairly generic, with limited
  quantification or limited consideration beyond model performance.

- 2 = The framework is weakly defined, relies mostly on raw performance
  comparison, or shows poor prioritisation of criteria for the task.

- 1 = The framework is missing, unclear, or not meaningfully used to
  choose the final model.

**5.** Quality of notebook communication and structure

What is being judged: whether the section is well organised and easy to
follow as part of a reproducible model-improvement workflow.

- 5 = Markdown and code are very well structured, with clear headings,
  logical flow, and concise explanations of tuning, comparison, and
  final decision.

- 4 = The section is clear and readable overall, with only minor issues
  in organisation or explanation.

- 3 = The section is understandable but somewhat uneven in flow,
  explanation, or structure.

- 2 = The section is hard to follow in places because of weak
  organisation or unclear markdown.

- 1 = The section is poorly organised or confusing.

**\**

**6 Producing reproducible packaging**

这部分不需要在notebook中创建任何内容

在工作目录中创建requirements.txt文件，涵盖所使用的所有package

生成简短README.md，说明需要哪些文件，如何运行，输出包含哪些会保存到哪里，并说明本notebook采用了哪些方法确保reproducibility（例如random
seed的控制）

**Prompt:**

6 Producing Reproducible Packaging

Do not add any new content to the notebook in this step.

In the working directory, create a \`requirements.txt\` file that
includes all Python packages used in the experiment.

Also create a short \`README.md\` file that clearly states:

\- which input files are required

\- how to run the notebook

\- what outputs are produced and where they are saved

\- what steps were taken to support reproducibility, such as fixed
random seed control

Keep the packaging concise, clear, and consistent with the workflow
implemented in the notebook.

**Success Criteria**

**Objective criteria (basic function)**

1.  No new content is added to the notebook in this step

2.  A requirements.txt file is created in the working directory

3.  requirements.txt includes the Python packages used in the experiment

4.  A README.md file is created in the working directory

5.  The README states the required input files

6.  The README explains how to run the notebook

7.  The README describes the main outputs and where they are saved

8.  The README states the main reproducibility measures used

**Quality criteria**

**1.** Quality of requirements.txt

What is being judged: whether the dependency file is complete, correct,
and reasonably minimal.

- 5 = requirements.txt is complete, correct, and reasonably minimal. It
  includes the packages actually used, without obvious important
  omissions or unnecessary clutter.

- 4 = requirements.txt is mostly complete and usable, with only minor
  omission, redundancy, or imprecision.

- 3 = requirements.txt is broadly usable but somewhat incomplete, overly
  broad, or loosely matched to the actual notebook.

- 2 = requirements.txt has noticeable omissions or unnecessary entries
  that reduce reproducibility or clarity.

- 1 = requirements.txt is missing, seriously incomplete, or not usable.

**2.** Quality of README.md for reproducibility

What is being judged: whether the README is clear enough for an
unfamiliar team to rerun the experiment.

- 5 = The README is clear, concise, and practically useful for a new
  team. It explains required files, execution steps, outputs, save
  locations, and reproducibility measures in a way that would
  realistically support rerunning the workflow.

- 4 = The README is mostly clear and usable, with only minor gaps or
  ambiguity.

- 3 = The README is broadly understandable but somewhat incomplete,
  generic, or only partly sufficient for reliable rerunning.

- 2 = The README is vague or incomplete, and would make rerunning
  difficult for an unfamiliar team.

- 1 = The README is missing, very unclear, or not useful for
  reproducibility.

**3.** Quality of consistency with notebook outputs

What is being judged: whether the packaging matches what was actually
done and produced in the notebook.

- 5 = The packaging is fully consistent with the notebook workflow,
  methods, and outputs. File names, output locations, and
  reproducibility notes align clearly with the implemented experiment.

- 4 = The packaging is mostly consistent with the notebook, with only
  minor mismatch or omission.

- 3 = The packaging is broadly related to the notebook, but some details
  are generic, incomplete, or not fully aligned.

- 2 = The packaging shows noticeable mismatch with the notebook workflow
  or outputs.

- 1 = The packaging is seriously inconsistent with the notebook or
  appears largely disconnected from the actual experiment.

**4.** Quality of communication and packaging clarity

What is being judged: whether the packaging is presented in a clean,
professional, and easy-to-audit way.

- 5 = The packaging is concise, well organised, and easy to inspect. It
  communicates the experiment setup clearly without unnecessary detail.

- 4 = The packaging is clear overall, with only minor issues in clarity
  or organisation.

- 3 = The packaging is understandable but somewhat uneven, wordy, or
  lacking polish.

- 2 = The packaging is hard to follow in places because of weak
  organisation or unclear wording.

- 1 = The packaging is confusing, poorly organised, or difficult to use.

**7 Writing documentation**

这部分不需要在notebook中创建任何内容

基于notebook运行结果，创建一个面向政府艺术部门的，non-technical的400字英文数据分析报告

**注意：先运行notebook，确保有output后，再输入Prompt生成报告**

**Prompt:**

7 Writing Documentation

Do not add any new content to the notebook in this step.

Based on the completed notebook results, write a non-technical English
report for a government arts department and save it in the workspace as
\`Report\_\[agent_name\].md\`.

The report should be approximately 400 words and written in markdown
format.

The report should follow a clear high-level structure, but you may
organise the wording and emphasis as you judge appropriate:

\- the purpose of the analysis

\- a brief overview of the data and approach

\- the main findings from the modelling results

\- the final model choice and why it was selected

\- the practical implications for public arts engagement

\- key limitations and cautions

Write for a policy-facing, non-technical audience. Keep the language
clear and accessible. Avoid code, formulas, and unnecessary technical
jargon. Do not overclaim what the model can do, and do not imply causal
conclusions from predictive results alone.

Use the actual results from the notebook rather than generic placeholder
language.

**Success Criteria**

**Objective criteria (basic function)**

1.  No new content is added to the notebook in this step

2.  A file named Report\_\[agent_name\].md is created in the workspace

3.  The report is written in English and markdown format

4.  The report is approximately 400 words

5.  The report is written for a non-technical government arts audience

6.  The report covers specified sections

7.  The report uses the actual notebook results rather than placeholder
    language

**Quality criteria**

**1.** Quality of grounding in notebook results

What is being judged: whether the report accurately reflects the actual
notebook outputs and does not invent unsupported claims.

- 5 = The report is clearly grounded in the notebook results, accurately
  reflects the model comparison and final choice, and does not introduce
  unsupported claims or made-up details.

- 4 = The report is mostly well grounded in the notebook results, with
  only minor imprecision or slight overgeneralisation.

- 3 = The report is broadly consistent with the notebook, but some
  statements are generic, weakly supported, or not tightly tied to the
  actual results.

- 2 = The report only partly reflects the notebook results and includes
  noticeable vagueness, mismatch, or unsupported interpretation.

- 1 = The report is largely ungrounded, misleading, or inconsistent with
  the notebook outputs.

**2.** Quality of communication for a non-technical policy audience

What is being judged: whether the report is genuinely understandable and
appropriate for a government arts audience.

- 5 = The report is clear, accessible, and well pitched for a
  non-technical policy audience. It explains the work without
  unnecessary jargon and keeps the focus on what matters for
  decision-making.

- 4 = The report is mostly clear and suitable for the intended audience,
  with only minor issues in tone or accessibility.

- 3 = The report is understandable overall, but somewhat generic,
  uneven, or too technical in places.

- 2 = The report is only partly suitable for a non-technical audience
  and contains substantial jargon, weak framing, or unclear explanation.

- 1 = The report is poorly matched to the audience and difficult for
  non-technical readers to use.

**3.** Quality of rationale and practical interpretation

What is being judged: whether the report explains the model choice and
practical meaning of the results in a sensible and convincing way.

- 5 = The report gives a clear and credible rationale for the final
  model choice, links it to the purpose of the task, and draws practical
  implications that are relevant and well reasoned.

- 4 = The rationale and interpretation are mostly clear and sensible,
  with only minor weakness in depth or connection to the task.

- 3 = The rationale is broadly acceptable, but somewhat generic, thin,
  or only partly convincing.

- 2 = The rationale is weak, unclear, or only loosely connected to the
  task and results.

- 1 = The rationale is missing, implausible, or not meaningfully
  supported.

**4.** Quality of caution and responsible reporting

What is being judged: whether the report communicates limitations
appropriately and avoids overclaiming.

- 5 = The report communicates limitations clearly, avoids causal
  overclaiming, and presents the model as decision support rather than
  definitive truth.

- 4 = The report is mostly responsible and appropriately cautious, with
  only minor overstatement or incomplete caveats.

- 3 = The report includes some caution, but limitations are
  underdeveloped or the tone is somewhat overconfident.

- 2 = The report gives weak or minimal caution and risks overstating
  what the model can support.

- 1 = The report is clearly overclaimed, misleading, or lacks meaningful
  limitations.
