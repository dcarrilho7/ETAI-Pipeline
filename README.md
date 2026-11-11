Student: David Carrilho - 20231693

# Results

Week 2
- Logistic Regression: 67.9% training accuracy, 67.7% test accuracy.
- Decision Tree: 82.9% training accuracy, 63.1% test accuracy.

Logistic Regression performed better on the test data. The Decision Tree had a much larger gap between training and test scores due to overfit.

Week 3 

 After applying the EDA-based data cleaning, Logistic Regression achieved 67.8% training accuracy and 65.5% test accuracy, while the Decision Tree achieved 79.9% training accuracy and 60.9% test accuracy. Both models had lower test accuracy than in Week 2. Logistic Regression still generalised better, while the Decision Tree continued to show considerable overfitting.

Week 4

I moved the preprocessing and model into the same scikit-learn pipeline. The data is now split into a development set and a locked test set. I use stratified 5-fold cross-validation on the development set, and the imputer, encoder, and scaler are fitted inside each fold.

Added a Dummy Classifier and Random Forest, and tested target encoding, KNN imputation, and a depth-limited Decision Tree.

| Model / setup | CV accuracy | Locked test accuracy | Mean train-validation gap |
|---|---:|---:|---:|
| Dummy baseline | 54.9% | 55.0% | 0.0 percentage points |
| Random Forest | 64.6% | 64.7% | 9.0 percentage points |
| Decision Tree (`max_depth=5`) | 66.4% | 66.0% | 1.9 percentage points |
| Logistic Regression (target encoding, median imputation, robust scaling) | 67.2% | 65.8% | 0.3 percentage points |

The depth-limited Decision Tree overfit less than the unrestricted tree. Logistic Regression had the best cross-validation accuracy and the smallest train-validation gap, so it is currently the best model in this experiment. The KNN imputer and target encoder were tested as alternatives, but neither produced a clear improvement over the current setup.

Week 5

This week I added Optuna to tune the models. I tested `max_depth`, `min_samples_leaf`, and `criterion` for the Decision Tree, `C` for Logistic Regression, and several parameters for Random Forest. I used nested cross-validation so the models were evaluated on folds that were not used during tuning.

| Model | Default CV mean ± std | Tuned nested CV mean ± std | Best trial | Optimism | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 61.1% ± 1.5% | 67.5% ± 2.0% | 68.0% | 0.0 pp | `max_depth=9`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic Regression | 67.2% ± 1.3% | 67.3% ± 1.6% | 67.8% | +0.2 pp | `C=0.072` |
| Random Forest | 64.6% ± 1.6% | 68.2% ± 1.5% | 68.3% | -0.1 pp | `n_estimators=137`, `max_depth=None`, `min_samples_leaf=33`, `max_features=log2` |

Random Forest performed best after tuning, while Logistic Regression changed very little. The Decision Tree also improved compared with its default settings. The test set was kept separate from the tuning process. Random Forest used 15 trials; the other two models used 30.

Week 6

This week I added Random Forest, Oblique Random Forest, and AdaBoost to the model registry. They use the same pipeline and nested-CV tuning process as the earlier models. Random Forest and Oblique Random Forest use many trees, while AdaBoost builds shallow trees one after another.

| Model | Default CV mean ± std (gap) | Tuned nested CV mean ± std (gap) | Best trial | Chosen hyperparameters |
|---|---:|---:|---:|---|
| Logistic Regression | 67.4% ± 1.1% (+0.001) | 67.5% ± 1.1% (+0.002) | 67.6% | `C=0.023` |
| Decision Tree | 59.1% ± 0.8% (+0.116) | 66.8% ± 1.3% (+0.014) | 67.3% | `max_depth=5`, `min_samples_leaf=2`, `criterion=entropy` |
| Random Forest | 64.1% ± 1.6% (+0.112) | 67.2% ± 0.8% (+0.018) | 67.8% | `max_depth=24`, `min_samples_leaf=16`, `max_features=sqrt` |
| Oblique Random Forest | 64.2% ± 1.4% (+0.137) | 67.5% ± 1.2% (+0.011) | 68.0% | tree `max_depth=3`, tree `min_samples_leaf=88`, `n_components=8` |
| AdaBoost | 67.4% ± 0.7% (+0.005) | 67.6% ± 0.9% (+0.007) | 67.9% | base depth `4`, `n_estimators=362`, `learning_rate=0.304` |

AdaBoost had the highest nested-CV mean, but its lead over Oblique Random Forest and Logistic Regression is smaller than the fold-to-fold variation. The tuned forests reduced the overfitting gap compared with their default settings. I would report AdaBoost's 67.6% nested-CV score, while treating the three best ensembles as very close.

I also added three fixed engineered features: priors per adult year, an age-priors interaction, and total juvenile offenses. In a small five-trial nested-CV comparison, Logistic Regression improved from 67.3% to 67.5%, while Random Forest changed from 67.9% to 67.6% and AdaBoost from 68.0% to 67.9%. The Logistic Regression gain was smaller than the fold-to-fold standard deviation, so the features did not give a clear improvement.







Week 7

This week I added Gradient Boosting, HistGradientBoosting and Naive Bayes to the model list. I also changed the evaluation so that one run records accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC and average precision. I used ROC-AUC as the selection metric because it uses the predicted probabilities instead of depending on only the 0.5 cutoff.

The runs below used five Optuna trials so that I could compare the models without making the experiments too long. The test set was still kept locked. The same rows are also saved in `results/runs.csv`.

| Model | Default CV ROC-AUC | Tuned nested CV ROC-AUC |
|---|---:|---:|
| adaboost | 72.2% | 72.7% +/- 1.2% |
| random_forest | 68.1% | 72.6% +/- 1.4% |
| gradient_boosting | 72.5% | 72.6% +/- 1.2% |
| oblique_random_forest | 68.0% | 72.5% +/- 1.1% |
| hist_gradient_boosting | 71.1% | 72.5% +/- 1.2% |
| logistic_regression | 72.2% | 72.1% +/- 1.2% |
| decision_tree | 58.4% | 71.6% +/- 1.0% |
| naive_bayes | 70.0% | 71.0% +/- 1.1% |
| dummy | 50.0% | 50.0% (no tuning) |

AdaBoost had the highest score in this small comparison. Random Forest and Gradient Boosting were very close. The Dummy Classifier is the floor at 50.0%, so all of the real models were above it. These results are only a first comparison because I used five tuning trials.


Week 8

This week I added XGBoost and a feature-selection step to the pipeline. The selector uses a random forest, gradient boosting and L1 logistic regression to vote on which features to keep, and it runs inside each CV fold so it does not see the validation rows. I compared tuned XGBoost with and without selection using the same five folds.

| XGBoost setup | Tuned nested CV ROC-AUC |
|---|---:|
| With VoteSelector | 72.6% +/- 1.1% |
| Without selection | 72.7% +/- 1.2% |

The corrected paired test gave a p-value of 0.63, so the small difference is not meaningful in this experiment. The selected version kept 4 of 13 features, but I would keep selection off for now because it did not improve the score; it could still be useful if the goal is a smaller model.


Week 9

I chose Option A and applied the pipeline to the Adult income dataset. I added a separate config file, made the data cleaning work with the new columns, and saved predictions for the new rows in `adult_predictions.csv`.

# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model and pipeline construction
│   ├── evaluate.py         # cross-validation, metrics + fairness check
│   ├── tuning.py           # Optuna tuning + nested cross-validation
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 5 | Hyperparameter tuning | Added a single `build_pipeline()` function, Optuna tuning for a decision tree, and nested cross-validation so the tuning score is estimated on folds that were not used to choose the parameters. |
| 6 | Ensembles and feature engineering | Added Random Forest, Oblique Random Forest, AdaBoost, and three fixed engineered features. All models use the same config-driven pipeline and nested-CV tuning. |
| 7 | More models and metrics | Added Gradient Boosting, HistGradientBoosting and Naive Bayes, and saved a scoreboard with several CV metrics. |
| 8 | XGBoost and feature selection | Added XGBoost and a VoteSelector inside the pipeline, then compared tuned XGBoost with and without selection. |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- default and nested cross-validation results
- a classification report from out-of-fold development-set predictions
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score
- the selected hyperparameters and the locked test-set status

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
