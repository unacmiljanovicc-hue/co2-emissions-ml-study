# CO₂ Emissions Prediction

This project started as my graduation research in Information Systems and Technologies. I wanted to see how different ways of handling missing data affect CO₂ emissions predictions.

## What I did

I worked with country-level CO₂ emissions data from the Global Carbon Budget, together with socioeconomic and energy-related data from Our World in Data.

I tested four different approaches for handling missing values:

- **V1:** removed columns with too many missing values and then removed the remaining incomplete rows
- **V2:** filled missing values using the median for each decade and continent
- **V3:** used iterative imputation with `RandomForestRegressor` (missForest-style)
- **V4:** used iterative imputation with `BayesianRidge` (MICE-style), with `Year` included as an additional predictor

I then tested each version of the dataset with four models:

- Linear Regression
- MLP neural network
- Random Forest
- Transformer-style model built in PyTorch

For evaluation, I used MAE, RMSE and R². Since this is time-based data, I used rolling-window validation instead of a random train/test split.

The setup was:
- 20 years for training
- 5 years for testing
- 5-year step
- 8 folds in total

## Target leakage

One of my first results looked suspiciously good, with R² very close to 1.

After checking the features, I found that some columns were directly related to the target variable `Total`. These included individual fuel/process components and ratios calculated from the target.

I removed these columns and reran the experiments.

This ended up being a useful part of the project because it showed how easy it is to get misleadingly good results when information related to the target gets into the feature set.

## Results

In the saved rolling-window experiments, the MICE-style approach had the highest mean R² for all four models.

| Model | MAE | RMSE | Mean R² |
|---|---:|---:|---:|
| Linear Regression | 23.6009 | 64.3322 | 0.9885 |
| MLP | 21.4794 | 49.5968 | 0.9918 |
| Random Forest | 16.0262 | 82.6863 | 0.9790 |
| Transformer | 27.3331 | 102.7118 | 0.9669 |

The full results, including results for individual folds, are in `results/`.

## Limitation

There is one limitation in the way I handled V3 and V4.

For V1 and V2, missing values can be handled using only the training data for each fold. For V3 and V4, I ran iterative imputation on the full dataset before rolling-window evaluation because fitting the imputers separately for every fold was computationally expensive.

This means that some information from later years could influence the imputation of earlier data.

If I continued the project, I would change this by fitting the imputer separately on each training fold and then using it to transform only that fold's test data.


Open `notebooks/co2_emissions_ml_study.ipynb` and run the cells in order. The full Transformer rolling-window experiment is intentionally disabled by default because it is computationally intensive; set `RUN_FULL_EXPERIMENT = True` in the notebook to rerun it.

## Tech stack

Python · pandas · NumPy · scikit-learn · PyTorch · Matplotlib · SciPy
