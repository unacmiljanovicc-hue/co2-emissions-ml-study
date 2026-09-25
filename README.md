# CO₂ Emissions Prediction - Missing-Value Strategies & Model Evaluation

Machine-learning study of how missing-value treatment affects annual CO₂-emissions prediction on country-year data. This project originated as my graduation research in Information Systems and Technologies.

## What I explored

The workflow combines Global Carbon Budget emissions data with socioeconomic and energy-related variables from Our World in Data. I compared four missing-value strategies and four predictive models, then evaluated them with time-based rolling-window validation.

**Imputation strategies**
- V1 deletion of highly incomplete columns and remaining incomplete rows
- V2 grouped median imputation, by decade and continent
- V3 missForest-style iterative imputation (`IterativeImputer` + `RandomForestRegressor`)
- V4 MICE-style iterative imputation (`IterativeImputer` + `BayesianRidge`) with `Year` added as a predictor

**Models**
- Linear Regression
- MLP neural network 
- Random Forest
- Custom Transformer-style model implemented in PyTorch

**Evaluation**
- MAE
- RMSE
- R²
- 8-fold rolling-window validation: 20-year training window, 5-year test window, 5-year step

## A useful mistake: detecting target leakage

An early experiment produced R² values close to 1.0. Investigating the result showed that several input columns directly define or derive from the target `Total` (including fuel/process components and target-derived ratios). Those variables were removed from the predictive feature set before the final comparison.

This was an important part of the project because it changed the experimental design rather than simply accepting an unrealistically strong score.

## Results

The saved rolling-window experiment showed the MICE-style variant with the highest mean R² within each of the four model families:

| Model | MAE | RMSE | Mean R² |
|---|---:|---:|---:|
| Linear Regression | 23.6009 | 64.3322 | 0.9885 |
| MLP | 21.4794 | 49.5968 | 0.9918 |
| Random Forest | 16.0262 | 82.6863 | 0.9790 |
| Transformer | 27.3331 | 102.7118 | 0.9669 |


Full results and fold-level results are available in `results/`.

## Methodological limitation

V1 and V2 can apply imputation using training-fold information only. In the original experiment, V3 and V4 were iteratively imputed on the full dataset before the rolling-window model evaluation because repeatedly fitting those imputers for every fold was computationally expensive. This can introduce temporal leakage during imputation.

A natural next step would be to refit each imputer independently inside every training fold and transform the corresponding test fold without using future information.


Open `notebooks/co2_emissions_ml_study.ipynb` and run the cells in order. The full Transformer rolling-window experiment is intentionally disabled by default because it is computationally intensive; set `RUN_FULL_EXPERIMENT = True` in the notebook to rerun it.

## Tech stack

Python · pandas · NumPy · scikit-learn · PyTorch · Matplotlib · SciPy
