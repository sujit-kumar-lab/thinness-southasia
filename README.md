# MERF-Based Decision Support System for Child and Adolescent Thinness

An interactive machine learning-based decision support system for predicting and projecting child and adolescent thinness across South Asia.

The system uses a Mixed Effects Random Forest (MERF) model to account for nonlinear relationships and country-level heterogeneity in longitudinal data. Users can select a country and projection year and explore customized what-if scenarios by modifying key socioeconomic and demographic determinants.

## Key Features

- Country-specific prediction of child and adolescent thinness
- Scenario-based projection through 2030
- User-defined socioeconomic and demographic scenarios
- Historical and projected thinness trends
- Relative importance of model predictors
- Approximate 95% prediction interval
- MERF-based prediction accounting for country-level heterogeneity

## Final Predictors

The decision support system uses four final predictors:

1. Out-of-pocket health expenditure
2. Urban population
3. Fertility rate
4. Unemployment

## Repository Structure

```text
app-thinness/
│
├── app.py
├── data/
│   └── panel_data_imputed.xlsx
│
├── model/
│   ├── model_merf.pkl
│   └── feature_names.pkl
│
├── requirements.txt
├── README.md
└── LICENSE
