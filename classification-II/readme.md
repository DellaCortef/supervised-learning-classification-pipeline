## Project Summary: Classification Models and Data Pipeline

This notebook demonstrates end-to-end workflows for preparing datasets, building classification models, tuning hyperparameters, and evaluating performance using several machine learning techniques. Below is an outline of the key steps and methodologies applied.

### Key Steps

1. Data Preparation
Data Source: 
- Olist dataset (propensao_revenda_abt.csv) is used to build the classification models.

Training and Evaluation Split:
- df_train: Data before "2018-03-01" for training.
- df_oot: Data from "2018-03-01" for out-of-time testing.

Feature Engineering:
- key_vars: Identifiers such as data_ref_safra and seller_id.
- num_vars: Numerical features (e.g., tot_orders_12m, recencia).
- cat_vars: Categorical features (e.g., uf).
- target: Binary classification target nao_revendeu_next_6m.

2. Data Imputation and Encoding
Handling Missing Values:
- Numerical features: Imputed using ArbitraryNumberImputer.
- Categorical features: Imputed using CategoricalImputer.
- Encoding Categorical Variables:
- OneHotEncoder and OrdinalEncoder are used for categorical feature transformation.

3. Model Training
Models tested include:
- Logistic Regression
- Decision Tree Classifier
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Models are implemented within pipelines to streamline preprocessing and training.

4. Hyperparameter Tuning
GridSearchCV: 
- Exhaustive search over a parameter grid for models like Decision Trees and Random Forests.

RandomizedSearchCV: 
- Randomized search for faster tuning with models like Random Forests.

5. Model Evaluation
Metrics used:
- ROC-AUC: Primary metric to evaluate classifier performance.
- Recall: To assess sensitivity in predicting positive outcomes.
- Accuracy, Precision, and F1-Score for detailed comparisons.
- Evaluation conducted on both:
- Training dataset.
- Out-of-time (OOT) testing dataset.

6. Model Explainability
SHAP (SHapley Additive exPlanations):
- Provides insights into feature importance and interactions.
- Visualizes the impact of features like recencia and receita_12m on predictions.

7. Data Drift Detection
Evidently:
- Generates a dashboard to detect and analyze data drift between reference and production datasets.
- Reports saved as relatorio.html for further review.

8. Model Deployment and Predictions
Predictions are made on new production datasets, and outputs include:
- Predicted class (classe_prevista).
- Prediction scores (score_1 and score_0).