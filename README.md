# Supervised Learning Classification Reference

Welcome to this repository! Here, you'll find everything you need to build supervised learning classification pipelines like a pro. This project has all the fun stuff: data preprocessing, training, hyperparameter tuning, evaluating models, and even detecting data drift (yep, we’ve got you covered!).

## What’s Inside?

Here’s what you’ll find in this project:

### 📂 Key Files:
- Jupyter Notebooks:
    - machine-learning-classification-pipeline.ipynb: Step-by-step guide for building ML pipelines.
    - comparing-the-algorithms.ipynb: Pits different classifiers against each other. Who wins? Check it out!
    - feature-selection.ipynb: Learn how to pick the best features like a true data scientist.
    - Grid-Main-Hyperparameters-Classification.ipynb: Tune your model to perfection with grid search.
    - hyperparameter-optimization.ipynb: Random search for when you’re feeling lucky.
    - predicting-new-data.ipynb: Take your trained model for a spin and predict on fresh data.

- Datasets:
    - Real-world data like olist_orders_dataset.csv and olist_order_items_dataset.csv to work with.
    - Preprocessed data like propensao_revenda_abt.csv to jump straight into modeling.

- Saved Models:
    - Pretrained models like best_model.pkl and pipeline_baseline_rl.joblib. Ready to go!

- Drift Report:
    - Keep an eye on your data with drift_report.html. Powered by Evidently.


## How to Get Started

### Clone the Repo:
```bash
git clone https://github.com/DellaCortef/supervised-learning-classification-reference.git
cd supervised-learning-classification-reference
```

### Install the Requirements:
- Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
```

- Install the dependencies:

```bash
pip install -r requirements.txt
```

- Run the Notebooks:
    - Fire up Jupyter:

```bash
jupyter notebook
```


## What Can You Do Here?

1. Preprocess Data Like a Pro:
- Fill in missing values, scale your features, encode categorical variables — all in one pipeline!
2. Train Models:
- Try out models like Logistic Regression, Random Forest, XGBoost, and CatBoost.
- Use cross-validation to check performance.
3. Tune Hyperparameters:
- Use GridSearchCV or RandomizedSearchCV to squeeze the most out of your models.
4. Detect Data Drift:
- Keep your models honest by tracking changes in your data over time.