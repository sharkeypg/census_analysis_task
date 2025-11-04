import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from catboost import CatBoostClassifier

def make_coarse_age_feature(col: pd.Series) -> pd.Series:
    """
    Group continuous age variable into discrete bins
    """

    age_bins = [0,20,40,60,80,100]
    age_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
    return pd.cut(col, bins = age_bins, labels = age_labels, right = False)

def make_coarse_occupation_feature(col: pd.Series) -> pd.Series:
    """
    Map occupation codes to coarse categories
    """
    
    occupation_map = {
        ' Not in universe': 'Not in universe',
        ' Precision production craft & repair': 'Blue-collar',
        ' Professional specialty': 'Professional',
        ' Executive admin and managerial': 'Professional',
        ' Handlers equip cleaners etc ': 'Blue-collar',
        ' Adm support including clerical': 'Clerical',
        ' Machine operators assmblrs & inspctrs': 'Blue-collar',
        ' Other service': 'Clerical',
        ' Sales' : 'Clerical',
        ' Private household services': 'Clerical',
        ' Technicians and related support': 'Professional',
        ' Transportation and material moving': 'Blue-collar',
        ' Farming forestry and fishing': 'Blue-collar',
        ' Protective services': 'Blue-collar',
        ' Armed Forces': 'Armed Forces' 
    }
    return col.map(occupation_map)

def log_transform_cap_gains(col: pd.Series) -> pd.Series:
    return np.log(col + 1)

def make_coarse_education_feature(col: pd.Series) -> pd.Series:
    """
    Map education groups to coarse categories
    """
    
    education_map = {
        ' 1st 2nd 3rd or 4th grade': 'Primary school',
        ' Less than 1st grade': 'Primary school',
        ' 5th or 6th grade': 'Primary school',
        ' 7th and 8th grade': 'Primary school',
        ' 9th grade': 'Primary school',
        ' 10th grade': 'Primary school',
        ' 11th grade': 'Primary school',
        ' 12th grade no diploma': 'Primary school',
        ' High school graduate': 'High school graduate',
        ' Some college but no degree': 'High school graduate',
        ' Associates degree-academic program': 'Associate degree',
        ' Associates degree-occup /vocational': 'Associate degree',
        ' Masters degree(MA MS MEng MEd MSW MBA)': 'Postgrad/Prof degree',
        ' Prof school degree (MD DDS DVM LLB JD)': 'Postgrad/Prof degree',
        ' Doctorate degree(PhD EdD)': 'Postgrad/Prof degree',
        ' Children': 'Children',
        ' Bachelors degree(BA AB BS)': 'Bachelors degree'
    }
    return col.map(education_map)

def make_coarse_class_of_worker_feature(col: pd.Series) -> pd.Series:
    """
    Map worker class groups to coarse categories
    """
    
    class_of_worker_map = {
        ' Not in universe': 'Not in universe',
        ' Self-employed-not incorporated': 'Self-employed',
        ' Private': 'Private',
        ' Local government': 'Government',
        ' Federal government': 'Government',
        ' Self-employed-incorporated':'Self-employed',
        ' State government': 'Government',
        ' Without pay': 'No pay',
        ' Never worked': 'No pay'
    }
    return col.map(class_of_worker_map)

def make_coarse_occupation_feature(col: pd.Series) -> pd.Series:
    """
    Map occupation codes to coarse categories
    """

    occupation_map = {
        ' Not in universe': 'Not in universe',
        ' Precision production craft & repair': 'Blue-collar',
        ' Professional specialty': 'Professional',
        ' Executive admin and managerial': 'Professional',
        ' Handlers equip cleaners etc ': 'Blue-collar',
        ' Adm support including clerical': 'Clerical',
        ' Machine operators assmblrs & inspctrs': 'Blue-collar',
        ' Other service': 'Clerical',
        ' Sales' : 'Clerical',
        ' Private household services': 'Clerical',
        ' Technicians and related support': 'Professional',
        ' Transportation and material moving': 'Blue-collar',
        ' Farming forestry and fishing': 'Blue-collar',
        ' Protective services': 'Blue-collar',
        ' Armed Forces': 'Armed Forces' 
    }
    return col.map(occupation_map)

def make_coarse_marital_feature(col: pd.Series) -> pd.Series:
    """
    Map marital status groups to coarse categories
    """
    
    marital_status_map = {
        ' Widowed': 'Was married',
        ' Divorced': 'Was married',
        ' Never married': 'Never married',
        ' Married-civilian spouse present': 'Married, spouse present',
        ' Separated': 'Married, spouse not present',
        ' Married-spouse absent': 'Married, spouse not present',
        ' Married-A F spouse present': 'Married, spouse present'
    }
    return col.map(marital_status_map)

def make_coarse_part_full_time_feature(col: pd.Series) -> pd.Series:
    """
    Map full/part time groups to coarse categories
    """

    part_full_time_map = {
        ' Not in labor force': 'Not in labor force',
        ' Children or Armed Forces': 'Children or Armed Forces',
        ' Full-time schedules': 'Full-time',
        ' Unemployed full-time': 'Unemployed',
        ' Unemployed part- time': 'Unemployed',
        ' PT for non-econ reasons usually FT': 'Part-time',
        ' PT for econ reasons usually PT': 'Part-time',
        ' PT for econ reasons usually FT': 'Part-time'
    }  
    return col.map(part_full_time_map)

def make_coarse_citizenship_feature(col: pd.Series) -> pd.Series:
    """
    Map citizenship groups to coarse categories
    """

    citizenship_map = {
        ' Native- Born in the United States': 'Citizen',
        ' Foreign born- Not a citizen of U S ': 'Not citizen',
        ' Foreign born- U S citizen by naturalization': 'Citizen',
        ' Native- Born abroad of American Parent(s)': 'Citizen',
        ' Native- Born in Puerto Rico or U S Outlying': 'Citizen'
    }
    return col.map(citizenship_map)

def clean_df(df: pd.DataFrame, clean_vars: List[str]) -> pd.DataFrame:
    """
    Data cleaning step: drop duplicate rows and select only the columns we look to use in feature engineering and analysis
    """

    df = df.drop_duplicates()
    df = df[clean_vars]

    return df


def engineer_features(df: pd.DataFrame, select_feature_vars: List[str], target_var: str, weight_var: str) -> pd.DataFrame:
    """
    Engineer features based on defining coarser categories and include target variable and instance weight
    """

    df['age_mapped'] = make_coarse_age_feature(df['age'])
    df['log_capital_gains'] = log_transform_cap_gains(df['capital gains'])
    df['education_mapped'] = make_coarse_education_feature(df['education'])
    df['occupation_mapped'] = make_coarse_occupation_feature(df['major occupation code'])
    df['marital_status_mapped'] = make_coarse_marital_feature(df['marital stat'])
    df['class_of_worker_mapped'] = make_coarse_class_of_worker_feature(df['class of worker'])
    df['citizenship_mapped'] = make_coarse_citizenship_feature(df['citizenship'])
    df['part_full_time_mapped'] = make_coarse_part_full_time_feature(df['full or part time employment stat'])
    df = df[select_feature_vars + [target_var] + [weight_var]]
    return df

def train_and_test_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, target_var: str, weight_var: str) -> Tuple[pd.DataFrame]:
    """
    Create sklearn X and y inputs for training and test datasets
    """

    X_train =  train_df.drop([target_var, weight_var], axis=1)
    y_train = train_df[target_var]

    X_test = test_df.drop([target_var, weight_var], axis=1)
    y_test = test_df[target_var]

    return X_train, y_train, X_test, y_test

def define_preprocessor(numeric_features: List[str], categorical_features: List[str], model_features: List[str]) -> ColumnTransformer:
    """
    Define sklearn preprocessor with the following steps:
    1. Scale numeric quantities by subtracting mean and dividing by standard deviation
    2. One-hot encode categorical quantities
    """

    v0_numeric_features = [x for x in numeric_features if x in model_features]
    v0_categorical_features = [x for x in categorical_features if x in model_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), v0_numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), v0_categorical_features),
        ],
        remainder='passthrough'
    )

    return preprocessor

def train_logistic_regression_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series) -> Pipeline:
    """
    Fit logistic regression model to training data. Use class_weight quantity to handle imbalanced classes. To ensure inference over population,
    weight each observation by instance weight.
    """

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced'))
    ]
    )

    pipeline.fit(X_train, y_train, classifier__sample_weight = sample_weight)
    return pipeline

def train_catboost_model(preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: pd.Series) -> Pipeline:
    """
    Fit CatBoost model to training data. Weight each class by inverse class frequencies to impose balance. To ensure inference over population,
    weight each observation by instance weight.
    """
    
    model = CatBoostClassifier(
        learning_rate=0.1,
        depth=6,
        class_weights = {' - 50000.': 6, ' 50000+.' : 94},
        verbose=0
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train, classifier__sample_weight = sample_weight)
    return pipeline

def get_test_set_predictions(pipeline: Pipeline, X_test: pd.DataFrame) -> Tuple:
    """
    Get test set predicted salary bands and associated probabilities
    """

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def get_evaluation_metrics(y_test: pd.Series, y_pred: np.array, weight: pd.Series) -> Dict:

    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred, sample_weight = weight),
        'precision': precision_score(y_test, y_pred, pos_label= ' 50000+.', sample_weight = weight),
        'recall': recall_score(y_test, y_pred, pos_label= ' 50000+.', sample_weight = weight),
        'f1': f1_score(y_test, y_pred, pos_label= ' 50000+.', sample_weight = weight)
    }
    return metrics

def plot_roc_curve(y_test, y_proba, weight):

    auc = roc_auc_score(y_test, y_proba, sample_weight=weight)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label= ' 50000+.', sample_weight = weight)

    plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int) -> None:

    importance_df['rel_importance'] = importance_df['importance']/importance_df['importance'].sum()
    top_features = importance_df.sort_values('importance', ascending = False).head(top_n)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features['feature'][::-1], 
        top_features['rel_importance'][::-1], 
        color='#1f77b4' # A standard Matplotlib blue color
    )
    plt.title("Top 10 importance features for model v2", fontsize=16, pad=20)
    plt.xlabel('Feature Importance Score (Lower = Less Important)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def feature_importance_logreg(model: Pipeline) -> pd.DataFrame:
    """
    Get measure of importance of each feature in the model. For logistic regression, the model
    coefficients are a measure of each feature's individual contribution to a prediction
    """

    preprocessor_logit = model.named_steps['preprocessor']
    model_logit = model.named_steps['classifier']

    logit_feature_names = preprocessor_logit.get_feature_names_out()
    logit_coefs = model_logit.coef_.flatten()

    logit_importance_df = pd.DataFrame({
        'feature': logit_feature_names,
        'importance': np.abs(logit_coefs)
    })
    return logit_importance_df

def feature_importance_catboost(model: Pipeline) -> pd.DataFrame:
    """
    Get measure of importance of each feature in the CatBoost model
    """
    
    preprocessor_cat = model.named_steps['preprocessor']
    model_cat = model.named_steps['classifier']

    cat_feature_names = preprocessor_cat.get_feature_names_out()
    cat_importance = model_cat.get_feature_importance()

    cat_importance_df = pd.DataFrame({
        'feature': cat_feature_names,
        'importance': cat_importance
    })
    return cat_importance_df

def model_pipeline(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame, 
                   clean_vars: List[str], 
                   select_feature_vars: List[str], 
                   target_var: str,
                   weight_var: str, 
                   model_architecture: str,
                   numeric_features: List[str], 
                   categorical_features: List[str]) -> Tuple[Pipeline, Dict]:
    
    """
    End to end data cleaning, transformation, training and evaluation pipeline
    """

    #Step 1: Clean data
    clean_df_train = clean_df(train_df, clean_vars)
    clean_df_test = clean_df(test_df, clean_vars)
    print("Data cleaning complete")

    #Step 2: Engineer features
    feature_df_train = engineer_features(clean_df_train, select_feature_vars, target_var, weight_var)
    feature_df_test = engineer_features(clean_df_test, select_feature_vars, target_var, weight_var)
    print("Feature engineering complete")

    #Step 3: Train model
    X_train, y_train, X_test, y_test = train_and_test_xy(feature_df_train, feature_df_test, target_var, weight_var)

    preprocessor = define_preprocessor(numeric_features, categorical_features, select_feature_vars)

    if model_architecture == 'log_reg':
        model = train_logistic_regression_model(preprocessor, X_train, y_train, feature_df_train[weight_var])
        importance_df = feature_importance_logreg(model)
    elif model_architecture == 'catboost':
        model = train_catboost_model(preprocessor, X_train, y_train, feature_df_train[weight_var])
        importance_df = feature_importance_catboost(model)
        
    print("Model training complete")

    #Step 4: Model evaluation
    y_pred, y_proba = get_test_set_predictions(model, X_test)

    eval_metrics = get_evaluation_metrics(y_test, y_pred, feature_df_test[weight_var])

    plot_roc_curve(y_test, y_proba, feature_df_test[weight_var])
    print("Model evaluation complete")

    return model, eval_metrics, importance_df



