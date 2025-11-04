#Config file for defining lists of variables/features for use in the model pipeline

clean_vars = ['age', 
            'class of worker', 
            'education',
            'marital stat',
            'major occupation code', 
            'race', 
            'sex',
            'full or part time employment stat', 
            'capital gains', 
            'instance weight', 
            'citizenship',
            'weeks worked in year', 
            'salary'
            ]

model_v0_features = [
    'weeks worked in year',
    'age_mapped',
    'log_capital_gains',
    'class_of_worker_mapped',
    'occupation_mapped',
    'education_mapped',
    'marital_status_mapped',
    'citizenship_mapped',
    'part_full_time_mapped',
    'race',
    'sex'
]

model_v1_features = model_v0_features

model_v2_features = [
   'weeks worked in year',
   'age',
   'log_capital_gains',
   'class of worker',
   'major occupation code',
   'education',
   'marital stat',
   'citizenship',
   'full or part time employment stat',
   'race',
   'sex'
]

categorical_features = [
    'age_mapped',
    'class_of_worker_mapped',
    'occupation_mapped',
    'education_mapped',
    'marital_status_mapped',
    'citizenship_mapped',
    'part_full_time_mapped',
    'class of worker',
    'major occupation code',
    'education',
    'marital stat',
    'full or part time employment stat',
    'citizenship',
    'race',
    'sex'
]

numeric_features = [
    'weeks worked in year',
    'age',
    'log_capital_gains'
]