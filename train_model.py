### Import Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import Data For training
url = 'https://raw.githubusercontent.com/MattDBailey/ANOP330/refs/heads/main/Data/BucknellLendingClubHistoricalData.csv'
past_df = pd.read_csv(url)

### Prep data
# Remove unneeded columns
drop_cols = ['id',            # not useful
             'issue_d',       # not useful
             'last_pymnt_d',  # Post Outcome
             'loan_length',   # Post Outcome
             'recoveries',    # Post Outcome
             'total_pymnt',   # Post Outcome
             'installment',   # mltco - loan_amnt, int_rate, term_num
             'funded_amnt',   # mltco - loan_amnt
             'term',          # mltco - term_nums
             'grade'          # mltco
            ]
df = past_df.drop(columns = drop_cols, errors = 'ignore')

# Log transform income
df['annual_inc'] = np.log1p(df['annual_inc'])

# Handle outliers/missing
df['dti'] = df['dti'].replace(999, np.nan)
df['dti'] = df['dti'].fillna(df['dti'].median())

df['revol_util'] = df['revol_util'].clip(upper=150)
df['revol_bal'] = df['revol_bal'].clip(upper=df['revol_bal'].quantile(0.99))

# Convert earliest credit line → credit age
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['credit_age'] = 2020 - df['earliest_cr_line'].dt.year
df = df.drop(columns=['earliest_cr_line'])

# FICO average
df['fico_avg'] = (df['fico_range_high'] + df['fico_range_low']) / 2
df = df.drop(columns=['fico_range_high', 'fico_range_low'])

# Encode Numerical information
categorical_cols = ['home_ownership', 
                    'verification_status', 
                    'purpose',
                    'emp_length'
                   ]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Test Train Split
X = df.drop(columns=['ret_PESS','loan_status'])
y1 = df['ret_PESS']
y2 = df['loan_status']

# Train test Splits
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, 
    y1, 
    test_size=0.3, 
    random_state=42
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y2,
    test_size = 0.3,
    random_state = 42,
    stratify = y2
)

# Scale Data
scaler = StandardScaler()

X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

# Random Forest Regressor Model
model_regressor = RandomForestRegressor(
    n_estimators=200,      # number of trees
    max_depth=50,          # let trees grow deep
    min_samples_leaf=18,   # helps prevent overfitting
    max_features="sqrt",   # standard best practice
    random_state=42,
    n_jobs=-1
)

# Logistic Regression Classifier
model_classifier = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Train models
model_regressor.fit(X_train1, y_train1)
model_classifier.fit(X_train2_scaled, y_train2)

joblib.dump(model_regressor, "ret_PESS_model.pkl")
joblib.dump(model_classifier, "loan_status_model.pkl")
joblib.dump(scaler, "classifier_scaler.pkl")

print("Models created successfully")
