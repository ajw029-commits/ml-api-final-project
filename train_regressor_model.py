### Import Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Import Data For training
url = 'https://raw.githubusercontent.com/MattDBailey/ANOP330/refs/heads/main/Data/BucknellLendingClubHistoricalData.csv'
past_df = pd.read_csv(url)

### Prep data
# Remove unneeded columns
drop_cols = ['loan_status',   # Post Outcome
             'id',            # not useful
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
X = df.drop(columns=['ret_PESS'])
y = df['ret_PESS']


# Train test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor Model
model = RandomForestRegressor(
    n_estimators=300,      # number of trees
    max_depth=None,        # let trees grow fully
    min_samples_leaf=10,   # helps prevent overfitting
    max_features="sqrt",   # standard best practice
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, "ret_PESS_model.pkl")

print("Model saved as churn_model.pkl")
