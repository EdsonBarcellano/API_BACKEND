import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('original.csv')

# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna(subset=['label (fail=1, pass=0)'])

# Split into X and y
y = df['label (fail=1, pass=0)']
X = df.drop(columns=['label (fail=1, pass=0)'])

# Encode categorical variables
X = pd.get_dummies(X)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Save feature columns
os.makedirs("trained_data", exist_ok=True)
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define classifier pipeline
clf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # ✅ this handles class imbalance!
        random_state=42
    ))
])

# Train model
clf_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(clf_pipeline, 'trained_data/model_cls.pkl')

print("✅ Training complete. Model saved to 'trained_data/model_cls.pkl'")
