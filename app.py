
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ucimlrepo import fetch_ucirepo


app = FastAPI()

# Fetch dataset
student_performance = fetch_ucirepo(id=320)

# Data
X = student_performance.data.features
y = student_performance.data.targets

# Select target
target = "G3"
y = y[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "student_performance_model.pkl")


# Input schema
class StudentPerformanceInput(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int


# Prepare input
def prepare_input(request_data):
    input_dict = request_data.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Match training feature order
    input_df = input_df[X.columns]

    return input_df


# Endpoints
@app.get("/")
def root():
    return {"message": "Student Performance API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(input_data: StudentPerformanceInput):

    input_df = prepare_input(input_data)

    prediction = float(model.predict(input_df)[0])

    return {
        "predicted_grade": round(prediction, 2),
        "target": "G3"
    }