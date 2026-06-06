import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic_preprocessed.csv")

# Pisahkan fitur dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]