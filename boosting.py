import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data_train  = pd.read_parquet('archive/Zzzs_train_multi.parquet')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data_train['series_id']=le.fit_transform(data_train['series_id'])

data_train=data_train.drop('timestamp',axis=1)
data_train=data_train.groupby('series_id').apply(lambda group:group.iloc[::12*15]).reset_index(drop=True)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
# Load your separate training and test datasets (replace with your actual data)
# Assuming 'X_train', 'y_train' are training features and labels
# and 'X_test', 'y_test' are test features and labels

X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
print("DATA READ")
# Assuming you have X_train and y_train from your data preparation steps

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Base models
base_models = [
    GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=True),
    LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    SVC(kernel='linear', probability=True, verbose=True),
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
]

# Initialize Gradient Boosting meta-learner
gradient_boost_meta_learner = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=True)

# Create an AdaBoost ensemble
adaboost_model = AdaBoostClassifier(base_estimator=gradient_boost_meta_learner, n_estimators=50, random_state=42)

# Fit the AdaBoost model
adaboost_model.fit(X_train_split, y_train_split)

# Evaluate on validation set
accuracy = adaboost_model.score(X_val, y_val)
print(f"AdaBoost ensemble accuracy: {accuracy:.4f}")