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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your separate training and test datasets (replace with your actual data)
# Assuming 'X_train', 'y_train' are training features and labels
# and 'X_test', 'y_test' are test features and labels

X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]


X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("Data_read")

# Base models
base_models = [
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42,verbose=True)),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42,verbose=True,n_jobs=-1)),
    ('svm', SVC(kernel='linear', probability=True,verbose=True)),
    ('knn', KNeighborsClassifier(n_neighbors=5,n_jobs=-1))
]

# Meta-learner (Random Forest classifier)
meta_learner = RandomForestClassifier(n_estimators=100, random_state=42,verbose=True,n_jobs=-1)

# Create the stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_learner, cv=6,verbose=True)

# Train the stacking model
stacking_model.fit(X_train_split, y_train_split)

# Evaluate on test set
accuracy = stacking_model.score(X_val, y_val)
print(f"Stacking ensemble accuracy: {accuracy:.4f}")



                                        