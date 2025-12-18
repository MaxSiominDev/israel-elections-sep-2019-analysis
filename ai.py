from collections import Counter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import prepare_df
import feature
import nn

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

class TrainingResult:
    def __init__(self, X_test, y_train, y_test, y_pred):
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature.numeric_features_names),
        ('cat', OneHotEncoder(handle_unknown='ignore'), feature.categorical_features_names),
    ],
)


def predict(df: pd.DataFrame, model) -> TrainingResult:
    X = df[feature.all_features_names]
    y = df['winner_party']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=69,
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )

    print("Fitting...")
    pipeline.fit(X_train, y_train)

    print("Predicting...")
    y_pred = pipeline.predict(X_test)

    return TrainingResult(X_test, y_train, y_test, y_pred)

def calculate_weighted_random_accuracy(result: TrainingResult) -> float:
    counts = Counter(result.y_train)
    if hasattr(result.y_test, 'unique'):
        classes = result.y_test.unique()
    else:
        classes = np.unique(result.y_test)
    probs = np.array([counts[c] / len(result.y_train) for c in classes])
    y_weighted_random = np.random.choice(classes, size=len(result.y_test), p=probs)
    return accuracy_score(result.y_test, y_weighted_random)

def verify(result: TrainingResult):
    accuracy = accuracy_score(result.y_test, result.y_pred)
    print("Accuracy: ", accuracy)

    random_accuracy = calculate_weighted_random_accuracy(result)
    print("Weighted random accuracy: ", random_accuracy)

    df = result.X_test
    df['winner_party'] = result.y_test
    df['prediction'] = result.y_pred
    df['correct'] = df['winner_party'] == df['prediction']

    df['has_cluster'] = df['socio_econ_cluster_2015'].notna().astype(int)

    result = df.groupby('has_cluster')['correct'].mean() * 100
    print(f"Cluster impact: ${result}")

    print(df.head(3))

models = [
    RandomForestClassifier(n_estimators=100, random_state=69),
    LogisticRegression(max_iter=1000, random_state=69),
    KNeighborsClassifier(n_neighbors=5),
    GradientBoostingClassifier(n_estimators=100, random_state=69),
]

if __name__ == "__main__":
    dataframe = prepare_df.prepare_df()

    for model in models:
        print(f"For {model}:")
        training_result = predict(dataframe, model)
        verify(training_result)
        print("\n\n\n")

    verify(nn.predict_with_nn(dataframe))
