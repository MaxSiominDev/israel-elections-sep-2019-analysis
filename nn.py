import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from ai import preprocessor, TrainingResult
import feature


def predict_with_nn(df: pd.DataFrame) -> TrainingResult:
    X = df[feature.all_features_names]
    y = df['winner_party']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    y_categorical = to_categorical(y_encoded)

    y_train_labels = y_train.values

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_processed, y_categorical, epochs=200, batch_size=32, validation_split=0.1)

    y_pred_cat = model.predict(X_test_processed)
    y_pred_encoded = np.argmax(y_pred_cat, axis=1)
    y_pred = le.inverse_transform(y_pred_encoded)

    return TrainingResult(X_test, y_train_labels, y_test, y_pred)
