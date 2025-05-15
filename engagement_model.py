import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def extract_hashtags(text):
    return re.findall(r"#\w+", str(text).lower())

def preprocess_and_train(df, weather_features=None):
    if weather_features is None:
        weather_features = ['temperature', 'humidity', 'wind_speed']

    # Target variable
    df['engagement'] = df['likes'] + df['retweets']

    # Sentiment score
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    # Time features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # Text length
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))

    # Hashtags
    df['hashtags'] = df['text'].apply(extract_hashtags)
    all_hashtags = [tag for sublist in df['hashtags'] for tag in sublist]
    top_hashtags = [tag for tag, _ in Counter(all_hashtags).most_common(5)]

    for tag in top_hashtags:
        df[f'hashtag_{tag}'] = df['hashtags'].apply(lambda tags: int(tag in tags))

    # Final feature list
    feature_cols = ['sentiment', 'hour', 'dayofweek', 'text_length']
    feature_cols += [f'hashtag_{tag}' for tag in top_hashtags]
    feature_cols += [col for col in weather_features if col in df.columns]

    df = df.dropna(subset=feature_cols + ['engagement'])

    X = df[feature_cols]
    y = df['engagement']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "model": model,
        "r2_score": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
