import pandas as pd
import numpy as np

# Simulate 100 posts
dates = pd.date_range("2025-05-15 00:00", "2025-05-16 23:00", freq="15min")[:100]
hashtags = ["#Fitness", "#Run", "#HealthyLiving"]
data = []

for i, date in enumerate(dates):
    hashtag = np.random.choice(hashtags, p=[0.5, 0.3, 0.2])
    likes = np.random.randint(0, 100)
    retweets = np.random.randint(0, 20)
    data.append({
        "created_at": date,
        "text": f"Sample post {i+1}: Loving my workout! {hashtag}",
        "likes": likes,
        "retweets": retweets,
        "replies": np.random.randint(0, 10),
        "language": "en",
        "hashtags": [hashtag, "#Motivation"] if np.random.rand() > 0.5 else [hashtag],
        "sentiment": np.random.choice(["POSITIVE", "NEGATIVE"], p=[0.8, 0.2]),
        "hour_of_day": date.hour,
        "is_weekend": date.weekday() >= 5,
        "temperature": np.random.uniform(10, 20),  # Auckland, May
        "precipitation": np.random.choice([0, 0.5, 1], p=[0.7, 0.2, 0.1]),
        "engagement": likes + retweets
    })

df = pd.DataFrame(data)
df.to_csv("data/x_posts_with_weather.csv", index=False)