import pandas as pd
from textblob import TextBlob

# This file merges the X and reddit data, sorts by text, engagement etc to ensure all data is covered
x_posts = pd.read_csv("data/x_posts_with_weather.csv")
reddit_posts = pd.read_csv("data/reddit_all_recent_posts.csv")

# Combine 'title' and 'selftext' into a single 'text' column
reddit_posts['text'] = reddit_posts['title'].fillna('') + ". " + reddit_posts['selftext'].fillna('')

reddit_posts['created_at'] = pd.to_datetime(reddit_posts['created'], errors='coerce')

reddit_posts['hour_of_day'] = reddit_posts['created_at'].dt.hour
reddit_posts['is_weekend'] = reddit_posts['created_at'].dt.dayofweek >= 5

reddit_posts['engagement'] = reddit_posts['score'] + reddit_posts['comments']

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

reddit_posts['sentiment'] = reddit_posts['text'].apply(get_sentiment)

reddit_renamed = reddit_posts.rename(columns={
    "created_at": "created_at",
    "text": "text",
    "engagement": "engagement",
    "sentiment": "sentiment",
    "hour_of_day": "hour_of_day",
    "is_weekend": "is_weekend"
})

reddit_cleaned = reddit_renamed[[
    "created_at", "text", "sentiment", "hour_of_day", "is_weekend", "engagement"
]]

x_posts['created_at'] = pd.to_datetime(x_posts['created_at'], errors='coerce')
x_posts_cleaned = x_posts[[
    "created_at", "text", "sentiment", "hour_of_day", "is_weekend", "engagement"
]]

# --- Merge datasets ---
combined_df = pd.concat([x_posts_cleaned, reddit_cleaned], ignore_index=True)

combined_df.dropna(subset=["text", "engagement"], inplace=True)

combined_df.to_csv("data/combined_social_data.csv", index=False)
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

print(combined_df)