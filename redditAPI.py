import praw
import pandas as pd
from datetime import datetime

# Reddit API credentials
reddit = praw.Reddit(
    client_id='v5b2CYNg37amXniM43bNmQ',
    client_secret='cqVeL5VR-vENbiLAjnfC-xoRn45qaQ',
    user_agent="MyRedditSentimentApp/0.1 by noahcrampton"
)

subreddit = reddit.subreddit("all")
posts = []

# You can use .hot(), .new(), or .top(), ill use hot() to get treding postss
for post in subreddit.new(limit=500):
    posts.append({
        "title": post.title,
        "score": post.score,
        "comments": post.num_comments,
        "created": datetime.utcfromtimestamp(post.created_utc),
        "url": post.url,
        "selftext": post.selftext,
        "subreddit": str(post.subreddit)
    })

df = pd.DataFrame(posts).drop_duplicates(subset="title")
df.sort_values(by="created", ascending=False, inplace=True)

print("Columns:", df.columns.tolist())

df.to_csv("data/reddit_all_recent_posts.csv", index=False)
print("Saved to reddit_all_recent_posts.csv")
