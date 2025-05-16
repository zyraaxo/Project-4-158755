import praw
import pandas as pd

# Reddit API credentials
reddit = praw.Reddit(
    client_id='v5b2CYNg37amXniM43bNmQ',
    client_secret='cqVeL5VR-vENbiLAjnfC-xoRn45qaQ',
    user_agent="MyRedditSentimentApp/0.1 by noahcrampton"
)

# Fetch posts from a subreddit
subreddit = reddit.subreddit("fitness")

posts = []
for post in subreddit.hot(limit=50):  # or .new, .top
    posts.append({
        "title": post.title,
        "score": post.score,
        "comments": post.num_comments,
        "created": post.created_utc,
        "url": post.url,
        "selftext": post.selftext
    })

df = pd.DataFrame(posts)
print(df.head())
