import streamlit as st
from newsapi import NewsApiClient
import pandas as pd

# Initialize NewsApiClient
API_KEY = "7af7d5e56edc4148aac908f2c9f86ac3"  # Replace with your key or keep for testing
newsapi = NewsApiClient(api_key=API_KEY)

st.title("Interactive Data Journalism Dashboard: Social Media + News")

# User input for topic
topic = st.text_input("Enter a topic keyword (e.g., #Fitness, climate change):", "#Fitness")

# Fetch news articles related to topic
if topic:
    with st.spinner("Fetching news articles..."):
        all_articles = newsapi.get_everything(
            q=topic,
            language='en',
            sort_by='publishedAt',
            page_size=10
        )
    articles = all_articles.get('articles', [])

    st.header(f"Latest News on {topic}")
    if articles:
        for article in articles:
            st.subheader(article['title'])
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")
            st.write(f"Published at: {article['publishedAt']}")
            st.markdown("---")
    else:
        st.write("No news articles found for this topic.")


# Load your social media data (assuming you have a CSV with hashtag data)
@st.cache_data
def load_social_data():
    df = pd.read_csv("data/x_posts_with_weather.csv")
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df


df = load_social_data()

# Filter social media data by topic/hashtag
if topic:
    mask = df['hashtags'].str.contains(topic.replace("#", ""), case=False, na=False)
    filtered_df = df[mask]
    st.header(f"Social Media Posts on {topic}")
    st.write(f"Total posts found: {filtered_df.shape[0]}")

    if not filtered_df.empty:
        st.line_chart(filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).size())
    else:
        st.write("No social media posts found for this topic.")

# Add more interactive filters and charts as needed
