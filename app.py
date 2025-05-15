import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
from engagement_model import preprocess_and_train

# --- Initialize News API ---
API_KEY = "7af7d5e56edc4148aac908f2c9f86ac3"  
newsapi = NewsApiClient(api_key=API_KEY)

st.title("📊 Real-Time Social + News Dashboard with Engagement Forecasting")

# --- User Topic Input ---
topic = st.text_input("Enter a topic keyword (e.g., #Fitness, climate change):", "#Fitness")

# --- News Fetching ---
if topic:
    with st.spinner("Fetching news articles..."):
        all_articles = newsapi.get_everything(
            q=topic,
            language='en',
            sort_by='publishedAt',
            page_size=10
        )
    articles = all_articles.get('articles', [])

    st.header(f"📰 Latest News on {topic}")
    if articles:
        for article in articles:
            st.subheader(article['title'])
            st.write(article['description'])
            st.markdown(f"[Read more]({article['url']})")
            st.write(f"Published at: {article['publishedAt']}")
            st.markdown("---")
    else:
        st.write("No news articles found for this topic.")

# --- Load Dataset ---
@st.cache_data
def load_social_data():
    df = pd.read_csv("data/x_posts_with_weather.csv")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    return df

df = load_social_data()

# --- Filter Dataset ---
if topic:
    mask = df['hashtags'].str.contains(topic.replace("#", ""), case=False, na=False)
    filtered_df = df[mask]

    st.header(f"📱 Social Media Posts on {topic}")
    st.write(f"Total posts found: {filtered_df.shape[0]}")

    if not filtered_df.empty:
        st.line_chart(filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).size())
    else:
        st.write("No social media posts found for this topic.")

# --- ML Model Integration ---
if not filtered_df.empty:
    st.subheader("📈 Predicting Engagement (Likes + Retweets)")

    # Rename for compatibility with model
    if 'created_at' in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={"created_at": "timestamp"})

    try:
        with st.spinner("Training ML model..."):
            result = preprocess_and_train(filtered_df)

        st.success("Model trained successfully!")

        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")

        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values

        st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])

    except Exception as e:
        st.error(f"Model error: {e}")
