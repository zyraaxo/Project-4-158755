import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
from textblob import TextBlob

# Page config
st.set_page_config(page_title="Social Media Trend Forecaster", layout="wide")

# Header
st.title("Real-Time Social Media Trend Forecaster")
st.markdown("""
Monitor trending fitness hashtags on X in real-time. Enter a hashtag or select a trend to view insights and engagement predictions.
""")

# Sidebar for filters
st.sidebar.header("Filters")
date_range = st.sidebar.slider("Select Date Range (Hours)", 1, 48, 24)
sentiment_filter = st.sidebar.multiselect("Sentiment", ["POSITIVE", "NEGATIVE"], default=["POSITIVE", "NEGATIVE"])

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    keyword = st.text_input("Enter a hashtag (e.g., #Fitness)", "#Fitness")
with col2:
    trending_options = ["#Fitness", "#Run", "#HealthyLiving"]
    selected_trend = st.selectbox("Or select a trending hashtag", trending_options, index=0)
keyword = selected_trend if st.session_state.get("use_trend", False) else keyword

# Load placeholder data
try:
    df = pd.read_csv("data/x_posts_with_weather.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["hashtags"] = df["hashtags"].apply(ast.literal_eval)  # convert string list to real list

    # Live sentiment analysis simulation using TextBlob
    def analyze_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        return "POSITIVE" if polarity >= 0 else "NEGATIVE"

    df["sentiment"] = df["text"].apply(analyze_sentiment)

except FileNotFoundError:
    st.error("Data file not found. Please run generate_placeholder_posts.py.")
    st.stop()

# Filter data
df = df[df["hashtags"].apply(lambda tags: any(keyword.lower() in h.lower() for h in tags))]
df = df[df["sentiment"].isin(sentiment_filter)]
df = df[df["created_at"] >= df["created_at"].max() - pd.Timedelta(hours=date_range)]

# Visualizations
st.header("Trend Insights")

# Word cloud
st.subheader("Related Hashtags")
if not df.empty:
    all_hashtags = [htag for sublist in df["hashtags"] for htag in sublist]
    if all_hashtags:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_hashtags))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("No hashtags found for the selected criteria.")
else:
    st.write("No posts match the selected criteria.")

# Engagement bar chart
st.subheader("Predicted Engagement (Top Posts)")
if not df.empty:
    df_top = df.nlargest(5, "engagement")
    fig = px.bar(
        df_top,
        x="created_at",
        y="engagement",
        title=f"Top Posts for {keyword}",
        labels={"engagement": "Likes + Retweets", "created_at": "Post Time"},
        hover_data=["text"]
    )
    fig.update_layout(xaxis_title="Post Time", yaxis_title="Engagement")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No engagement data available.")

# Hourly engagement breakdown
st.subheader("Hourly Engagement Breakdown")
if not df.empty:
    hourly_df = df.groupby("hour_of_day")["engagement"].mean().reset_index()
    fig2 = px.line(hourly_df, x="hour_of_day", y="engagement", markers=True,
                   title="Average Engagement by Hour",
                   labels={"hour_of_day": "Hour of Day", "engagement": "Avg Engagement"})
    fig2.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No data available for hourly trends.")

# Insights
st.header("Actionable Insights")
if not df.empty:
    peak_hour = df.groupby("hour_of_day")["engagement"].mean().idxmax()
    weather_impact = df.groupby("precipitation")["engagement"].mean().idxmax()
    st.markdown(f"""
    - **Optimal Posting Time**: Post {keyword} around {peak_hour}:00 for maximum engagement.
    - **Recommendation**: Share positive {keyword} content during peak hours on clear days.
    """)
else:
    st.write("Insufficient data for insights.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit by [Your Name] | Data: X API (May 2025) | Deployed: May 2025")
