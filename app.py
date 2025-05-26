import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {e}")
    st.stop()

# Initialize VADER
try:
    sid = SentimentIntensityAnalyzer()
    test_score = sid.polarity_scores("I love this! It's amazing!")
    logger.debug(f"VADER test score: {test_score}")
    if test_score['compound'] == 0:
        st.warning("VADER sentiment analyzer may not be functioning correctly.")
except Exception as e:
    st.error(f"Failed to initialize VADER: {e}")
    st.stop()

# Streamlit page configuration
st.set_page_config(layout="wide")
st.title("📊 Real-Time Social Media Trend Forecaster")


@st.cache_data
def load_combined_data():
    try:
        df = pd.read_csv("data/combined_social_data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        required_columns = ['created_at', 'text']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Data file missing required columns: {required_columns}")
            return pd.DataFrame()
        st.write(f"Loaded CSV with shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        st.error("Data file 'combined_social_data.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


# Compute sentiment with enhanced error handling
def compute_sentiment(text):
    try:
        text = str(text).strip()
        if not text or text.lower() in ['nan', 'none', '']:
            logger.debug(f"Empty or invalid text: {text}")
            return 0.0
        scores = sid.polarity_scores(text)
        compound_score = scores['compound']
        logger.debug(f"Text: {text[:50]}... Sentiment: {compound_score}")
        return compound_score
    except Exception as e:
        logger.error(f"Sentiment computation error for text '{text[:50]}...': {e}")
        return 0.0


# Engagement prediction function
def preprocess_and_train(df):
    try:
        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media']
        df['text_length'] = df['text'].apply(lambda x: len(str(x)))
        df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
        df['is_media'] = df['text'].str.contains('https://t.co', na=False).astype(int)
        X = df[features].fillna(0)
        y = df['engagement'].fillna(0)
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 rows required).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    except Exception as e:
        st.error(f"Error in training model: {e}")
        return None


# Load data
combined_df = load_combined_data()

if combined_df.empty:
    st.warning("No data loaded. Please check 'combined_social_data.csv'.")
    st.stop()

# Input for topic
topic = st.text_input("Enter a topic keyword (e.g., #Fitness, Climate Change):", "#Fitness")

if topic:
    keyword = topic.replace("#", "").lower()
    topic_mask = combined_df['text'].str.lower().str.contains(keyword, na=False)
    filtered_df = combined_df.loc[topic_mask].copy()
    st.write(f"Filtered posts for '{keyword}': {filtered_df.shape[0]}")

    st.header(f"📱 Social Media Posts on {topic}")
    st.write(f"Total posts found: {filtered_df.shape[0]}")

    if not filtered_df.empty:
        # Clean text data
        filtered_df = filtered_df.dropna(subset=['text'])
        filtered_df['text'] = filtered_df['text'].astype(str).replace('', np.nan).dropna()
        st.write(f"Filtered DataFrame shape after cleaning: {filtered_df.shape}")

        # Compute sentiment with debugging
        if 'sentiment' not in filtered_df.columns or filtered_df['sentiment'].isnull().all():
            st.write("Computing sentiment scores...")
            filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)
            st.write("Sample sentiment scores:")
            st.dataframe(filtered_df[['text', 'sentiment']].head(10))
            if (filtered_df['sentiment'] == 0).all():
                st.warning("All sentiment scores are 0. Possible issues with text data or VADER analyzer.")

        # Ensure numeric sentiment and engagement
        filtered_df['sentiment'] = pd.to_numeric(filtered_df['sentiment'], errors='coerce').fillna(0)
        if 'engagement' not in filtered_df.columns:
            engagement_cols = [col for col in ['likes', 'retweets', 'shares'] if col in filtered_df.columns]
            if engagement_cols:
                filtered_df['engagement'] = filtered_df[engagement_cols].sum(axis=1)
            else:
                filtered_df['engagement'] = 0
        filtered_df['engagement'] = pd.to_numeric(filtered_df['engagement'], errors='coerce').fillna(0)
        filtered_df = filtered_df.dropna(subset=["created_at"])

        # Engagement and Sentiment Over Time
        time_series = filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).agg({
            'engagement': 'sum',
            'sentiment': 'mean'
        }).reset_index()

        st.subheader("📊 Engagement & Sentiment Over Time")
        if not time_series.empty and len(time_series) > 1:
            chart_data = pd.DataFrame({
                'Time': time_series['created_at'],
                'Engagement': time_series['engagement'].fillna(0),
                'Sentiment': time_series['sentiment'].fillna(0)
            }).set_index('Time')
            st.line_chart(chart_data[['Engagement', 'Sentiment']])
        else:
            st.warning("Insufficient data for engagement and sentiment trends. Need at least 2 time points.")

        # Sentiment Distribution
        st.subheader("💬 Sentiment Distribution")
        if filtered_df['sentiment'].notnull().sum() > 0:
            sentiment_binned = pd.cut(filtered_df['sentiment'], bins=10)
            sentiment_counts = sentiment_binned.value_counts().sort_index()
            chart_data = pd.DataFrame({
                'Sentiment Range': [str(interval) for interval in sentiment_counts.index],
                'Count': sentiment_counts
            }).set_index('Sentiment Range')
            st.bar_chart(chart_data['Count'])
        else:
            st.warning("No sentiment data available to display distribution.")

        # Sample Posts
        st.subheader("🔍 Sample Posts")
        display_cols = [col for col in ['created_at', 'text', 'sentiment', 'engagement'] if col in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(10))

        # Word Cloud
        st.subheader("🌐 Word Cloud")
        text = " ".join(filtered_df['text'].dropna().tolist())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No valid text available for word cloud. Using sample text.")
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(
                "fitness gym workout motivation health")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # Popular Subtopics
        st.subheader("🔍 Popular Subtopics within this Topic")
        stop_words = set(stopwords.words('english')) - {'run', 'pump'}
        texts = filtered_df['text'].dropna().tolist()
        st.write(f"Number of posts after filtering: {len(texts)}")

        processed_texts = []
        for doc in texts:
            tokens = [
                word for word in word_tokenize(doc.lower())
                if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
            ]
            processed_texts.append(" ".join(tokens))
        processed_texts = [doc for doc in processed_texts if len(doc.split()) > 1]
        st.write(f"Number of posts after cleaning: {len(processed_texts)}")

        if len(processed_texts) >= 5:
            try:
                vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000)
                dtm = vectorizer.fit_transform(processed_texts)
                lda = LatentDirichletAllocation(n_components=3, random_state=42)
                lda.fit(dtm)
                words = vectorizer.get_feature_names_out()
                for i, topic_dist in enumerate(lda.components_):
                    topic_words = [words[i] for i in topic_dist.argsort()[-5:][::-1]]
                    st.write(f"**Topic {i + 1}:** {', '.join(topic_words)}")
            except Exception as e:
                st.warning(f"Failed to perform topic modeling: {e}")
                hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
                st.write(
                    "**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))
        else:
            hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
            st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))

        

        # Optimal Posting Times
        st.subheader("⏰ Optimal Posting Times")
        filtered_df['hour'] = filtered_df['created_at'].dt.hour
        hourly_engagement = filtered_df.groupby('hour')['engagement'].mean().reset_index()
        if not hourly_engagement.empty:
            chart_data = pd.DataFrame({
                'Hour of Day': hourly_engagement['hour'].astype(str),
                'Average Engagement': hourly_engagement['engagement']
            }).set_index('Hour of Day')
            st.bar_chart(chart_data['Average Engagement'])
        else:
            st.warning("No data available for optimal posting times.")

        # Predicting Engagement
        st.subheader("📈 Predicting Engagement")
        try:
            with st.spinner("Training ML model..."):
                result = preprocess_and_train(filtered_df)
            if result:
                st.success("✅ Model trained successfully!")
                st.write(f"**R² Score**: {result['r2_score']:.2f}")
                st.write(f"**RMSE**: {result['rmse']:.2f}")

                # Create chart DataFrame
                chart_df = result['X_test'].copy()
                chart_df['Predicted Engagement'] = result['y_pred']
                chart_df['Actual Engagement'] = result['y_test'].values

                # Map indices back to timestamps if available
                test_indices = result['X_test'].index
                if 'created_at' in filtered_df.columns and not filtered_df.loc[
                    test_indices, 'created_at'].isnull().any():
                    chart_df['Time'] = filtered_df.loc[test_indices, 'created_at']
                    chart_df = chart_df.set_index('Time')
                    st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
                else:
                    st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
                    st.warning("Using post indices as x-axis because 'created_at' is missing or invalid.")
            else:
                st.warning("Failed to train engagement prediction model.")
        except Exception as e:
            st.error(f"❌ Model error: {e}")
    else:
        st.warning(f"No posts found for '{topic}'. Check if the keyword exists in the data.")
