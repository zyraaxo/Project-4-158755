import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from xgboost import XGBRegressor
import numpy as np
import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk_resources = [
    'punkt', 'punkt_tab', 'stopwords', 'vader_lexicon',
    'averaged_perceptron_tagger', 'wordnet', 'omw-1.4'
]
for res in nltk_resources:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res, quiet=True)

# Streamlit page configuration
st.set_page_config(page_title="Social Trends Forecaster", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #fafafa; }
        h1 { color: #1f77b4; }
        .stButton>button { background-color: #1f77b4; color: white; border-radius: 5px; }
        .stDownloadButton>button { background-color: #2ca02c; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Real-Time Social Media Trend Forecaster")

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

@st.cache_data
def load_combined_data(_version):  # Add a dummy parameter to force cache invalidation
    try:
        df = pd.read_csv("data/combined_social_data.csv")
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        required_columns = ['created_at', 'text']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Data file missing required columns: {required_columns}")
            return pd.DataFrame()
        logger.debug(f"Loaded CSV with shape: {df.shape}, Columns: {list(df.columns)}")
        st.write(f"Raw dataset date range: {df['created_at'].min()} to {df['created_at'].max()}")
        return df
    except FileNotFoundError:
        st.error("Data file 'combined_social_data.csv' not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

# Use a timestamp to invalidate the cache
combined_df = load_combined_data(_version=datetime.now().timestamp())

@st.cache_data
def load_recent_news():
    try:
        newsapi = NewsApiClient(api_key="7af7d5e56edc4148aac908f2c9f86ac3")
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q="*", from_param=start_date, to=end_date, language='en', page_size=1000)
        news_df = pd.DataFrame([{
            "published_at": a['publishedAt'],
            "title": a['title'],
            "description": a['description']
        } for a in articles['articles']])
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])
        news_df['text'] = news_df['title'].fillna('') + " " + news_df['description'].fillna('')
        return news_df
    except Exception as e:
        st.warning(f"Failed to fetch news: {e}")
        return pd.DataFrame()

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

def add_extra_features(df):
    df['emoji_count'] = df['text'].str.count(r'[😀-🙏]')
    df['question_flag'] = df['text'].str.contains(r'\?').astype(int)
    df['text_length_log'] = np.log1p(df['text'].apply(len))
    df['capital_word_count'] = df['text'].str.findall(r'\b[A-Z]{2,}\b').apply(len)
    df['punctuation_count'] = df['text'].str.count(r'[.!?]')
    df['text_length'] = df['text'].apply(len)
    df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
    df['is_media'] = df['text'].str.contains('https://t.co', na=False).astype(int)
    df['hour'] = df['created_at'].dt.hour
    df['is_weekend'] = df['created_at'].dt.weekday.isin([5, 6]).astype(int)
    return df

def extract_topics(texts):
    stop_words = set(stopwords.words('english')) - {'run', 'pump'}
    processed_texts = [
        " ".join([
            word for word in word_tokenize(doc.lower())
            if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
        ]) for doc in texts if isinstance(doc, str) and doc.strip()
    ]
    processed_texts = [doc for doc in processed_texts if len(doc.strip().split()) > 1]
    if len(processed_texts) < 5:
        return [0] * len(texts)
    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
        dtm = vectorizer.fit_transform(processed_texts)
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(dtm)
        topics = lda.transform(dtm).argmax(axis=1)
        padded_topics = [topics[i] if i < len(topics) else 0 for i in range(len(texts))]
        return padded_topics
    except Exception as e:
        logger.error(f"Topic modeling error: {e}")
        return [0] * len(texts)

def drop_constant_columns(df):
    return df.loc[:, df.nunique() > 1]

def preprocess_and_train(df, model_choice="RandomForest"):
    try:
        features = ['sentiment', 'text_length', 'hashtag_count', 'is_media', 'hour', 'is_weekend']
        X = df[features].fillna(0)
        y = df['engagement'].fillna(0)
        if len(X) < 10:
            raise ValueError("Not enough data for training (minimum 10 rows required).")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_choice == "RandomForest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
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

def hybrid_prophet_xgb(df):
    try:
        logger.info("Starting hybrid Prophet+XGBoost prediction")
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
        prophet_df = df[['created_at', 'engagement']].copy()
        prophet_df = prophet_df.rename(columns={'created_at': 'ds', 'engagement': 'y'})
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)
        prophet_features = forecast[['ds', 'trend', 'weekly', 'daily']]
        merged_df = df.merge(prophet_features, left_on='created_at', right_on='ds', how='left').drop(columns=['ds'])
        features = ['sentiment', 'hour', 'is_weekend', 'trend', 'weekly', 'daily']
        merged_df = merged_df.dropna(subset=features + ['engagement'])
        if len(merged_df) < 10:
            raise ValueError("Not enough data for XGBoost training.")
        X = merged_df[features]
        y = merged_df['engagement']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        return {
            "prophet_forecast": forecast[['ds', 'yhat']],
            "xgb_model": xgb_model,
            "xgb_features": features,
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "X_test": X_test.assign(created_at=merged_df.loc[X_test.index, 'created_at']),
            "y_test": y_test,
            "y_pred": y_pred
        }
    except Exception as e:
        logger.error(f"Hybrid model error: {str(e)}", exc_info=True)
        return None

# Load data
combined_df = load_combined_data()
if combined_df.empty:
    st.warning("No data loaded. Please check 'combined_social_data.csv'.")
    st.stop()

# Display raw dataset details
st.info(f"Raw dataset size: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
st.write(f"Raw dataset date range: {combined_df['created_at'].min()} to {combined_df['created_at'].max()}")
st.subheader("Sample of Raw Dataset")
st.dataframe(combined_df[['created_at', 'text']].tail(5))  # Show last 5 rows

# Sidebar
st.sidebar.title("Filter Settings")
keyword = st.sidebar.text_input("Enter a topic keyword:", "Fitness").lower().replace("#", "")
show_news = st.sidebar.checkbox("📰 Show Latest News Headlines")
model_choice = st.sidebar.selectbox("Select Regression Model", ["RandomForest", "GradientBoosting"])

# Filter data
filtered_df = combined_df[combined_df['text'].str.lower().str.contains(keyword, na=False)].copy()
st.info(f"Dataset size after keyword filtering: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
if filtered_df.empty:
    st.warning(f"No posts found for '{keyword}'. Check if the keyword exists in the data.")
    st.stop()

    if show_news:
        news_df = load_recent_news()
        if not news_df.empty:
            st.subheader("🗞 Recent News Highlights")
            for i, row in news_df.head(5).iterrows():
                st.markdown(f"**{row['published_at'].strftime('%Y-%m-%d %H:%M')}** - {row['title']}")
        else:
            st.info("No recent news available.")

# Display filtered dataset details
st.info(f"Filtered dataset sizssssse: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")
st.write(f"Filtered dataset date range: {filtered_df['created_at'].min()} to {filtered_df['created_at'].max()}")
st.subheader("Sample of Filtered Dataset")
st.dataframe(filtered_df[['created_at', 'text']].tail(5))  # Show last 5 rows

# Preprocess data
filtered_df = filtered_df.dropna(subset=['text', 'created_at'])
filtered_df['text'] = filtered_df['text'].astype(str).replace('', np.nan).dropna()
filtered_df = add_extra_features(filtered_df)
filtered_df['sentiment'] = filtered_df['text'].apply(compute_sentiment)
if (filtered_df['sentiment'] == 0).all():
    st.warning("All sentiment scores are 0. Possible issues with text data or VADER analyzer.")
filtered_df['engagement'] = pd.to_numeric(filtered_df.get('engagement', 0), errors='coerce').fillna(0)
if 'engagement' not in filtered_df.columns or filtered_df['engagement'].sum() == 0:
    engagement_cols = [col for col in ['likes', 'retweets', 'shares'] if col in filtered_df.columns]
    if engagement_cols:
        filtered_df['engagement'] = filtered_df[engagement_cols].sum(axis=1)
    else:
        filtered_df['engagement'] = 0
filtered_df['topic'] = extract_topics(filtered_df['text'].tolist())
st.success(f"✅ Total filtered posts: {filtered_df.shape[0]}")

# Time series aggregation
time_df = filtered_df.groupby(filtered_df['created_at'].dt.floor('H')).agg({
    'sentiment': 'mean',
    'engagement': 'sum',
    'topic': lambda x: x.mode()[0] if not x.mode().empty else 0,
    'hour': lambda x: x.mode()[0] if not x.mode().empty else 0,
    'is_media': 'mean'
}).dropna()

# Visualizations
st.header("📈 Topic-Driven Engagement Forecasting")

st.header("📊 Dataset Overview")
st.info(f"Raw dataset size: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
st.info(f"Filtered dataset size: {filtered_df.shape[0]} rows, {filtered_df.shape[1]} columns")

# Engagement & Sentiment Over Time
st.subheader("📊 Engagement & Sentiment Over Time")
if not time_df.empty and len(time_df) > 1:
    chart_data = pd.DataFrame({
        'Time': time_df.index,
        'Engagement': time_df['engagement'].fillna(0),
        'Sentiment': time_df['sentiment'].fillna(0)
    }).set_index('Time')
    st.line_chart(chart_data[['Engagement', 'Sentiment']])
else:
    st.warning("Insufficient data for engagement and sentiment trends.")

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

# Topic vs. Average Engagement
st.subheader("📌 Topic vs. Average Engagement")
if filtered_df['topic'].nunique() > 1:
    chart_data = filtered_df.groupby('topic')['engagement'].mean()
    st.bar_chart(chart_data)
else:
    st.warning("Insufficient topic diversity for engagement analysis.")

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
    st.warning("No valid text for word cloud. Using sample text.")
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(
        "fitness gym workout motivation health")
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Popular Subtopics
st.subheader("🔍 Popular Subtopics")
stop_words = set(stopwords.words('english')) - {'run', 'pump'}
texts = filtered_df['text'].dropna().tolist()
processed_texts = [
    " ".join([
        word for word in word_tokenize(doc.lower())
        if (word.isalnum() or word.startswith('#') or word in ['🦵🏽', '💪🏽']) and word not in stop_words
    ]) for doc in texts if isinstance(doc, str) and doc.strip()
]
processed_texts = [doc for doc in processed_texts if len(doc.strip().split()) > 1]
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
        st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))
else:
    hashtags = filtered_df['text'].str.findall(r'#\w+').explode().value_counts().head(5)
    st.write("**Top Hashtags**: " + ", ".join(hashtags.index if not hashtags.empty else ["No hashtags found"]))

# Optimal Posting Times
st.subheader("⏰ Optimal Posting Times")
hourly_engagement = filtered_df.groupby('hour')['engagement'].mean().reset_index()
if not hourly_engagement.empty:
    chart_data = pd.DataFrame({
        'Hour of Day': hourly_engagement['hour'].astype(str),
        'Average Engagement': hourly_engagement['engagement']
    }).set_index('Hour of Day')
    st.bar_chart(chart_data['Average Engagement'])
else:
    st.warning("No data available for optimal posting times.")

# Forecasting with ARIMA
st.subheader("📅 Forecast with ARIMA")
if len(time_df) >= 5:
    try:
        model = ARIMA(time_df['engagement'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=24)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()
        future_index = pd.date_range(start=time_df.index.max() + pd.Timedelta(hours=1), periods=24, freq='H')
        fig, ax = plt.subplots()
        ax.plot(future_index, forecast_mean, label='Forecasted Engagement', color='tab:blue')
        ax.fill_between(future_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                        color='blue', alpha=0.2, label='95% Confidence Interval')
        ax.set_xlabel('Time')
        ax.set_ylabel('Engagement')
        ax.set_title('ARIMA 24-Hour Engagement Forecast')
        ax.legend()
        st.pyplot(fig)
        actual = time_df['engagement'].iloc[-24:].values
        predicted = model_fit.predict(start=len(time_df)-24, end=len(time_df)-1)
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        st.info(f"RMSE (last 24h backtest): {rmse:.2f}")
    except Exception as e:
        st.warning(f"ARIMA Forecast failed: {e}")
else:
    st.warning("Insufficient data for ARIMA forecast (minimum 5 data points).")

# Forecasting with Prophet
st.subheader("🔮 Forecast with Prophet")
try:
    prophet_df = time_df.reset_index().rename(columns={'timestamp': 'ds', 'engagement': 'y'})[['ds', 'y']]
    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=24, freq='H')
    forecast = prophet_model.predict(future)
    fig1 = prophet_model.plot(forecast)
    st.pyplot(fig1)
    fig2 = prophet_model.plot_components(forecast)
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Prophet forecast error: {e}")

# Time Series Regression with VAR
st.subheader("🧠 Time Series Regression with VAR")
try:
    model_data = time_df[['engagement', 'sentiment', 'topic', 'hour', 'is_media']]
    model_data = drop_constant_columns(model_data)
    if len(model_data.columns) < 2:
        st.warning("Not enough variable columns for VAR model after dropping constants.")
    elif len(model_data) < 2:
        st.warning("Not enough data points for VAR model.")
    else:
        model = VAR(model_data)
        results = model.fit(maxlags=1)
        forecast = results.forecast(model_data.values[-1:], steps=24)
        forecast_df = pd.DataFrame(forecast, columns=model_data.columns)
        st.line_chart(forecast_df[['engagement']])
        with st.expander("Show VAR Coefficients"):
            st.dataframe(results.params)
except Exception as e:
    st.warning(f"VAR model error: {e}")

# Hybrid Prophet+XGBoost
st.subheader("Hybrid Prophet + XGBoost Forecast")
try:
    with st.spinner("Training hybrid model..."):
        result = hybrid_prophet_xgb(filtered_df)
    if result:
        st.success("Hybrid model trained successfully!")
        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(result['prophet_forecast']['ds'], result['prophet_forecast']['yhat'], label='Prophet Forecast')
        plt.xlabel('Time')
        plt.ylabel('Engagement')
        plt.title('Prophet Time Series Forecast')
        plt.legend()
        st.pyplot(fig)
        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values
        if 'created_at' in chart_df.columns and not chart_df['created_at'].isnull().any():
            chart_df = chart_df.set_index('created_at')
            st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
        else:
            st.line_chart(chart_df[['Predicted Engagement', 'Actual Engagement']])
            st.warning("Using post indices as x-axis because 'created_at' is missing or invalid.")
except Exception as e:
    st.error(f"Hybrid model error: {e}")

# Regression Model
st.subheader("📈 Predict Engagement (Regression)")
try:
    with st.spinner("Training regression model..."):
        result = preprocess_and_train(filtered_df, model_choice)
    if result:
        st.success(f"✅ {model_choice} model trained successfully!")
        st.write(f"**R² Score**: {result['r2_score']:.2f}")
        st.write(f"**RMSE**: {result['rmse']:.2f}")
        chart_df = result['X_test'].copy()
        chart_df['Predicted Engagement'] = result['y_pred']
        chart_df['Actual Engagement'] = result['y_test'].values
        fig, ax = plt.subplots()
        ax.plot(chart_df.index, chart_df['Predicted Engagement'], label='Predicted', color='tab:purple')
        ax.plot(chart_df.index, chart_df['Actual Engagement'], label='Actual', color='tab:red')
        ax.set_title('Predicted vs Actual Engagement')
        ax.set_xlabel('Post Index')
        ax.set_ylabel('Engagement')
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"Regression model error: {e}")

# Sample Posts
st.subheader("🔍 Sample Posts")
display_cols = [col for col in ['created_at', 'text', 'sentiment', 'engagement'] if col in filtered_df.columns]
st.dataframe(filtered_df[display_cols].tail(10))

# Download Data
st.download_button("📥 Download Data", filtered_df.to_csv(index=False), file_name="filtered_topic_data.csv")
