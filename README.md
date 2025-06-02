Project 4 (158755) – Real-Time Social Media Trend Forecaster

Overview
This project delivers a real-time trend forecasting web app with a primary focus on analyzing fitness-related hashtags (e.g., #Fitness, #Workout, #HealthyLiving) on X and Reddit posts, while also supporting other topics like #ClimateChange and #Ukraine. It integrates social media data with current news headlines (via the News API) and uses NLP and machine learning to extract trending fitness keywords, predict post engagement (likes, retweets, upvotes) for fitness content, and forecast the popularity of fitness topics over 24–48 hours. The tool is deployed as an interactive Streamlit dashboard, featuring visualizations like word clouds and trend curves, with a strong emphasis on fitness trends. A Jupyter notebook documents the full data science workflow, highlighting the analysis and modeling for fitness-related data.
Problem Statement:
Trends on X emerge and fade rapidly. Marketers, influencers, and researchers often struggle to anticipate these shifts. This project addresses that challenge by forecasting trend lifecycles, helping users optimize content timing and stay ahead of competitors.

Objectives

    Extract Trends using NLP from simulated X data and real-time news.

    Predict Engagement using regression models based on post features, sentiment, and weather.

    Forecast Hashtag Popularity with time-series models for 24–48 hour outlooks.

    Visualize Insights in a Streamlit app with interactive plots, word clouds, and filters.

    Ensure Originality by generating a custom dataset augmented with live news and weather data, avoiding API limitations and overused public datasets.


How to run:

    Clone directory from main branch
    run requirements.txt via terminal "pip install -r requirements.txt"
    Run nltk script.py ONCE only. This is for NLP
    Run the data file(s). This includes. redditAPI.py, engagement_model.py
    Run the merge file after parameters and raw data is there. The merge file is called "mergeDATA.py"
    In venv terminal run streamlit app.py

