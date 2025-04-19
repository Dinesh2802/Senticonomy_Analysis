import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from streamlit_option_menu import option_menu
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import nltk
import torch
import torch.nn as nn
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Filter warnings
warnings.filterwarnings('ignore')

# NLTK download (required for sentiment analysis)
nltk.download('stopwords')

# Set up the Streamlit page configuration
st.set_page_config(page_icon="🌐", page_title="News Sentiment Analysis", layout="wide")

# Title of the web page
st.markdown('''<h1 style="color: #FFA500;">🌐 <span style="color: orange;">Welcome to News Sentiment Analysis and Economic Impact Visualization</span></h1>''', unsafe_allow_html=True)

# Upload dataset through Streamlit
uploaded_file = st.file_uploader("Upload a Top Headlines NEWS articles file", type=["csv"])

# Initialize df as None to check if file is uploaded
df = None

# Set background image for Streamlit (you can replace this URL with an actual image URL)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://www.example.com/background-image.jpg");  
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}   
[data-testid="stSidebarContent"]{
background-color: rgba(0, 0, 0, 0);
}
</style>
"""
# Apply the background image
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar navigation menu
with st.sidebar:
    selected_page = option_menu('News Sentiment Analysis Dashboard', 
                                ['Home', 'Sentiment Analysis VADER', 'Sentiment Analysis TextBlob', 'KMeans Clustering', 'Sentiment Analysis HuggingFace'],
                                icons=['house', 'bar-chart', 'person', 'briefcase', 'pie-chart'], 
                                default_index=0,
                                orientation='vertical',  
                                styles={
                                    "container": {"padding":"3px", "background-color": "#F5F5F5"},
                                    "icon": {"color": "orange", "font-size": "20px"},
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"3px", "--hover-color": "#f0f0f5"},
                                    "nav-link-selected": {"background-color": "#ADD8E6"},
                                }
                                )

# Home Page
if selected_page == 'Home':
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['title', 'description', 'content']):
            st.warning("Uploaded CSV must contain 'title', 'description', and 'content' columns.")
        else:
            st.write("CSV file uploaded successfully!")
            st.markdown('<h1 style="color: skyblue;">Sentiment Analysis and Visualization</h1>', unsafe_allow_html=True)
    else:
        st.write("Please upload a CSV file to proceed.")

# VADER Sentiment Analysis Page
elif selected_page == 'Sentiment Analysis VADER':
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['title', 'description', 'content']):
            st.warning("Uploaded CSV must contain 'title', 'description', and 'content' columns.")
        else:
            st.header("Sentiment Analysis using VADER (NLTK)")

            # Fill NaN values with empty strings
            df = df.fillna("")

            # Convert columns to string type
            df["title"] = df["title"].astype(str)
            df["description"] = df["description"].astype(str)
            df["content"] = df["content"].astype(str)

            # Sentiment Analysis with VADER
            sia = SentimentIntensityAnalyzer()
            df["sentiment_vader"] = df[["title", "description", "content"]].astype(str).agg(" ".join, axis=1).apply(lambda x: sia.polarity_scores(x)["compound"])
            df["sentiment_label_vader"] = df["sentiment_vader"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

            # Sentiment statistics and counts
            st.write(df["sentiment_vader"].describe())
            st.write(df["sentiment_label_vader"].value_counts())

            # Pie chart for sentiment distribution (VADER)
            sentiment_counts = df['sentiment_label_vader'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon', 'lightgreen'])
            ax.set_title('Sentiment Analysis Distribution (VADER)')
            st.pyplot(fig)

            # Sentiment VADER Label Distribution (Bar Plot)
            sentiment_vader_counts = df['sentiment_label_vader'].value_counts()

            # Create a bar plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.bar(sentiment_vader_counts.index, sentiment_vader_counts.values, color = ['#a4b6a3', '#d68c1e', '#e4d3c3'])

            # Set titles and labels
            ax.set_title('Sentiment VADER Label Distribution (Positive, Negative, Neutral)')
            ax.set_xlabel('Sentiment Label')
            ax.set_ylabel('Frequency')

            # Display the plot in Streamlit
            st.pyplot(fig)

    else:
        st.write("Please upload a CSV file to proceed.")

# Sentiment Analysis using TextBlob Page
elif selected_page == 'Sentiment Analysis TextBlob':
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['title', 'description', 'content']):
            st.warning("Uploaded CSV must contain 'title', 'description', and 'content' columns.")
        else:
            st.header("Sentiment Analysis using TextBlob")

            # Ensure no missing values
            df["description"] = df["description"].fillna("")

            def classify_sentiment(polarity):
                if polarity > 0.05:
                    return "Positive"
                elif polarity < -0.05:
                    return "Negative"
                else:
                    return "Neutral"

            # Apply TextBlob Sentiment Analysis for Polarity & Subjectivity
            df["sentiment_polarity"] = df["description"].apply(lambda text: TextBlob(text).sentiment.polarity)
            df["sentiment_subjectivity"] = df["description"].apply(lambda text: TextBlob(text).sentiment.subjectivity)

            # Classify each sentiment based on polarity
            df["sentiment_label_textblob"] = df["sentiment_polarity"].apply(classify_sentiment)

            # Display the TextBlob sentiment analysis results
            st.write(df[["description", "sentiment_polarity", "sentiment_subjectivity", "sentiment_label_textblob"]])

            # Pie chart for sentiment distribution (textblob)
            sentiment_counts = df['sentiment_label_textblob'].value_counts()  # Corrected to use 'sentiment_label_textblob'

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors = ['#4682B4', '#00BFFF', '#7FFFD4'])
            ax.set_title('Sentiment Analysis Distribution (TextBlob)')
            st.pyplot(fig)


            # Sentiment Label Distribution (Bar Plot) for TextBlob
            sentiment_counts = df['sentiment_label_textblob'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=['lightgreen', 'lightyellow', 'lightcoral'])
            ax.set_title('Sentiment TextBlob Label Distribution (Positive, Negative, Neutral)')
            ax.set_xlabel('Sentiment Label')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to proceed.")
       
# KMeans Clustering Page
elif selected_page == 'KMeans Clustering':
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['title', 'description', 'content']):
            st.warning("Uploaded CSV must contain 'title', 'description', and 'content' columns.")
        else:
            st.header("KMeans Clustering on Top Headlines NEWS Categories")

            # Fill NaN values with empty strings
            df = df.fillna("")

            # Convert columns to string type
            df["title"] = df["title"].astype(str)
            df["description"] = df["description"].astype(str)
            df["content"] = df["content"].astype(str)

            # KMeans Clustering for Text
            vect = TfidfVectorizer(stop_words="english")
            x = vect.fit_transform(df["title"] + " " + df["description"] + " " + df["content"])

            model = KMeans(n_clusters=7, random_state=42)
            df["cluster"] = model.fit_predict(x)

            # Define Cluster Names
            cluster_names = {
                0: "Business",
                1: "Technology",
                2: "Health",
                3: "Entertainment",
                4: "Politics",
                5: "Sports",
                6: "Science"
            }

            # Map cluster labels to their corresponding names
            df["cluster_name"] = df["cluster"].map(cluster_names)

            # Display Cluster Counts
            st.subheader("Sentiment Analysis using Clustering Algorithms")
            st.write("Cluster Counts are displayed below")
            cluster_counts = df["cluster_name"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            st.dataframe(cluster_counts)

            # Visualizing clusters with a pie chart (Plotly)
            fig = px.pie(cluster_counts, values="Count", names="Cluster", title="NEWS Article Cluster Distribution")
            st.plotly_chart(fig)

    else:
        st.write("Please upload a CSV file to proceed.")

# Sentiment Analysis using HuggingFace Transformer (BERT)s
elif selected_page == 'Sentiment Analysis HuggingFace':
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['title', 'description', 'content']):
            st.warning("Uploaded CSV must contain 'title', 'description', and 'content' columns.")
        else:
            st.header("Sentiment Analysis using HuggingFace Transformer")

            # Fill NaN values with empty strings
            df = df.fillna("")

            # Convert columns to string type
            df["title"] = df["title"].astype(str)
            df["description"] = df["description"].astype(str)
            df["content"] = df["content"].astype(str)

            # Load the pre-trained sentiment analysis pipeline from Hugging Face
            sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

            # Perform sentiment analysis using the BERT-based model
            df["sentiment_huggingface"] = df["content"].apply(lambda x: sentiment_pipeline(x)[0]['label'])

            # Display HuggingFace sentiment results
            st.write(df[['content', 'sentiment_huggingface']])

            # Sentiment distribution plot (HuggingFace)
            sentiment_counts = df['sentiment_huggingface'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightseagreen', 'lightgray'])
            ax.set_title('Sentiment Distribution (HuggingFace Transformer)')
            st.pyplot(fig)

            # Count occurrences of each unique title
            title_counts = df["title"].value_counts().head(10)  # Top 10 titles

            # Create a pie chart for the most frequent news titles
            st.header("Top 10 Most Frequent News Titles")
            fig = plt.figure(figsize=(20, 10))
            plt.pie(title_counts, labels=title_counts.index, autopct="%1.1f%%", colors=plt.cm.Paired.colors[:len(title_counts)])
            plt.title("Top 10 Most Frequent News Titles")
            st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to proceed.")
