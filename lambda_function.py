import json
import requests
import pandas as pd
import time
import boto3
import sqlalchemy
import os
import logging
from sqlalchemy.types import String, DateTime

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment Variables
API_KEY = os.environ.get("NEWS_API_KEY")
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_NAME = os.environ.get('DB_NAME')
S3_BUCKET = os.environ.get('S3_BUCKET')

categories = ["business", "entertainment", "general", "health", "science", "sports", "technology"]

def fetch_news(timeout_limit=30):
    articles = []
    start_time = time.time()

    for category in categories:
        # Check if we've exceeded the timeout limit before making the API request
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_limit:
            logger.warning("Timeout threshold reached. Stopping fetch early.")
            break

        url = f"https://gnews.io/api/v4/top-headlines?category={category}&lang=en&max=40&apikey={API_KEY}"
        logger.info(f"Fetching category: {category}")

        try:
            response = requests.get(url, timeout=(5, 10))  # Making the API request
            if response.status_code == 200:
                data = response.json()
                if 'articles' in data:
                    for article in data['articles']:
                        articles.append(flatten_article(article))
            else:
                logger.warning(f"Failed to fetch {category}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching news for {category}: {str(e)}")

        time.sleep(2)  # Respect API delay

    return articles


def flatten_article(article):
    return {
        "title": article.get("title"),
        "description": article.get("description"),
        "content": article.get("content"),
        "url": article.get("url"),
        "image": article.get("image"),
        "published_at": article.get("publishedAt"),
        "source_name": article.get("source", {}).get("name"),
        "source_url": article.get("source", {}).get("url"),
    }

def save_to_rds(df):
    engine_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = sqlalchemy.create_engine(engine_string, echo=False)

    try:
        logger.info(f"Inserting {df.shape[0]} rows with {df.shape[1]} columns.")
        df.to_sql(
            "news_categories",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=10,
            method="multi",
        )
        logger.info("✅ Saved to RDS successfully.")
    except Exception as e:
        logger.error(f"❌ Error saving to RDS: {e}")

def upload_to_s3(file_path):
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, S3_BUCKET, "news_categories.csv")
        logger.info("✅ Uploaded to S3 successfully.")
    except Exception as e:
        logger.error(f"❌ Error uploading to S3: {str(e)}")

def lambda_handler(event, context):
    logger.info("🚀 Lambda function started")

    articles = fetch_news()
    df = pd.DataFrame(articles)

    if df.empty:
        logger.warning("⚠️ No articles fetched.")
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No news data found."})
        }

    # Clean the data by stripping spaces and converting 'published_at' to datetime if needed
    df['title'] = df['title'].str.strip().str.lower()
    df['description'] = df['description'].str.strip().str.lower()
    df['content'] = df['content'].str.strip().str.lower()
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    # Drop rows with NaN values in important columns
    df.dropna(subset=['title', 'description', 'content', 'published_at'], inplace=True)

    # Now drop duplicates
    df.drop_duplicates(subset=['title', 'description', 'content', 'published_at'], keep='first', inplace=True)


    # Serialize complex fields
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    logger.info(f"📊 DataFrame dtypes:\n{df.dtypes}")

    try:
        save_to_rds(df)
    except Exception as e:
        logger.error(f"❌ Error during save_to_rds(): {e}")

    file_path = "/tmp/news_categories.csv"
    df.to_csv(file_path, index=False)
    upload_to_s3(file_path)

    logger.info("✅ Lambda function completed successfully")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "News data processed and saved successfully!"})
    }
