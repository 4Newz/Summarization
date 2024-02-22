import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import os
from newsapi import NewsApiClient
from newsdataapi import NewsDataApiClient


# Configure logging with a custom format
logging.basicConfig(
    filename='newsAPI.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Custom date format
)

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
news_api_key = os.getenv("NEWS_API_KEY")
news_dataio_api_key = os.getenv("NEWS_DATAIO_API_KEY")



# Define the Article class
class Article(BaseModel):
    heading: Optional[str] = None
    content: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    urlToImage: Optional[str] = None







# Initialize the NewsApiClient
newsapi = NewsApiClient(api_key=news_api_key)

# Function to get articles from the News API
def get_articles_newsAPI(query, number_of_articles):
    # Get the articles from the News API
    try:
        response = newsapi.get_everything(
            q=query,
            language='en',
            page_size=number_of_articles,
            page=1,
            sort_by="relevancy"
        )
    except Exception as e:
        logging.error(f"Error getting articles: {str(e)}")
        return None
    logging.info(f"Articles retrieved: {response}")

    list_of_articles = []
    try:
        articles_data = response.get('articles', [])
        logging.info(f"Articles data: {articles_data}")
        for article_data in articles_data:
            article = Article(
                heading=article_data.get('title'),
                content=article_data.get('description'),
                date=article_data.get('publishedAt'),
                url=article_data.get('url'),
                source=article_data.get('source', {}).get('name'),
                urlToImage=article_data.get('urlToImage')
            )
            list_of_articles.append(article)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # Return the list of articles as dictionary
    articles = [article.dict() for article in list_of_articles]
    logging.info(f"List of articles: {articles}")
    return articles

# get_articles_newsAPI("covid", 10)








# Initialize the NewsDataApiClient
newsdataapi = NewsDataApiClient(apikey=news_dataio_api_key)

# Function to get articles from the NewsData API
def get_articles_newsDataAPI(query):
    # Get the articles from the NewsData API
    try:
        articles = newsdataapi.news_api( q= query , language= "en")
    except Exception as e:
        logging.error(f"Error getting articles: {str(e)}")
        return None
    logging.info(f"Articles retrieved: {articles}")

    list_of_articles = []
    try:
        articles_data = articles.get('results', [])
        for article_data in articles_data:
            article = Article(
                heading=article_data.get('title'),
                content=article_data.get('description'),
                date=article_data.get('pubDate'),
                url=article_data.get('link'),
                source=article_data.get('source_id'),
                urlToImage=article_data.get('image_url')
            )
            list_of_articles.append(article)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # Return the list of articles as dictionary
    articles = [article.dict() for article in list_of_articles]
    logging.info(f"List of articles: {articles}")
    return articles

# get_articles_newsDataAPI("covid")

