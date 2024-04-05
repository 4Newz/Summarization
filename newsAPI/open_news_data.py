import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
from newsapi import NewsApiClient
from newsdataapi import NewsDataApiClient
from urllib.parse import quote

# Configure logging with a custom format
logger = logging.getLogger(__name__)
handler = logging.FileHandler("open_news_data.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
    Similarity: Optional[float] = None
    db_source: Optional[str] = None
    nlp_summary: Optional[str] = None


class News_Fetcher:
    def __init__(self, prompt: str, number_of_articles: int):
        self.prompt = prompt
        self.number_of_articles = number_of_articles

    async def runner(self):
        async def newsAPI_runner():
            return self.get_articles_newsAPI(self.prompt, self.number_of_articles)

        async def newsDataAPI_runner():
            return self.get_articles_newsDataAPI(self.prompt)

        result = await asyncio.gather(newsAPI_runner(), newsDataAPI_runner())
        output = []
        output.extend(result[0])
        output.extend(result[1])
        return output

    # Function to get articles from the News API
    def get_articles_newsAPI(self, query: str, number_of_articles: int):
        # Initialize the NewsApiClient
        newsapi = NewsApiClient(api_key=news_api_key)

        logger.info("Inside get_articles_newsAPI function...")
        # Get the articles from the News API

        keywords = query.split(" ")
        modified_query = "+".join(keywords)
        logger.info(f"Modified query: {modified_query}")
        try:
            response = newsapi.get_everything(
                q=modified_query,
                language="en",
                page_size=number_of_articles,
                page=1,
                sort_by="relevancy",
            )
        except Exception as e:
            logger.error(f"Error getting articles: {str(e)}")
            return None
        logger.info(f"Articles retrieved: {response}")

        list_of_articles = []
        try:
            articles_data = response.get("articles", [])
            logger.info(f"Articles data: {articles_data}")
            for article_data in articles_data:
                article = Article(
                    heading=article_data.get("title"),
                    content=article_data.get("description"),
                    date=article_data.get("publishedAt"),
                    url=article_data.get("url"),
                    source=article_data.get("source", {}).get("name"),
                    urlToImage=article_data.get("urlToImage"),
                    db_source="newsAPI",
                )
                list_of_articles.append(article)

        except Exception as e:
            print(f"Error getting articles: {str(e)}")
            logger.error(f"Error getting articles: {str(e)}")
            return None

        # Return the list of articles as dictionary
        articles = [article.dict() for article in list_of_articles]
        logger.info(f"List of articles: {articles}")
        return articles

    # get_articles_newsAPI("covid", 10)

    # Function to get articles from the NewsData API
    def get_articles_newsDataAPI(self, query: str):
        # Initialize the NewsDataApiClient
        newsdataapi = NewsDataApiClient(apikey=news_dataio_api_key)

        logger.info("Inside get_articles_newsDataAPI function...")
        # Get the articles from the NewsData API
        try:
            #  modify query to have %20 between words to get more relevant results
            keywords = query.split(" ")
            modified_query = "%20".join(keywords)
            logger.info(f"Modified query: {modified_query}")
            articles = newsdataapi.news_api(q=modified_query, language="en")
        except Exception as e:
            logger.error(f"Error getting articles: {str(e)}")
            return None
        logger.info(f"Articles retrieved: {articles}")

        list_of_articles = []
        db_source = "newsDataio"
        try:
            articles_data = articles.get("results", [])
            for article_data in articles_data:
                article = Article(
                    heading=article_data.get("title"),
                    content=article_data.get("description"),
                    date=article_data.get("pubDate"),
                    url=article_data.get("link"),
                    source=article_data.get("source_id"),
                    urlToImage=article_data.get("image_url"),
                    db_source=db_source,
                )
                list_of_articles.append(article)

        except Exception as e:
            print(f"Error getting articles: {str(e)}")
            logger.error(f"Error getting articles: {str(e)}")
            return None

        # Return the list of articles as dictionary
        articles = [article.dict() for article in list_of_articles]
        logger.info(f"List of articles: {articles}")
        return articles

    # get_articles_newsDataAPI("covid")
