from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from Assistant_Api.summarizer import Summarize
from Similarity.similarity import Similarity
from newsAPI.open_news_data import get_articles_newsAPI, get_articles_newsDataAPI
from fastapi.responses import JSONResponse
from Chirava.chirava import Scraper
import logging


class Article(BaseModel):
    heading: Optional[str] = None
    content: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    urlToImage: Optional[str] = None


class News_Articles(BaseModel):
    prompt: str
    news_articles: List[Article]


class Similarity_Payload(BaseModel):
    prompt: str
    news_articles: List[Article]


class Chirava_Payload(BaseModel):
    url: str


class Chirava_Response(BaseModel):
    title: str
    text: str
    keywords: list
    summary: str
    authors: list
    publish_date: str = None
    top_image: str
    error: str = None


class Get_Article_Payload(BaseModel):
    heading: str
    articles: Optional[List[Article]] = None



app = FastAPI()


@app.post("/summarize/")
async def process_strings(payload: News_Articles):
    # get the list of strings from the JSON file
    news_articles = payload.news_articles
    # call the summarizer function
    summary = Summarize(news_articles)

    # return JSON file with the summarized text
    return {"summary": summary}


@app.post("/similarity")
async def get_similarity(payload: Similarity_Payload):
    print(payload)
    response = {}
    try:
        if len(payload.articles):
            similarity = Similarity(payload.prompt, payload.articles)
            output = await similarity.runner()
        response["data"] = output
    except Exception as e:
        response["error"] = str(e)
    print(response)
    return response


@app.post("/chirava")
async def get_chirava(payload: Chirava_Payload) -> Chirava_Response:
    response = Chirava_Response()
    try:
        if len(payload.url):
            scraper = Scraper(payload.url)
            output = await scraper.runner()
            response = Chirava_Response(**output)
    except Exception as e:
        response.error = str(e)
    return response





@app.get("/")
async def root():
    return {"message": "Hello World from 1-news-app backend!"}






# create a new get route for news fetching with return type as News_Articles
@app.get("/newsfetch")
async def get_news(query: str) -> News_Articles:
    # get the articles from the News API first and then from the NewsData API and then combine them into a single News_Articles class
    list_of_articles = []

    # get the articles from the News API
    try:
        response = get_articles_newsAPI(query, number_of_articles = 20 )
        list_of_articles.extend(response)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # get the articles from the NewsData API
    try:
        response = get_articles_newsDataAPI(query)
        list_of_articles.extend(response)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # return the combined list of articles
    news = News_Articles(prompt=query, news_articles=list_of_articles)
    return news










@app.post("/article")
async def get_article(payload: Get_Article_Payload) -> Article:
    print(payload.heading, payload.articles)
    response = Article(
        heading=payload.heading,
        content='Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry\'s standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.',
    )

    return response
