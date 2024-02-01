from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from Assistant_Api.summarizer import Summarize
from Similarity.similarity import Similarity
from fastapi.responses import JSONResponse
from Chirava.chirava import Scraper

from dotenv import load_dotenv
import json


# class for the input JSON file
class News_Articles(BaseModel):
    news_articles: List[Article]


class Article(BaseModel):
    heading: str
    content: str


class Similarity_Payload(BaseModel):
    prompt: str
    articles: List[Article]

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

app = FastAPI()


@app.post("/summarize/")
async def process_strings(payload: News_Articles):
    # get the list of strings from the JSON file
    news_articles = payload.articles
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
