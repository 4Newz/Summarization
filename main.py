from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from Assistant_Api.summarizer import Summarize
from Similarity.similarity import Similarity
from newsAPI.open_news_data import News_Fetcher
from fastapi.responses import JSONResponse
from Chirava.chirava import Scraper
import logging
import uvicorn

# Configure logging with a custom format
logger = logging.getLogger(__name__)
handler = logging.FileHandler("app.log")
# c_handler = logging.StreamHandler()
# c_handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
# c_handler.setFormatter(c_format)
# logger.addHandler(c_handler)
logger.addHandler(handler)


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


class News_Articles(BaseModel):
    prompt: str
    news_articles: List[Article]
    summary: Optional[str] = None


class Similarity_Payload(BaseModel):
    prompt: str
    news_articles: List[Article]


class Get_Article_Payload(BaseModel):
    heading: str
    articles: Optional[List[Article]] = None


app = FastAPI()


@app.post("/summarize/")
async def process_strings(payload: News_Articles):
    # get the list of strings from the JSON file
    try:
        news_articles = payload.news_articles
        # call the summarizer function
        summary = await Summarize(news_articles, payload.prompt)

        # return JSON file with the summarized text
        return {"summary": summary}
    except:
        return {"summary": "lorem ipsum"}


@app.post("/similarity")
async def get_similarity(payload: Similarity_Payload):
    max_tries = 2
    while max_tries:
        try:
            if len(payload.news_articles):
                similarity = Similarity(payload.prompt, payload.news_articles)
                output = await similarity.runner()
            response = News_Articles(prompt=payload.prompt, news_articles=output)

            return response
        except Exception as e:
            if max_tries == 0:
                res = {}
                res["error"] = str(e)
                return res
            else:
                max_tries -= 1


@app.post("/chirava")
async def get_chirava(payload: Article) -> Article:
    try:
        scraper = Scraper(payload.url)
        output = await scraper.response()
        return output
    except Exception as e:
        return {"error": str(e)}


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
        news_fetcher = News_Fetcher(query, 30)
        response = news_fetcher.get_articles_newsAPI(query, number_of_articles=30)
        list_of_articles.extend(response)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # get the articles from the NewsData API
    try:
        news_fetcher = News_Fetcher(query, 30)
        response = news_fetcher.get_articles_newsDataAPI(query)
        list_of_articles.extend(response)

    except Exception as e:
        print(f"Error getting articles: {str(e)}")
        logging.error(f"Error getting articles: {str(e)}")
        return None

    # return the combined list of articles
    news = News_Articles(prompt=query, news_articles=list_of_articles)
    return news


##############################################################################################


# combine all the routes into a single route one where news is fetched, then scraped and scored for similarity, with highest similar news in top and then sent to summarizer
@app.get("/kamisama_tasukete")
async def newsAI_api_v1(query: str, model: str) -> News_Articles:
    # get news from News_Fetcher
    try:
        if query:
            get_news = News_Fetcher(query, 30)
            response_newsArticles = await get_news.runner()
            logger.info("News articles retrieved successfully")
        else:
            logger.error("Bad Request - No query provided")
            return JSONResponse(status_code=400, content={"message": "Bad Request"})
    except Exception as e:
        logger.error(f"Error getting news articles: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

    data = News_Articles(prompt=query, news_articles=response_newsArticles)

    # get content from Chirava scraper
    try:
        scraper = Scraper(data.news_articles)
        data.news_articles = await scraper.runner()
        logger.info("Chirava scraper response retrieved successfully")
    except Exception as e:
        logger.error(f"Error scraping articles in main.py: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

    # remove the articles with None content
    for i in range(len(data.news_articles) - 1, -1, -1):
        if data.news_articles[i] is None:
            del data.news_articles[i]

    # get similarity scores
    similarity_retries = 2
    while similarity_retries:
        try:
            similarity = Similarity(query, data.news_articles)
            response_similarity = await similarity.runner()
            logger.info("Similarity scores retrieved successfully")
            break
        except Exception as e:
            logger.error(f"Error getting similarity scores: {str(e)}")
            if similarity_retries == 0:
                return JSONResponse(status_code=500, content={"message": str(e)})
            else:
                similarity_retries -= 1
    # data = News_Articles(prompt=query, news_articles=response_similarity)

    # get summary
    if model == "gpt3.5":
        try:
            summary = await Summarize(data.news_articles, query)
            logger.info("Summary retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return JSONResponse(status_code=500, content={"message": str(e)})
        data.summary = summary

        return data


# run the app as asynchrnous lib dosent work with normal run using uvicorn
HOST = "localhost"
PORT = 8000

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
