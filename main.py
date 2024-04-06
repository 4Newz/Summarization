from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from Assistant_Api.summarizer import Summarize_openAI, Summarize_Gemini
from Similarity.similarity import Similarity
from newsAPI.open_news_data import News_Fetcher
from fastapi.responses import JSONResponse
from Chirava.chirava import Scraper
import logging
import uvicorn
import heapq

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


class Source(BaseModel):
    url: str
    image: str
    heading: str


class Doc_Sentence_Map(BaseModel):
    similarity: float
    source: int


class Reference_Data(BaseModel):
    doc_sentence_map: list[Doc_Sentence_Map | None]
    sources: list[Source]


class Article_Response(BaseModel):
    summary: str
    articles: list[Article]
    reference: Reference_Data


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World from 1-news-app backend!"}


# combine all the routes into a single route one where news is fetched, then scraped and scored for similarity, with highest similar news in top and then sent to summarizer
@app.get("/kamisama_tasukete")
async def newsAI_api_v1(query: str, model: str):
    # get news from News_Fetcher
    try:
        if query:
            get_news = News_Fetcher(query, 7)
            response_newsArticles = await get_news.runner()
            logger.info("News articles retrieved successfully")
            logger.info(
                f"Number of news articles retrieved: {len(response_newsArticles)}"
            )
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
        logger.info(f"Chirava scraper response retrieved successfully")
        # print data to log
        # logger.info(f"data after chirava: {data}")
    except Exception as e:
        logger.error(f"Error scraping articles in main.py: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

    # print data to log to evaluate the responses
    # # logger.info(f"data after chirava: {json.dumps(data.dict(), indent=4)}")

    # get similarity scores

    data.news_articles = similarity_filter(data.news_articles, data.prompt)
    # data = News_Articles(prompt=query, news_articles=response_similarity)

    # get summary
    if model == "gpt3.5":
        try:
            summary = await Summarize_openAI(data.news_articles, query)
            logger.info("Summary retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return JSONResponse(status_code=500, content={"message": str(e)})
        data.summary = summary

    elif model == "gemini":
        try:
            summary = Summarize_Gemini(data.news_articles, query)
            logger.info("Summary retrieved successfully")
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return JSONResponse(status_code=500, content={"message": str(e)})
        data.summary = summary

    return data


@app.post("/summary_validation")
async def validate_summary(data: News_Articles):
    logger.info("Validating summary")
    try:
        response = await Validate(data)
        logger.info("Summary validated successfully")
        return response

    except Exception as e:
        logger.error(f"Error validating summary: {str(e)}")
        return {"error": str(e)}


async def news_fetch(query: str):
    if query:
        get_news = News_Fetcher(query, 7)
        response_newsArticles = await get_news.runner()
        logger.info("News articles retrieved successfully")
        logger.info(f"Number of news articles retrieved: {len(response_newsArticles)}")

    return News_Articles(prompt=query, news_articles=response_newsArticles)


# Sort articles by similarity and pick best N articles and return it1edc 4dws
def similarity_filter(articles: list[Article], prompt: str, N=3):
    documents = [article.content for article in articles if article.content]
    sentences = [prompt]
    similarity = Similarity.document_similarity(documents, sentences).tolist()[0]
    best_N_indices = [similarity.index(i) for i in heapq.nlargest(N, similarity)]

    best_documents = []
    for index in best_N_indices:
        best_documents.append(articles[index])
    return best_documents


async def summarize(articles: list[Article], prompt: str, model: str) -> str:
    if model == "gpt3.5":
        summary = await Summarize_openAI(articles, prompt)

    elif model == "gemini":
        summary = Summarize_Gemini(articles, prompt)

    logger.info("Summary retrieved successfully")
    return summary


# Check the similarity of each sentence in genArticle with usedArticles and map them
def get_references(summarized: str, articles: list[Article]) -> Reference_Data:
    def sparsify(arr: list[Doc_Sentence_Map]) -> list[Doc_Sentence_Map | None]:
        arr = arr[:]
        for i in range(len(arr) - 1):
            if arr[i].source == arr[i + 1].source:
                arr[i] = None  # type: ignore

        return arr  # type: ignore

    documents = [article.content for article in articles if article.content]
    sentences = summarized.split(".")
    similarity: list[list[int]] = Similarity.document_similarity(
        documents, sentences
    ).tolist()

    doc_sentence_map = [
        Doc_Sentence_Map(similarity=max(line), source=line.index(max(line)))
        for line in similarity
    ]
    print(articles)
    sources = [
        Source(heading=article.heading, image=article.urlToImage, url=article.url)
        for article in articles
    ]
    print(sources)
    return Reference_Data(doc_sentence_map=sparsify(doc_sentence_map), sources=sources)


#
@app.get("/generate_article")
async def newsAI_api_v2(query: str, model: str):
    try:
        data = await news_fetch(query)
        if len(data.news_articles) == 0:
            raise Exception("No News Found")
    except Exception as e:
        logger.error(f"Error getting news articles: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

    data.news_articles = similarity_filter(data.news_articles, query)

    try:
        summarized_article = await summarize(data.news_articles, query, model)
    except Exception as e:
        logger.error(f"Error getting summary: {str(e)}")
        return JSONResponse(status_code=500, content={"message": str(e)})

    reference = get_references(summarized_article, data.news_articles)

    response = Article_Response(
        summary=summarized_article, articles=data.news_articles, reference=reference
    )

    return response


# run the app as asynchrnous lib dosent work with normal run using uvicorn
HOST = "localhost"
PORT = 8000

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
