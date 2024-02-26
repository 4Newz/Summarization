from newspaper import Article
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging


logger = logging.getLogger(__name__)
handler = logging.FileHandler('chirava.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



class Article_Data(BaseModel):
    heading: Optional[str] = None
    content: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    urlToImage: Optional[str] = None
    Similarity: Optional[float] = None
    db_source: Optional[str] = None
    nlp_summary: Optional[str] = None


class Scraper:
    def __init__(self, data: list[Article_Data]):
        self.news_articles = data
        self.article = None

    async def runner(self):
        logger.info("Running chirava runner....")
        # Run the chirava function for each article asynchronously
        result = await asyncio.gather(
            *[self.chirava(article) for article in self.news_articles]
        )
        # return type should be List[Article]
        # remove the None values from the list due to errors
        logger.info("Chirava runner complete and returning result....")
        return [article for article in result if article is not None]


    async def chirava(self, article: Article_Data):

        try:
            logger.info(f"Scraping article: {article.url}")
            self.article = Article(article.url)
            self.article.download()
            self.article.parse()
            self.article.nlp()

            Article_Data.content = self.article.text
            Article_Data.nlp_summary = self.article.summary
            return self.article
        except Exception as e:
            logger.error(f"Error scraping article: {article.url}")
            logger.error(str(e))
            return None



    async def response(self):
        output = {
            "title": "",
            "text": "",
            "keywords": [],
            "summary": "",
            "authors": [],
            "publish_date": None,
            "top_image": "",
            "error": None
        }

        try:
            output["title"] = self.article.title
            output["text"] = self.article.text
            output["keywords"] = self.article.keywords
            output["summary"] = self.article.summary
            output["authors"] = self.article.authors
            output["publish_date"] = str(self.article.publish_date)
            output["top_image"] = self.article.top_image
        except Exception as e:
            output["error"] = str(e)

        return output
