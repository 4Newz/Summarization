from newspaper import Article
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging


logger = logging.getLogger(__name__)
handler = logging.FileHandler("chirava.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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

    def to_dict(self):
        return {
            "heading": self.heading,
            "content": self.content,
            "date": self.date,
            "url": self.url,
            "source": self.source,
            "urlToImage": self.urlToImage,
            "Similarity": self.Similarity,
            "db_source": self.db_source,
            "nlp_summary": self.nlp_summary,
        }


class Scraper:
    def __init__(self, data: list[Article_Data]):
        self.news_articles = data
        self.article = None
        self.article_data = Article_Data()

    async def runner(self):
        logger.info("Running chirava runner....")
        # Run the chirava function for each article asynchronously
        result = await asyncio.gather(
            *[self.chirava(article) for article in self.news_articles]
        )
        # return type should be List[Article]
        # remove the None values from the list due to errors
        logger.info("Chirava runner complete and returning result....")
        return [article for article in result if article.content is not None]

    async def chirava(self, article_data: Article_Data):
        self.article_data = article_data

        try:
            logger.info(f"Scraping article: {article_data.url}")
            self.article = Article(article_data.url)
            self.article.download()
            self.article.parse()
            self.article.nlp()

            # update the article object with the new data
            # resultant_article = Article_Data(
            #     heading=article_data.heading,
            #     content=self.article.text,
            #     date=article_data.date,
            #     url=article_data.url,
            #     source=article_data.source,
            #     urlToImage=article_data.urlToImage,
            #     Similarity=article_data.Similarity,
            #     db_source=article_data.db_source,
            #     nlp_summary=self.article.summary
            # )

            article_data.content = self.article.text
            article_data.nlp_summary = self.article.summary

            # logger.info(f"Scraped article type: {type(resultant_article)}")
            # logger.info(f"return article: {resultant_article}")
            return article_data
            # return resultant_article
        except Exception as e:
            logger.error(f"Error scraping article: {article_data.url}")
            logger.error(str(e))
            article_data.content = article_data.content[:-4]
            return article_data

    async def response(self):
        output = {
            "title": "",
            "text": "",
            "keywords": [],
            "summary": "",
            "authors": [],
            "publish_date": None,
            "top_image": "",
            "error": None,
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
