from newspaper import Article

class Scraper:
    def __init__(self, url: str):
        self.url = url
        self.article = Article(url)
        self.article.download()
        self.article.parse()
        self.article.nlp()
    async def runner(self):
        return {
            "title": self.article.title,
            "text": self.article.text,
            "keywords": self.article.keywords,
            "summary": self.article.summary,
            "authors": self.article.authors,
            "publish_date": self.article.publish_date,
            "top_image": self.article.top_image,
        }