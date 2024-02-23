from newspaper import Article

class Scraper:
    def __init__(self, url: str):
        self.url = url
        self.article = Article(url)
        self.article.download()
        self.article.parse()
        self.article.nlp()
        # print("Scraper class created")

    async def runner(self):
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
