from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util
import numpy
from typing import Optional

from dotenv import load_dotenv
import asyncio
import requests
import json
import os
import statistics
import logging


# Configure logging with a custom format
logger = logging.getLogger(__name__)
handler = logging.FileHandler("similarity.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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


class Similarity:
    def __init__(self, prompt: str, text: list[Article]):
        self.vectorizer = TfidfVectorizer()
        self.prompt = prompt
        self.text = text
        self.similarity_indices = []

        self.model = None
        self.tokenizer = None

    async def runner(self):
        async def cosine_runner():
            result = [self.cosine(self.prompt, text.content) for text in self.text]
            logger.info("Executed cosine runner")
            return result

        async def bert_runner():
            result = [self.bert(self.prompt, text.content) for text in self.text]
            logger.info("Executed bert runner")
            return result

        async def sentence_transformer_runner():
            result = self.sentence_transformer(
                self.prompt, [text.content for text in self.text]
            )
            logger.info("Executed sentence transformer runner")
            return result

        result = await asyncio.gather(
            cosine_runner(), bert_runner(), sentence_transformer_runner()
        )

        # combine the scores from the different similarity methods using weights and store the result in the Similarity field
        weights = [0.2, 0.4, 0.4]

        for i in range(len(self.text)):
            self.text[i].Similarity = sum(
                [result[j][i] * weights[j] for j in range(len(result))]
            )
        logger.info(f"Returning similarity scores: {self.text} ")
        return self.text

        # output = []
        # for i in range(len(self.text)):
        #     output.append(
        #         {
        #             "cosine": result[0][i],
        #             "bert": result[1][i],
        #             "hf": result[2][i],

        # return output

    def cosine(self, text1: str, text2: str):
        vectors = self.vectorizer.fit_transform([text1, text2])

        # Calculate the cosine similarity between the vectors
        similarity = cosine_similarity(vectors)
        return statistics.mean([similarity[0][-1], similarity[-1][0]])

    def bert(self, text1: str, text2: str):
        # Load the BERT model
        if not (self.model and self.tokenizer):
            self.model = BertModel.from_pretrained("./BertModel/")
            # self.model.save_pretrained()
            self.tokenizer = BertTokenizer.from_pretrained("./BertTokenizer/")
            # self.tokenizer.save_pretrained()
        model = self.model
        tokenizer = self.tokenizer

        encoding1 = tokenizer.encode(
            text1,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoding2 = tokenizer.encode(
            text2,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        output1 = model(encoding1)
        output2 = model(encoding2)

        embeddings1 = output1.last_hidden_state[:, 0, :].detach().numpy().flatten()
        embeddings2 = output2.last_hidden_state[:, 0, :].detach().numpy().flatten()
        # Calculate the cosine similarity between the embeddings
        similarity = numpy.dot(embeddings1, embeddings2) / (
            numpy.linalg.norm(embeddings1) * numpy.linalg.norm(embeddings2)
        )

        return float(similarity)

    def sentence_transformer(self, prompt: str, text: list[str]):
        load_dotenv()
        url = "https://api-inference.huggingface.co/models/DrishtiSharma/sentence-t5-large-quora-text-similarity"
        huggin_face_key = os.getenv("HUGGING_FACE_API_KEY")
        headers = {
            "Authorization": f"Bearer {huggin_face_key}",
            "Content-Type": "application/json",
        }

        data = {
            "inputs": {
                "source_sentence": prompt,
                "sentences": text,
            }
        }
        response = requests.post(url, data=json.dumps(data), headers=headers)
        # print(response.json())
        return response.json()

    @staticmethod
    def document_similarity(documents: list[str], sentences: list[str]):
        model = SentenceTransformer("./MiniLM-L6-v2")

        document_embeddings = model.encode(documents, convert_to_tensor=True)
        sentence_embdeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sentence_embdeddings, document_embeddings)

        return cosine_scores


if __name__ == "__main__":
    documents = [
        "<b>NEW DELHI: </b>The US has not asked India to reduce its imports of Russian oil and the price cap and sanctions regime imposed by the G7 are aimed at squeezing Moscow’s profits from crude sales and impeding its ability to finance the war in Ukraine, two US officials said on Thursday. The US officials, speaking after meetings to brief their Indian counterparts on the second phase of implementing the price cap, which came into effect in December 2022, said the focus will now be on channels created by Russia to export crude without using Western service providers for shipping or insurance. US assistant secretary for economic policy Eric Van Nostrand and acting assistant secretary for terrorist financing Anna Morris were asked during an interaction at Ananta Centre if there had been any fresh demand for India to reduce Russian oil imports. ",
        "A 21-year-old woman is being hailed for her heroic efforts to save a dog who was stuck in a burning building. Upon witnessing the flames, the woman fearlessly rushed into the house and managed to release the canine from what could have potentially been a life-threatening situation.The Instagram page Good News Movement shared about this incident. In the caption of the post, they informed, The woman @raenahh ran into a burning building to save a neighbour's dog. @sashamerci says the woman risked her life, running in all shaky-voiced to save Bubba, her sister's dog who likely wouldn't have otherwise made it out. Not all heroes wear capes! Dogs understand the meaning of some words like humans, create mental images: Study</a>)",
        "A total solar eclipse is set to occur on April 8. That is the date when our little Moon will shade the gigantic Sun by coming directly in front of it. The event will virtually turn the day into night for the length of the eclipse period. Not just that, the absence of sunlight will also make temperatures fall. The event is spectacular because of the totality, which occurs rarely. However, it will not be visible across the globe. It is limited to regions across the US, Mexico and more. These regions only will be covered in the path of totality.The magic that will happen then is the revealing of the Sun’s corona, which is never otherwise visible because of the brightness of the Sun. Needless to say, the event is unique and catching one during a lifetime is quite rare.",
    ]
    sentences = [
        "US has not asked India to reduce Russian oil imports: Officials",
        "Solar Eclipse 2024: Moon set to shade the Sun; know all about this awesome spectacle",
        "Woman, 21, runs into a burning building to save a caged dog. Watch viral video",
    ]
    print(Similarity.document_similarity(documents, sentences))
