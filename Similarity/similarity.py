from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
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


class Similarity:
    def __init__(self, prompt: str, text: list[Article]):
        self.vectorizer = TfidfVectorizer()
        self.prompt = prompt
        self.text = text
        self.similarity_indices = []

        self.model = None
        self.tokenizer = None

    async def runner(self):
        logger.info("Running similarity runner")
        async def cosine_runner():
            logger.info("Running cosine runner")
            return [self.cosine(self.prompt, text.content) for text in self.text]

        async def bert_runner():
            logger.info("Running bert runner")
            return [self.bert(self.prompt, text.content) for text in self.text]

        async def sentence_transformer_runner():
            logger.info("Running sentence transformer runner")
            return self.sentence_transformer(
                self.prompt, [text.content for text in self.text]
            )

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
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
