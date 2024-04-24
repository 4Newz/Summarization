from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util, models
import numpy
from typing import Optional
import torch

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


class BertForSTS(torch.nn.Module):
    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer("bert-base-uncased", max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)["sentence_embedding"]
        return output


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
        logger.info("Returning similarity scores")
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
    def document_similarity(documents: list[str], sentences: list[str], st_only=False):
        logger.info("Calculating document similarity")
        st = Similarity.st_similarity(documents, sentences).tolist()
        # print("ethi inside 0", st_only)
        if st_only:
            return st

        result = []
        for i, sentence in enumerate(sentences):
            result.append([])
            for j, document in enumerate(documents):
                custom = Similarity.custom_similarity(sentence, document)
                if custom:
                    result[-1].append((st[i][j] + custom) / 2)
                else:
                    result[-1].append(st[i][j])

        return result

    @staticmethod
    def custom_similarity(text1: str, text2: str):
        try:
            model = BertForSTS()
            model.load_state_dict(
                torch.load(
                    "./FineTunedBert/bert-sts.pt", map_location=torch.device("cpu")
                )
            )
            model.eval()
            tokenizer = tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            return None

        def predict_similarity(sentence_pair):
            test_input = tokenizer(
                sentence_pair,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            ).to(torch.device("cpu"))
            test_input["input_ids"] = test_input["input_ids"]
            test_input["attention_mask"] = test_input["attention_mask"]
            del test_input["token_type_ids"]
            output = model(test_input)
            sim = torch.nn.functional.cosine_similarity(
                output[0], output[1], dim=0
            ).item()
            return sim

        return predict_similarity([text1, text2])

    @staticmethod
    def st_similarity(documents: list[str], sentences: list[str]):
        model = SentenceTransformer("./MiniLM-L6-v2")

        document_embeddings = model.encode(documents, convert_to_tensor=True)
        sentence_embdeddings = model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sentence_embdeddings, document_embeddings)

        return cosine_scores
