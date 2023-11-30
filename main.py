from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import Summarize


# class for the input JSON file
class News_Articles(BaseModel):
    strings: List[str]



app = FastAPI()




@app.post("/summarize/")
async def process_strings(payload: News_Articles):
    #get the list of strings from the JSON file
    articles = payload.strings
    #call the summarizer function
    summary = Summarize(articles)


    #return JSON file with the summarized text
    return {"summary": summary}






@app.get("/")
async def root():
    return {"message": "Hello World"}