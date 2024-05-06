import logging
import json
from Assistant_Api.object_mapper import (
    serialize_assistant,
    serialize_chat_thread,
    serialize_thread_message,
    serialize_run,
)
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# Configure logging with a custom format
logger = logging.getLogger(__name__)
handler = logging.FileHandler("openAI.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI()
genai.configure(api_key=google_api_key)


async def Summarize_openAI(articles, prompt):
    logger.info(f"Summarizing articles on the topic: {prompt} using OpenAI")

    # Create assistant and log the response using object mapper serialization
    my_assistant = client.beta.assistants.create(
        instructions=f"Use the below articles to form a structured ordered story or news on the topic {prompt}. The summary should be concise and coherent to the reader. only contain things that are relevant to the topic. minimun 600 words",
        name="NewsAI",
        model="gpt-3.5-turbo-1106",
    )
    logger.info(f"Assistant created: {serialize_assistant(my_assistant)}")

    # conversation thread and log the response using object mapper serialization
    chat_thread = client.beta.threads.create()
    logger.info(f"Chat thread created: {serialize_chat_thread(chat_thread)}")

    # Sending messages to the conversation thread
    for i in range(len(articles)):
        # Format the content
        if articles[i].Similarity is None or articles[i].Similarity >= 0.5:
            # check if the token count of the content is more than 1000
            if len(articles[i].content.split()) > 1000:
                logger.info(f"Article {i} is too long, using the NLP summary instead")
                content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].nlp_summary}"
            else:
                logger.info(
                    f"Article {i} is within the token limit directly using the content"
                )
                content = (
                    f"{articles[i].date} - {articles[i].heading}\n{articles[i].content}"
                )

            thread_message = client.beta.threads.messages.create(
                thread_id=chat_thread.id,
                role="user",
                content=content,
            )
            # logger.info(f"Thread message sent: {serialize_thread_message(thread_message)}")
            logger.info(f"Thread message sent: {thread_message}")

        else:
            continue

    # run the assistant and log the response using object mapper serialization
    run = client.beta.threads.runs.create(
        thread_id=chat_thread.id, assistant_id=my_assistant.id
    )
    logger.info(f"Run created: {serialize_run(run)}")

    # Polling loop to check the status of the run
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=chat_thread.id, run_id=run.id)
        status = run.status
        logger.info(f"Run status: {status}")

        if status in ["completed", "failed", "cancelled", "expired"]:
            break

        time.sleep(2)  # Wait for 5 seconds before checking again

    # Retrieve the messages from the conversation thread
    messages = client.beta.threads.messages.list(
        thread_id=chat_thread.id,
    )
    # print(messages.data)
    logger.info(f"Messages retrieved: {serialize_thread_message(messages.data)}")

    # Retrieve the last message from the conversation thread containing the summarized text
    print(messages)
    summarized_text = messages.data[0].content[0].text.value

    # Delete the conversation thread once it's done
    response = client.beta.threads.delete(chat_thread.id)
    logger.info(f"Thread deleted: {response}")

    # Delete the assistant once it's done
    response = client.beta.assistants.delete(my_assistant.id)
    logger.info(f"Assistant deleted: {response}")

    logger.info(f"Summarized text: {summarized_text}")
    return summarized_text.split(" - ")[1]




async def Summarize_Gemini(articles, prompt):
    logger.info(f"Summarizing articles on the topic: {prompt} using Gemini")

    model = genai.GenerativeModel("gemini-1.0-pro")

    query = f"Use the below articles to form a structured ordered story or news on the topic {prompt}. The generated article/story should be detailed to the reader, also only contain things that are relevant to the topic. minimun 600 words \n"

    for i in range(len(articles)):
        # Format the content
        if articles[i].Similarity is None or articles[i].Similarity >= 0:
            # check if the token count of the content is more than 2000
            if len(articles[i].content) == 0:
                logger.info(f"Article {i} is empty skipping it")
                continue
            if model.count_tokens(articles[i].content).total_tokens > 2000:
                logger.info(f"Article {i} is too long, using the NLP summary instead")
                content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].nlp_summary}"
            else:
                logger.info(
                    f"Article {i} is within the token limit directly using the content"
                )
                content = (
                    f"{articles[i].date} - {articles[i].heading}\n{articles[i].content}"
                )
            query += content + "\n"

            logger.info(f"Article {i} added to the query")

        else:
            logger.info(f"Article {i} skipped due to low similarity score")
            continue
    print(query)
    response = model.generate_content(query)
    logger.info("Article generated successfully and returned to the user")
    return response.text





# create a function to sent a question and a paragraph to gemini and get the answer. the paragraph should be the context of the question and the question should be the question to be answered
async def ask_question(question: str, paragraph: str):
    model = genai.GenerativeModel("gemini-1.0-pro")

    prompt = "I have a question about the following topic in context. Can you help me with the answer? only answer if you are sure of the answer otherwise let me no if ur not sure\n"
    prompt += f" Question: {question}\n"

    # check if the token count of the content is more than 5000
    if model.count_tokens(paragraph).total_tokens > 5000:
        logger.info(f"Paragraph is too long")
        summary = model.generate_content(f"summarize: {paragraph}")
        prompt += f" Context: {summary.text}\n"

        # wait for 5 seconds before sending the question
        time.sleep(5)
    else:
        prompt += f" Context: {paragraph}\n"

    response = model.generate_content(prompt)
    return response.text

