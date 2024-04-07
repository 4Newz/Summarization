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
        instructions = f"Use the below articles to form a structured ordered story or news on the topic {prompt}. The summary should be concise and coherent to the reader. only contain things that are relevant to the topic. minimun 600 words",
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
                logger.info(f"Article {i} is within the token limit directly using the content")
                content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].content}"

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
    summarized_text = messages.data[0].content[0].text.value

    # Delete the conversation thread once it's done
    response = client.beta.threads.delete(chat_thread.id)
    logger.info(f"Thread deleted: {response}")

    # Delete the assistant once it's done
    response = client.beta.assistants.delete(my_assistant.id)
    logger.info(f"Assistant deleted: {response}")

    logger.info(f"Summarized text: {summarized_text}")
    return summarized_text












def Summarize_Gemini(articles, prompt):

    logger.info(f"Summarizing articles on the topic: {prompt} using Gemini")

    model = genai.GenerativeModel("gemini-1.0-pro")

    query = f"Use the below articles to form a structured ordered story or news on the topic {prompt}. The generated article/story should be detailed to the reader, also only contain things that are relevant to the topic. minimun 600 words \n"

    for i in range(len(articles)):
        # Format the content
        if articles[i].Similarity is None or articles[i].Similarity >= 0.5:

            # check if the token count of the content is more than 2000
            if model.count_tokens(articles[i].content).total_tokens > 2000:
                logger.info(f"Article {i} is too long, using the NLP summary instead")
                content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].nlp_summary}"
            else:
                logger.info(f"Article {i} is within the token limit directly using the content")
                content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].content}"
            query += content + "\n"

            logger.info(f"Article {i} added to the query")

        else:
            logger.info(f"Article {i} skipped due to low similarity score")
            continue

    response = model.generate_content(query)
    logger.info("Article generated successfully and returned to the user")
    return response.text























































# async def Validate(data):
#     dict_articles = {}
#     # create individual embeddings for each article and store them
#     for article in data.news_articles:
#         response = client.embeddings.create(
#             input=article.content,
#             model="text-embedding-3-small"
#         )
#         dict_articles[article.url] = response.data[0].embedding

#     # Split the summary into sentences
#     summary_sentences = data.summary.split('. ')

#     # Create embeddings for each sentence in the summary
#     sentence_embeddings = []
#     for sentence in summary_sentences:
#         response = client.embeddings.create(
#             input=sentence,
#             model="text-embedding-3-small"
#         )
#         sentence_embeddings.append(response.data[0].embedding)

#     # Calculate the cosine similarity between each sentence embedding and each article embedding
#     results = []
#     for sentence, sentence_embedding in zip(summary_sentences, sentence_embeddings):
#         similarities = []
#         for article, article_embedding in dict_articles.items():
#             similarity = cosine_similarity([sentence_embedding], [article_embedding])
#             similarities.append((article, similarity))

#         # Find the article with the highest similarity score
#         most_similar_article, max_similarity = max(similarities, key=lambda x: x[1])

#         # Extract the heading and URL from the most similar article
#         article_info = {"heading": most_similar_article.heading, "url": most_similar_article.url}

#         # Add the sentence, the article info, and the similarity score to the results
#         results.append({"sentence": sentence, "article": article_info, "score": max_similarity[0][0]})

#     return results



