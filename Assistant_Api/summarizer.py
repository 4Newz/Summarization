import logging
import json
from Assistant_Api.object_mapper import serialize_assistant, serialize_chat_thread, serialize_thread_message, serialize_run
from openai import OpenAI
from dotenv import load_dotenv
import os
import time



# Configure logging with a custom format
logger = logging.getLogger(__name__)
handler = logging.FileHandler('openAI.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()





async def Summarize(articles, prompt):

    # Create assistant and log the response using object mapper serialization
    my_assistant = client.beta.assistants.create(
        instructions = f"You are a professional journalist for a news paper. Summarize the given articles to form a structured chronological ordered story or news on the topic {prompt} without loss of information.",
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
        content = f"{articles[i].date} - {articles[i].heading}\n{articles[i].content}"

        thread_message = client.beta.threads.messages.create(
            thread_id=chat_thread.id,
            role="user",
            content=content,
        )
        # logger.info(f"Thread message sent: {serialize_thread_message(thread_message)}")
        logger.info(f"Thread message sent: {thread_message}")



    # run the assistant and log the response using object mapper serialization 
    run = client.beta.threads.runs.create(
        thread_id=chat_thread.id,
        assistant_id=my_assistant.id
    )
    logger.info(f"Run created: {serialize_run(run)}")



    # Polling loop to check the status of the run
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=chat_thread.id,
            run_id=run.id
        )
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





