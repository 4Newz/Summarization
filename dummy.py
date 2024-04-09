from openai import OpenAI

client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
)

response = client.create_completion(
    model="llama2",
    prompt="Who won the world series in 2020? The LA Dodgers won in 2020. Where was it played?",
    max_tokens=60
)

print(response.choices[0].text.strip())