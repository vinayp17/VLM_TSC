import openai
from openai import OpenAI

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = API_KEY

def query_openai(dataset, signal_str, answer, max_tokens=150):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=API_KEY,
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"You are given a question, timeseries data for {dataset}, and an answer to the question. Your task is to justify why the answer to the question is the one provided .. \
                Question : Which class does the following signal belong to. ? \
                Timeseries data for {dataset} : {signal_str} \
                Answer : {answer}",
            },
         ],
         max_tokens=max_tokens,
    )
    return(completion.choices[0].message.content)
