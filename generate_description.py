import openai
from openai import OpenAI

from stats import TimeSeriesStats
import numpy as np
import os

# Set your OpenAI API key
API_KEY=os.environ['OPENAIAPI_KEY']
openai.api_key = API_KEY

def query_openai( prompt ):

    # Make a request to the OpenAI API
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}"
                },
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0,

    )

    # Print the response
    generated_description = response.choices[0].message.content
    return generated_description

# Define the prompt to instruct the model
def generate_signal_analysis(dataset, signal_representation, answer ):

    signal_analysis_prompt = f"""
Given the following signal representation of {dataset} time series data:

{signal_representation}

Generate a pattern description of the time series, including descriptions of significant patterns or segments.

The ouput should be in a single paragraph"""
    signal_analysis = query_openai( signal_analysis_prompt )
    signal_list = signal_representation.split(',')
    signal_numerical = [float(x) for x in signal_list]
    ts_stats = TimeSeriesStats(data=np.array(signal_numerical))
    stats_prompt = f"mean:{ts_stats.mean} std_deviation:{ts_stats.std_deviation} variance:{ts_stats.variance} skewness:{ts_stats.skewness} kurtosis:{ts_stats.kurtosis} autocorrelation:{ts_stats.autocorrelation}\n"
    return stats_prompt + signal_analysis

def generate_rationale(dataset, signal_representation, answer ):
    rationale_prompt = f"""
Given the following signal representation of {dataset} time series data:

{signal_representation}

Given that the above signal belongs to Class {answer}

Generate a rationale as to why Class {answer} is the correct class

The ouput should be in a single paragraph"""
    rationale = query_openai( rationale_prompt )
    return rationale

def generate_stats( signal_representation ):
    signal_list = signal_representation.split(',')
    signal_numerical = [float(x) for x in signal_list]
    ts_stats = TimeSeriesStats(data=np.array(signal_numerical))
    stats_prompt = f"mean:{ts_stats.mean} std_deviation:{ts_stats.std_deviation} variance:{ts_stats.variance} skewness:{ts_stats.skewness} kurtosis:{ts_stats.kurtosis} autocorrelation:{ts_stats.autocorrelation}"
    return stats_prompt


if __name__ == "__main__":
    # The signal representation (input time series data)
    signal_representation = """
0.03, 0.03, 0.07, 0.08, 0.08, 0.22, 0.25, 0.29, 0.29, 0.16, 0.16, 0.14, 0.05, 0.05, 0.01, 0.03, -0.01, 0.01,
-0.03, -0.07, -0.12, -0.1, -0.09, -0.14, -0.14, -0.14, -0.14, -0.14, -0.07, -0.1, -0.65, -1.83, -2.96, -3.15,
-3.02, -2.85, -2.36, -1.96, -1.71, -1.45, -1.22, -0.92, -0.54, -0.31, -0.1, 0.05, 0.25, 0.48, 0.69, 0.77, 0.78,
0.82, 0.82, 0.88, 0.9, 0.94, 0.97, 1.03, 1.11, 1.14, 1.16, 1.2, 1.24, 1.26, 1.24, 1.18, 1.11, 1.03, 0.88, 0.75,
0.61, 0.48, 0.37, 0.25, 0.18, 0.1, 0.05, -0.03, -0.05, -0.07, -0.09, -0.1
"""
    rationale = generate_rationale("TwoLeadECG", signal_representation, "2")
    print("Rationale:", rationale)

    signal_description = generate_signal_analysis("TwoLeadECG", signal_representation, "2")
    print("Signal Description:", signal_description)

    stats = generate_stats(signal_representation)
    print("Stats:", stats)
