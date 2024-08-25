"""
Module that analyzes a bunch of text,
and returns the right downsampling parameter to use
"""

import re
from utils import format_numbers_combined, generate_question
from data_templates import DataRepresentation
from analyze_tokens import analyze_token_length
from aeon.datasets import load_classification

def compute_downsample_setting_new( raw_data, target, round_to, dataset, split, data_repr, model_name, context_length ):
    downsample = None
    question = generate_question( raw_data, target, None, round_to, dataset, split, data_repr )
    full_token_length, _ = analyze_token_length(question, model_name)
    pattern = r'\[\s*([^\]]+)\s*\]'
    match = re.search(pattern, question)
    if match:
        # Extract the content within the brackets
        timeseries_signal =  match.group(0)
    else:
        raise IndexError( "Unable to find timeseries portion of the string" )
    timeseries_token_length, _ = analyze_token_length( timeseries_signal, model_name )
    extra_token_length = full_token_length - timeseries_token_length
    #Add some epsilion num_tokens
    extra_token_length += 5
    if full_token_length < context_length:
        pass
    else:
        downsample = round( full_token_length / ( context_length - extra_token_length ) )
    print(f"Setting downsample to:{downsample} for dataset:{dataset}")
    return downsample


def compute_downsample_setting( X, model_name, context_length, round_to, extra_text_token_length, data_repr ):

    #Every training sample is enhanced with the following extra text
    #\n X num_dimensions
    #[] X num_dimensions
    #Dimension {dimension}: if X is multi dimensional
    #univariate_baseline : "Which class is the following signal from?\n\nClass: "
    #multivariate_baseline: "Given the following multi-dimensional timeseries\n\nWhich class is the above signal from\nClass: "
    #multivariate_preface = "Multi-variate "
    #with_rationale = " TimeSeries data:\nRational:\n Question:Which class is the above signal from? Class:"
    #with_signal_analysis = " TimeSeries data:\nSignal Analysis:\nQuestion:Which class is the above signal from? Class:"

    num_dimensions = len(X)
    text = ""
    for dimension in range(num_dimensions):
        text += f"{format_numbers_combined(X[dimension], round_to=round_to)}"

    is_multi_variate = num_dimensions > 1
    extra_text = "\n" * num_dimensions
    extra_text += "[]" * num_dimensions
    if is_multi_variate:
        for i in range(num_dimensions):
            extra_text += "Dimension {dimension}:"
    if data_repr == DataRepresentation.BASELINE:
        if is_multi_variate:
            extra_text += "Given the following multi-dimensional timeseries\n\nWhich class is the above signal from\nClass: "
        else:
            extra_text += "Which class is the following signal from?\n\nClass: "
    elif data_repr == DataRepresentation.WITH_RATIONALE:
        if is_multi_variate:
            extra_text += "Multi-variate "
        extra_text += "TimeSeries data:\nRationale:\n Question:Which class is the above signal from? Class:"
    elif data_repr == DataRepresentation.WITH_SIGNAL_ANALYSIS:
        if is_multi_variate:
            extra_text += "Multi-variate "
        extra_text += "TimeSeries data:{timeseries_data}\nSignal Analysis:\nQuestion:Which class is the above signal from? Class:"
    downsample = None
    token_length, _ = analyze_token_length( text, model_name )
    #print( f"Text:{text} Token Length:{token_length}" )
    residual_token_length, _ = analyze_token_length( extra_text, model_name )
    #print( f"ExtraText:{extra_text} ResidualTokenLength:{residual_token_length}" )
    extra_text_token_length += residual_token_length
    if token_length < context_length:
        pass
    else:
        #TODO : can do better here, should we downsample for every sample differently?
        downsample = round( token_length / ( context_length - extra_text_token_length ) )
    return downsample

def compute_downsample( dataset, context_length, data_repr, extra_len=0 ):
    X, y = load_classification( dataset )
    downsample = compute_downsample_setting( X[0], "liuhaotian/llava-v1.5-7b", context_length, 2, extra_len, data_repr )
    print(f"Downsampling factor for {dataset}:{downsample}")

if __name__ == "__main__":
    for context_length in [2048, 4096]:
        for data_repr in [DataRepresentation.BASELINE, DataRepresentation.WITH_RATIONALE, DataRepresentation.WITH_SIGNAL_ANALYSIS]:
            print( context_length, data_repr )
            for dataset in ["TwoLeadECG", "CinCECGTorso", "PenDigits", "ItalyPowerDemand", "FreezerSmallTrain", "PhalangesOutlinesCorrect", "HandOutlines"]:
                compute_downsample( dataset, context_length, data_repr, extra_len=0 if data_repr==DataRepresentation.BASELINE else 300 )
