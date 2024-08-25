"""
Generate different dataformats to feed into LLM
"""

from enum import Enum
from dataclasses import dataclass
from generate_description import generate_rationale, generate_signal_analysis, generate_stats
from cincecg_torso_features import extract_features


# Define an enum for data representation choices
class DataRepresentation(Enum):
    BASELINE = 1
    WITH_RATIONALE = 2
    WITH_SIGNAL_ANALYSIS = 3
    WITH_STATS = 4


def generate_conversation(timeseries_data, answer, dataset, split, data_repr_choice, num_dimensions, raw_data):

    univariate_baseline = f"Which class is the following signal from?\n{timeseries_data}Class: "
    multivariate_baseline = f"Given the following multi-dimensional timeseries\n{timeseries_data}Which class is the above signal from\nClass: "
    multivariate_preface=""
    if num_dimensions > 1:
        multivariate_preface = "Multi-variate"

    if split == "train" or split == "validation" or split == "test" :
        if data_repr_choice == DataRepresentation.WITH_RATIONALE:
            #generate rationale
            rationale = generate_rationale(dataset, timeseries_data, answer)
            with_rationale = f"""{multivariate_preface} TimeSeries data:{timeseries_data}Rationale:{rationale}\n \
                                 Question:Which class is the above signal from? Class:"""
            return with_rationale
        elif data_repr_choice == DataRepresentation.WITH_SIGNAL_ANALYSIS:
            #generat signal analaysis
            signal_analysis = generate_signal_analysis(dataset, timeseries_data, answer )
            with_signal_analysis = f"""{multivariate_preface} TimeSeries data:{timeseries_data}Signal Analysis\
                                    :{signal_analysis}\nQuestion:Which class is the above signal from? Class:"""
            return with_signal_analysis
        elif data_repr_choice == DataRepresentation.WITH_STATS:
            #stats = generate_stats(raw_data, num_dimensions)
            feature_str = extract_features(raw_data[0])
            #with_stats = f"""{multivariate_preface} TimeSeries data:{timeseries_data}Stats:{stats}\nQuestion:Which class is the above signal from? Class:"""
            with_stats = f"{feature_str}\nQuestion:Which class is the above signal from? Class:"
            return with_stats
    if num_dimensions > 1:
        return multivariate_baseline
    else:
        return univariate_baseline
