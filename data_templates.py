"""
Generate different dataformats to feed into LLM
"""

from enum import Enum
from dataclasses import dataclass


# Define an enum for data representation choices
class DataRepresentation(Enum):
    BASELINE = 1
    WITH_RATIONALE = 2
    WITH_SIGNAL_ANALYSIS = 3


def generate_conversation(timeseries_data, answer, dataset, split, data_repr_choice, num_dimensions):

    univariate_baseline = f"Which class is the following signal from?\n{timeseries_data}\nClass: "
    multivariate_baseline = f"Given the following multi-dimensional timeseries\n{timeseries_data}\nWhich class is the above signal from\nClass: "
    multivariate_preface=""
    if num_dimensions > 1:
        multivariate_preface = "Multi-variate"

    if split == "train" or split == "validation":
        if data_repr_choice == DataRepresentation.WITH_RATIONALE:
            #generate rationale
            rationale = generate_rationale(dataset, timeseries_data, answer)
            with_rationale = f"""{multivariate_preface} TimeSeries data:{timeseries_data}\nRationale:{rationale}\n \
                                 Question:Which class is the above signal from? Class:"""
            return with_rationale
        elif data_repr_choice == DataRepresentation.WITH_SIGNAL_ANALYSIS:
            #generat signal analaysis
            signal_analysis = generate_signal_analysis(dataset, timeseries_data, answer )
            with_signal_analysis = f"""{multivariate_preface} TimeSeries data:{timeseries_data}\nSignal Analysis\
                                    :{signal_analysis}\nQuestion:Which class is the above signal from? Class:"""
            return with_signal_analysis
    if num_dimensions > 1:
        return multivariate_baseline
    else:
        return univariate_baseline
