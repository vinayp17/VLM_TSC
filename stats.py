import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from dataclasses import dataclass, field

@dataclass
class TimeSeriesStats:
    data: np.ndarray
    mean: float = field(init=False)
    std_deviation: float = field(init=False)
    variance: float = field(init=False)
    skewness: float = field(init=False)
    kurtosis: float = field(init=False)
    autocorrelation: float = field(init=False)
    rounding: int = 2

    def __post_init__(self):
        self.mean = round(np.mean(self.data), self.rounding)
        self.std_deviation = round(np.std(self.data), self.rounding)
        self.variance = round(np.var(self.data), self.rounding)
        self.skewness = round(skew(self.data), self.rounding)
        self.kurtosis = round(kurtosis(self.data), self.rounding )
        self.autocorrelation = round(pd.Series(self.data).autocorr(), self.rounding)

if __name__ == "__main__":
    # Provided time series data
    time_series_data = [
        0.03, 0.03, 0.07, 0.08, 0.08, 0.22, 0.25, 0.29, 0.29, 0.16, 0.16, 0.14, 0.05, 0.05, 0.01, 0.03, -0.01, 0.01,
        -0.03, -0.07, -0.12, -0.1, -0.09, -0.14, -0.14, -0.14, -0.14, -0.14, -0.07, -0.1, -0.65, -1.83, -2.96, -3.15,
        -3.02, -2.85, -2.36, -1.96, -1.71, -1.45, -1.22, -0.92, -0.54, -0.31, -0.1, 0.05, 0.25, 0.48, 0.69, 0.77, 0.78,
        0.82, 0.82, 0.88, 0.9, 0.94, 0.97, 1.03, 1.11, 1.14, 1.16, 1.2, 1.24, 1.26, 1.24, 1.18, 1.11, 1.03, 0.88, 0.75,
        0.61, 0.48, 0.37, 0.25, 0.18, 0.1, 0.05, -0.03, -0.05, -0.07, -0.09, -0.1
    ]

    # Create an instance of the TimeSeriesStats dataclass
    time_series_stats = TimeSeriesStats(data=np.array(time_series_data))

    # Print the calculated statistical properties
    print(f"Mean: {time_series_stats.mean}")
    print(f"Standard Deviation: {time_series_stats.std_deviation}")
    print(f"Variance: {time_series_stats.variance}")
    print(f"Skewness: {time_series_stats.skewness}")
    print(f"Kurtosis: {time_series_stats.kurtosis}")
    print(f"Autocorrelation: {time_series_stats.autocorrelation}")

