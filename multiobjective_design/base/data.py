from dataclasses import dataclass


@dataclass
class Data:  # iteration data
    def __init__(self, sequences, folder, results_metrics) -> None:
        self.sequences = sequences
        self.folder = folder
        self.results_metrics = (
            results_metrics  # the dataframe with the index, sequences and metrics
        )
