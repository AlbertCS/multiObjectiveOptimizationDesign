import ast
import os

import pandas as pd

from mood.base.silent_file import Silent


class analyseGeneticAlgorithm:

    def __init__(self, folders, native_sequence_index=None):

        pass

    def setTargetState(self, state, positive_state):
        pass

    def readScores(
        self, native_sequence_index=None, output_file="msd_scores.csv", overwrite=False
    ) -> pd.DataFrame:
        pass

    def getScores(
        self,
        dataset=None,
        folders=None,
        iterations=None,
        sequences=None,
        states=None,
        copy=False,
    ) -> pd.DataFrame:
        pass

    def extractStructures(
        self,
        data,
        output_folder,
        download=False,
        host=None,
        host_folder=None,
        models_states=None,
        overwrite=False,
    ):
        pass

    def writeSequencesToFasta(self, data, output_file):
        pass

    def _filterByIndex(self, values, elements, level) -> pd.DataFrame:
        pass

    def _getMutations(self, native_sequence_index):

        pass

    def _checkInputFolders(self):
        pass


def downloadFile(host, file_path, destination):
    pass


def readScoreFile(
    score_file, rewrite_score_file=False, count_entries=False
) -> pd.DataFrame:
    pass
