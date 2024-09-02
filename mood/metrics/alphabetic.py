from mood.metrics import Metric


class Alphabet(Metric):
    def __init__(self):
        super().__init__()

    def alphabetical_order_score(self, sequence):
        """
        Calculate a score that reflects how close the list of strings is to being in alphabetical order.
        The score is based on the number of inversions.
        """
        import jellyfish

        sorted_sequence = sorted(sequence)
        score = jellyfish.jaro_similarity(sequence, sorted_sequence)

        return score

    def compute(self, sequences):
        import pandas as pd

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        # Get the alphabetical score for each sequence
        for sequence in sequences:
            df.loc[df["Sequence"] == sequence, "Alphabetical Score"] = (
                self.alphabetical_order_score(sequence)
            )

        return df
