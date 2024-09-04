from mood.metrics import Metric


class Alphabet(Metric):
    def __init__(self):
        super().__init__()
        self.state = "Positive"
        self.name = "Alphabet"

    def alphabetical_order_score(self, sequence):
        """
        Calculate a score that reflects how close the list of strings is to being in alphabetical order.
        The score is based on the number of inversions.
        """
        score = 0
        for i in range(1, len(sequence)):
            score += abs(ord(sequence[i]) - ord(sequence[i - 1]))
        return score

    def compute(self, sequences):
        import pandas as pd

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        # Get the alphabetical score for each sequence
        for sequence in sequences:
            df.loc[df["Sequence"] == sequence, "Alphabet"] = (
                self.alphabetical_order_score(sequence)
            )

        return df
