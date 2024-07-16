from metrics.metric import Metric


class Alphabet(Metric):
    def __init__(self, iteration_folder, data=None):
        super().__init__(iteration_folder, data)

    def compute(self, strings):
        # Using sorted() function to get a new sorted list
        sorted_strings = sorted(strings)

        # Using sort() method to sort the list in place
        strings.sort()
