from metric import Metric


class Netsolp(Metric):
    """NetSolp metric.

    This metric is used to evaluate the performance of the solution in terms of
    the number of solutions found by the algorithm. The metric is calculated as
    the number of solutions found by the algorithm divided by the number of
    solutions in the Pareto front.

    Parameters
    ----------
    pareto_front : numpy.ndarray
        Pareto front.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        print(f"Netsolp metric computed. {self.data}")
