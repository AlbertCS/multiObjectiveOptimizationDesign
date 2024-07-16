class AlgorithmDataSingleton:
    _instance = None

    """
    sequences: {"index": "sequence"}
    data_frame: pd.DataFrame(columns=["seq_index", "Sequence", "iteration", "Metric1", "Metric2", ...])
    
    """

    def __new__(cls, sequences=None, data_frame=None):
        if cls._instance is None:
            cls._instance = super(AlgorithmDataSingleton, cls).__new__(cls)
            # Initialize the instance attributes
            cls._instance.sequences = sequences
            cls._instance.data_frame = data_frame
        return cls._instance

    def update_data(self, sequences, data_frame):
        self.sequences = sequences
        self.data_frame = data_frame

    def add_sequences_form_df(self):
        self.sequences = self.data_frame["Sequence"].to_dict()

    def add_sequence(self, sequence):
        self.sequences[len(self.sequences)] = sequence

    def get_data(self):
        return self.sequences, self.data_frame
