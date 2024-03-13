import os

import pandas as pd

# TODO may have to remove the relative import
from .metric import Metric


# TODO future testing of this metric
class Netsolp(Metric):
    """
    NetSolp metric.

    This metric is used to evaluate the performance of the solution in terms of
    the number of solutions found by the algorithm. The metric is calculated as
    the number of solutions found by the algorithm divided by the number of
    solutions in the Pareto front.

    Parameters
    ----------

    """

    def __init__(
        self,
        iteration_folder,
        chain,
        sequences,
        netsolp_path,
        model="ESM1b",
        prediction="SU",
    ):
        super().__init__(iteration_folder)

        models = ["ESM12", "ESM1b", "Distilled", "Both"]
        if model not in models:
            raise ValueError(
                f"The given model type is not recognised. It should be {str(models)}"
            )

        predictions = ["S", "U", "SU"]
        if prediction not in predictions:
            raise ValueError(
                f"The given prediction type is not recognised. It should be {str(predictions)}"
            )

        self.chain = chain
        self.sequences = sequences
        self.job_folder = iteration_folder + "/netsolp"
        self.netsolp_path = netsolp_path
        self.netsolp_script = netsolp_path + "/predict.py"
        self.models_path = netsolp_path + "/models"
        self.model = model
        self.prediction = prediction

        self.input_folder = None
        self.output_folder = None
        self.output_csv = None

    def _write_fasta_file(self, sequences, output_file):
        """
        Write sequences to a fasta file.

        Parameters
        ----------
        sequences : dict
            Dictionary containing as values the strings representing the sequences
            of the proteins to align and their identifiers as keys.

        output_file : str
            Path to the output fasta file
        """

        # Write fasta file containing the sequences
        with open(output_file, "w") as of:  # pylint: disable=unspecified-encoding
            for name in sequences:
                of.write(">" + name + "\n")
                of.write(sequences[name] + "\n")

    def _setup_iterations_inputs(self):
        """
        Set up the inputs for the netsolp calculation.
        """

        # Create job folders
        if not os.path.exists(self.job_folder):
            os.mkdir(self.job_folder)

        # Create input folder
        self.input_folder = self.job_folder + "/input_files"
        if not os.path.exists(self.input_folder):
            os.mkdir(self.input_folder)

        # Create output folder
        self.output_folder = self.job_folder + "/output_files"
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)

        self.output_csv = f"{self.output_folder}/{self.chain}_prediction.csv"

        # Generate input files
        self._write_fasta_file(
            self.sequences,
            self.input_folder + "/" + self.chain + "_" + "sequences.fasta",
        )

        # Create execution script
        command = (
            f"python {self.netsolp_script} --FASTA_PATH {self.input_folder}/{self.chain}_sequences.fasta "
            f"--OUTPUT_PATH {self.output_csv} --MODEL_TYPE {self.model} --PREDICTION_TYPE {self.prediction} --MODELS_PATH {self.models_path}\n"
        )
        with open(
            f"{self.input_folder}/predict.sh", "w"
        ) as bash:  # pylint: disable=unspecified-encoding
            bash.write(command)

    def compute(self):
        """
        Compute NetsolP predictions.
        """
        # Setup iterations
        self._setup_iterations_inputs()

        # Run NetsolP
        command = "bash " + self.input_folder + "/predict.sh\n"
        command += "cd ../\n"  # Check that it correctly goes back
        os.system(command)

        # TODO should be returning a column to add to the dataclass, with the prediction values
        pred = self.read_results()

    def read_results(self):
        """
        Read NetsolP predictions.
        """
        predictions = {}
        predictions["Chain"] = []
        predictions["Index"] = []

        if not os.path.exists(self.output_csv):
            raise ValueError("Something went wrong! NetsolP calculation failed.")

        # Read prediction data
        data = pd.read_csv(self.output_csv)

        # Store predicted values
        predictions["Chain"] += [self.chain] * data.shape[0]
        predictions["Index"] += data["sid"].tolist()

        if "predicted_solubility" in data:
            predictions.setdefault("Solubility", [])
            predictions["Solubility"] += data["predicted_solubility"].tolist()
        if "predicted_usability" in data:
            predictions.setdefault("Usability", [])
            predictions["Usability"] += data["predicted_usability"].tolist()

        # Convert to dataframe
        predictions = pd.DataFrame(predictions).set_index(["Chain", "Index"])
        predictions.to_csv(self.output_csv)

        return predictions
