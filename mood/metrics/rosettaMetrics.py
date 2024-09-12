from mood.metrics import Metric
from mood.metrics.pyrosetta_functions import PyrosettaFunctions


class RosettaMetrics(Metric):
    def __init__(self, params_folder, pdb):
        super().__init__()
        self.state = "Positive"
        self.name = "rosetta"
        self.params_folder = params_folder
        self.pdb = pdb

    # TODO see if these makes sense as a multithread, if not use the xml file to call the rosetta scritps
    def compute(self, sequences):
        import pandas as pd

        # Create df
        df = pd.DataFrame(sequences, columns=["Sequence"])

        pyrosetta_func = PyrosettaFunctions(
            params_folder=self.params_folder, pdb=self.pdb
        )

        # Get the alphabetical score for each sequence
        for sequence in sequences:

            pose = pyrosetta_func.mutate_native_pose(
                pose=pyrosetta_func.native_pose.copy(), sequence=sequence
            )

            total_energy = pyrosetta_func.relax(pose=pose)
            df.loc[df["Sequence"] == sequence, f"{self.name}_total_energy"] = (
                total_energy
            )
            interface_score = pyrosetta_func.calculate_Interface_Score(pose=pose)
            df.loc[df["Sequence"] == sequence, f"{self.name}_interface_score"] = (
                interface_score
            )
            # TODO think about the distance function
            distances = pyrosetta_func.distance()
            for dist in distances:
                df.loc[df["Sequence"] == sequence, f"{self.name}_distance_{dist}"] = (
                    distances[dist]
                )

        return df
