import os
import shutil

from mood.base.data import AlgorithmDataSingleton
from mood.metrics.alphabetic import Alphabet
from mood.metrics.rosetta_extras import XmlProtocols
from mood.multiObjectiveOptimization import MultiObjectiveOptimization


class mood_parser:

    def main(protocol):
        data = AlgorithmDataSingleton()
        optimizer = "genetic_algorithm"
        metrics = [Alphabet()]
        debug = True
        max_iteration = 5
        population_size = 20
        seed = 1235
        pdb = "/home/lavane/Users/acanella/Repos/multiObjectiveOptimizationDesign/tests/data/7R1K.pdb"
        chains = ["A"]
        mutable_positions = {
            "A": [34, 36, 55, 66, 70, 113, 120, 121, 122, 155, 156, 184]
        }
        mutable_aa = {
            "A": {
                34: ["H", "D", "Q", "S", "K", "M", "F", "L", "T", "A", "C"],
                36: ["R", "E", "D", "K", "A", "M", "S", "H", "L"],
                55: ["A", "Q", "E", "M", "L", "V", "C", "D", "K", "S", "F"],
                66: ["Q", "R", "E", "A", "M", "L", "S", "F", "H"],
                70: ["Q", "D", "S", "H", "T", "M", "A", "F", "R", "N", "V", "C", "E"],
                113: ["E", "D", "H", "Q", "S", "A", "C", "K", "N", "T", "V", "M"],
                120: ["S", "D", "H", "Q", "T", "C", "G", "A", "R", "K"],
                121: ["D", "S", "H", "T", "C", "V", "Y", "K", "Q", "F"],
                122: ["S", "H", "D", "K", "M", "R", "T", "A"],
                155: ["L", "C", "F", "M", "S", "A", "H", "I", "T"],
                156: ["H", "S", "T", "C", "D", "V"],
                184: ["K", "Q", "A", "R"],
            }
        }
        folder_name = "mood_job"
        xml_protocol = {}
        available_protocols = ["relaxAroundLigand", "relaxAroundResidues", "fullRelax"]
        if protocol not in available_protocols:
            raise ValueError(
                f"Protocol {protocol} not available. Available protocols are {available_protocols}"
            )
        elif protocol == "relaxAroundLigand":
            xml_protocol["main"] = XmlProtocols.relax_around_ligand()
        elif protocol == "relaxAroundResidues":
            xml_protocol["main"] = XmlProtocols.relax_around_residues()
        elif protocol == "fullRelax":
            xml_protocol["main"] = XmlProtocols.full_relax()

        moo = MultiObjectiveOptimization(
            optimizer=optimizer,
            metrics=metrics,
            debug=debug,
            max_iteration=max_iteration,
            pdb=pdb,
            chains=chains,
            data=data,
            mutable_positions=mutable_positions,
            mutable_aa=mutable_aa,
            folder_name=folder_name,
            seed=seed,
            population_size=population_size,
            offset=3,
        )
        if os.path.exists("mood_job"):
            shutil.rmtree("mood_job")
