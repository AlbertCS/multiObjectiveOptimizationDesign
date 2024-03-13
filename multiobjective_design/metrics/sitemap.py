import os
import time

import pandas as pd

# TODO may have to remove the relative import
from ..base import Silent
from .metric import Metric
from .utils import _copy_script_file, _parallel


class Sitemap(Metric):
    def __init__(self, iteration_folder, state, residue_selection, params=None):
        """
        Parameters
        ==========
        iteration_folder : str
            Path to the iteration folder.
        state : int
            The state for which the calculation will be carried out.
        residue_selection : list
            A list of the residues as tuples (residue_index, chain_id) defining
            where to look for the pocket.
        params : list
            A list of params files for reading the silent files from the state
            optimization outputs.
        """

        self.state = state
        self.iteration_folder = iteration_folder
        if isinstance(residue_selection, tuple):
            residue_selection = [residue_selection]
        self.residue_selection = residue_selection
        self.params = params

        self.job_folder = iteration_folder + "/sitemap"

        self.prepwizard_path = self.job_folder + "/prepwizard"
        self.prepwizard_state_folder = (
            self.prepwizard_path + "/state_" + str(self.state).zfill(2)
        )
        self.prepwizard_input_folder = self.prepwizard_state_folder + "/input_files"
        self.prepwizard_output_folder = self.prepwizard_state_folder + "/output_files"

        self.sitemap_path = self.job_folder + "/sitemap"
        self.sitemap_state_folder = (
            self.sitemap_path + "/state_" + str(self.state).zfill(2)
        )
        self.sitemap_input_folder = self.sitemap_state_folder + "/input_files"
        self.sitemap_output_folder = self.sitemap_state_folder + "/output_files"

    def setup_iteration_inputs(self):
        """
        Set up input folders for the given state optimized poses for prepwizard
        and sitemap calculations.
        """

        # Create job folders
        if not os.path.exists(self.job_folder):
            os.mkdir(self.job_folder)

        # Create calculation folders
        if not os.path.exists(self.prepwizard_path):
            os.mkdir(self.prepwizard_path)

        if not os.path.exists(self.sitemap_path):
            os.mkdir(self.sitemap_path)

        # Create state folders
        if not os.path.exists(self.prepwizard_state_folder):
            os.mkdir(self.prepwizard_state_folder)

        if not os.path.exists(self.sitemap_state_folder):
            os.mkdir(self.sitemap_state_folder)

        # Create input folders
        if not os.path.exists(self.prepwizard_input_folder):
            os.mkdir(self.prepwizard_input_folder)

        if not os.path.exists(self.sitemap_input_folder):
            os.mkdir(self.sitemap_input_folder)

        # Create output folders
        if not os.path.exists(self.prepwizard_output_folder):
            os.mkdir(self.prepwizard_output_folder)

        if not os.path.exists(self.sitemap_output_folder):
            os.mkdir(self.sitemap_output_folder)

        # Write input PDB files for prepwizard
        state_silent_file = (
            self.iteration_folder
            + "/output_files/"
            + str(self.state).zfill(2)
            + "_population.out"
        )
        if not os.path.exists(state_silent_file):
            raise ValueError(
                f"No optimization output was found for state {self.state} at {self.iteration_folder}/output_files/"
            )

        silent_object = Silent(state_silent_file, params=self.params)
        silent_object.extract_poses(self.prepwizard_input_folder)

    def compute_prepwizard(self, cpus=1):
        """
        Execute prepwizard calculations.
        """

        jobs = []
        for f in sorted(os.listdir(self.prepwizard_input_folder)):

            if not f.endswith(".pdb"):
                continue

            command = "cd " + self.prepwizard_output_folder + "\n"
            command += '"${SCHRODINGER}/utilities/prepwizard" '
            command += "../input_files/" + f + " "
            command += f + " "
            command += "-f 2005 "
            command += "-rmsd 0.3 "
            command += "-JOBNAME " + f.replace(".pdb", "") + " "
            command += "-HOST localhost:1 "
            command += "-WAIT\n"
            command += "cd " + "../" * 6 + "\n"

            jobs.append(command)

        _parallel(jobs, cpus=cpus, folder=self.prepwizard_state_folder)
        os.system("bash " + self.prepwizard_state_folder + "/commands")

        while not _is_finished(self.prepwizard_output_folder, ".pdb"):
            time.sleep(2)

    def compute_sitemap(
        self, cpus=1, site_box=10, resolution="fine", reportsize=100, sidechain=True
    ):
        """
        Execute sitemap calculations.
        """

        jobs = []
        for f in sorted(os.listdir(self.prepwizard_output_folder)):

            if not f.endswith(".pdb"):
                continue

            # Convert prepwizard output to sitemap MAE
            _copy_script_file(self.sitemap_state_folder, "prepareForSiteMap.py")
            script_path = self.sitemap_state_folder + "/._prepareForSiteMap.py"
            sequence_name = f.replace(".pdb", "")

            input_mae = self.sitemap_input_folder + "/" + sequence_name + "_protein.mae"
            if not os.path.exists(input_mae):
                command = '"${SCHRODINGER}/run" ' + script_path + " "
                command += self.prepwizard_output_folder + "/" + f + " "
                command += self.sitemap_input_folder + " "
                command += "--protein_only "
                os.system(command)

            command = "cd " + self.sitemap_output_folder + "\n"
            command += '"${SCHRODINGER}/sitemap" '
            command += "-j " + sequence_name + " "
            command += "-prot ../input_files/" + sequence_name + "_protein.mae "
            command += "-sitebox " + str(site_box) + " "
            command += "-resolution " + str(resolution) + " "
            command += "-keepvolpts yes "
            command += "-keeplogs yes "
            command += "-reportsize " + str(reportsize) + " "

            # For chain and residue index
            for r in self.residue_selection:
                if isinstance(r, tuple) and len(r) == 2:
                    command += (
                        '-siteasl "chain.name '
                        + str(r[1])
                        + " and res.num {"
                        + str(r[0])
                        + "} "
                    )
                else:
                    raise ValueError("Incorrect residue definition!")

                if sidechain:
                    command += "and not (atom.pt ca,c,n,h,o)"
                command += '" '
            command += "-HOST localhost:1 "
            command += "-TMPLAUNCHDIR "
            command += "-WAIT\n"
            command += "cd " + "../" * 6 + "\n"

            jobs.append(command)

        _parallel(jobs, cpus=cpus, folder=self.sitemap_state_folder)
        os.system("bash " + self.sitemap_state_folder + "/commands")

        while not _is_finished(self.sitemap_output_folder, "_protein.mae"):
            time.sleep(2)

    def read_results(self):

        predictions = {}
        predictions["Index"] = []

        for f in sorted(os.listdir(self.sitemap_output_folder)):

            if not f.endswith(".log"):
                continue

            sequence_name = f.replace(".log", "")
            cond = False
            with open(self.sitemap_output_folder + "/" + f) as lf:
                for l in lf:
                    if l.startswith("SiteScore"):
                        scores = l.split()
                        cond = True
                        continue
                    if cond:
                        for s, v in zip(scores, l.split()):
                            predictions.setdefault(s, [])
                            predictions[s].append(float(v))
                        predictions["Index"].append(sequence_name)
                        cond = False

        predictions = pd.DataFrame(predictions).set_index("Index")

        predictions.to_csv(
            self.job_folder + "/" + "sitemap_" + str(self.state).zfill(2) + ".csv"
        )

        return predictions

    def compute(self):
        """
        Execute the metric calculation.
        """

        self.setup_iteration_inputs()
        self.compute_prepwizard()
        self.compute_sitemap()
        return self.read_results()


def _check_log(log_file):
    success = False
    with open(log_file) as log:  # pylint: disable=unspecified-encoding
        for l in log:
            if "successfully" in l:
                success = True
    return success


def _is_finished(folder, extension):

    finished = {}
    for f in os.listdir(folder):
        if not f.endswith(extension):
            continue
        finished[f.replace(extension, ".log")] = False

    for m in finished:
        if os.path.exists(folder + "/" + m):
            finished[m] = _check_log(folder + "/" + m)
        else:
            return False

    if all([finished[s] for s in finished]):
        return True
    else:
        return False
