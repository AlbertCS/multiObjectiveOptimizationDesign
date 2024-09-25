import glob
import os
import pickle

import numpy as np
import pandas as pd
import pyrosetta as prs
from Bio.PDB import PDBParser
from mpi4py import MPI
from pyrosetta import rosetta, toolbox
from scipy.spatial import distance_matrix


class Mpi_relax:

    def relax_sequence(self, pose, jd, fastrelax_mover, sfxn):

        test_pose = prs.Pose()
        test_pose.assign(pose)

        counter = 0
        dE = {}
        while not jd.job_complete:
            counter += 1
            energy_ini = sfxn(test_pose)
            fastrelax_mover.apply(test_pose)
            energy_final = sfxn(test_pose)
            dE[counter] = energy_final - energy_ini
            jd.output_decoy(test_pose)

        energy = [float(x) for x in dE.values()]
        if not dE:
            raise ValueError("No energy values found")
        if len(energy) == 0:
            raise ValueError("No energy values found")

        mean_energy = sum(energy) / len(energy)

        return test_pose, mean_energy

    def calculate_Apo_Score(self, pose, sfxn, ligand_chain) -> float:
        """
        Calculate energy of the protein in apo form.

        Paramters
        =========
        pose : rosetta.
        ligand_chain : str.
        sfxn : rosetta score function.

        Returns
        =======
        apo_energy : float
        """
        if pose.pdb_info().num_chains() == 1:
            raise ValueError("The pose must contain more than one chain.")

        apo_pose = pose.clone()
        chain_id = rosetta.core.pose.get_chain_id_from_chain(ligand_chain, apo_pose)
        ligand_residues = rosetta.core.pose.get_chain_residues(apo_pose, chain_id)
        ligand_residues_id = [x.seqpos() for x in list(ligand_residues)]
        rosetta.protocols.grafting.delete_region(
            apo_pose, ligand_residues_id[0], ligand_residues_id[-1]
        )

        return sfxn(apo_pose)

    def distance(self, pose, atom1, atom2):
        a1 = np.array(pose.residue(atom1[0]).xyz(atom1[1]))
        a2 = np.array(pose.residue(atom2[0]).xyz(atom2[1]))
        return np.linalg.norm(a1 - a2)

    def _get_Coordinates(self, pose, residues=None, bb_only=False, sc_only=False):
        """
        Get all the pose atoms coordinates. An optional list of residues can be given
        to limit coordinates to only include the atoms of these residues.

        Parameters
        ==========
        pose : pyrosetta.rosetta.core.pose.Pose
            Pose from which to get the atomic coordinates.
        residues : list
            An optional list of residues to only get their coordinates.
        bb_only : bool
            Get only backbone atom coordinates from the pose.
        sc_only : bool
            Get only sidechain atom coordinates from the pose.

        Returns
        =======
        coordinates : numpy.ndarray
            The pose's coordinates.
        """

        if bb_only and sc_only:
            raise ValueError("bb_only and sc_only cannot be given simultaneously!")

        coordinates = []
        for r in range(1, pose.total_residue() + 1):
            # Check if a list of residue indexes is given.
            if residues != None:
                if r not in residues:
                    continue

            # Get residue coordinates
            residue = pose.residue(r)
            bb_indexes = residue.all_bb_atoms()
            for a in range(1, residue.natoms() + 1):

                # Skip non backbone atoms
                if bb_only:
                    if a not in bb_indexes:
                        continue

                # Skip backbone atoms
                if sc_only:
                    if a in bb_indexes:
                        continue

                # Get coordinates
                xyz = residue.xyz(a)
                xyz = np.array([xyz[0], xyz[1], xyz[2]])
                coordinates.append(xyz)

        coordinates = np.array(coordinates)

        return coordinates

    def calculate_Interface_Score(self, pose, sfxn, peptide_chain) -> float:
        """
        Calculate interaface score for a specified jump.

        Paramters
        =========
        pose : rosetta.

        """
        interface_pose = pose.clone()

        rosetta.core.scoring.constraints.remove_constraints_of_type(
            interface_pose, "AtomPair"
        )

        jump_id = rosetta.core.pose.get_jump_id_from_chain(
            peptide_chain, interface_pose
        )
        chain_id = rosetta.core.pose.get_chain_id_from_chain(
            peptide_chain, interface_pose
        )
        peptide_residues = rosetta.core.pose.get_chain_residues(
            interface_pose, chain_id
        )
        peptide_residues_id = [x.seqpos() for x in list(peptide_residues)]
        protein_residues_id = [
            r
            for r in range(1, interface_pose.total_residue())
            if r not in peptide_residues_id
        ]

        peptides_coor = self._get_Coordinates(interface_pose, peptide_residues_id)
        protein_coor = self._get_Coordinates(interface_pose, protein_residues_id)

        peptide_centroid = np.average(peptides_coor, axis=0)
        protein_centroid = np.average(protein_coor, axis=0)
        vector = peptide_centroid - protein_centroid
        vector = vector / np.linalg.norm(vector)
        vector = rosetta.numeric.xyzVector_double_t(vector[0], vector[1], vector[2])

        energy_ini = sfxn(interface_pose)

        peptide_mover = rosetta.protocols.rigid.RigidBodyTransMover()
        peptide_mover.trans_axis(vector)
        peptide_mover.step_size(1000)
        peptide_mover.rb_jump(jump_id)
        peptide_mover.apply(interface_pose)

        energy_fin = sfxn(interface_pose)

        return energy_ini - energy_fin

    def mutate_native_pose(self, pose, seq):
        for res, aa in zip(pose.residues, seq):
            if res.name1() == "Z":
                continue
            elif str(res.name1()) != str(aa):
                toolbox.mutate_residue(pose, res.seqpos(), aa)
        return pose

    def get_salt_bridges(self, pdb_path, code):
        # Load pdb in Biopython
        parser = PDBParser()
        structure = parser.get_structure(code, pdb_path)

        # Get residue coordinates
        acid_coords = []
        base_coords = []

        acid_mapping = {}
        base_mapping = {}

        ai = 0
        bi = 0
        for r in structure.get_residues():
            if r.resname in ["ASP", "GLU"]:
                for a in r:
                    if a.name in ["OD1", "OD2", "OE1", "OE2"]:
                        acid_coords.append(a.coord)
                        acid_mapping[ai] = a
                        ai += 1

            if r.resname in ["ARG", "LYS"]:
                for a in r:
                    if a.name in ["NE", "NH1", "NH2", "NZ"]:
                        base_coords.append(a.coord)
                        base_mapping[bi] = a
                        bi += 1

        acid_coords = np.array(acid_coords)
        base_coords = np.array(base_coords)

        M = distance_matrix(acid_coords, base_coords)
        salt_bridges = []
        for x in np.argwhere(M <= 3.2):
            acid = acid_mapping[x[0]].get_parent().id[1]
            base = base_mapping[x[1]].get_parent().id[1]
            salt_bridges.append((acid, base))
        salt_bridges = sorted(list(set(salt_bridges)))

        return len(salt_bridges)

    def distance(self, pose, atom1, atom2):
        a1 = np.array(pose.residue(atom1[0]).xyz(atom1[1]))
        a2 = np.array(pose.residue(atom2[0]).xyz(atom2[1]))
        return np.linalg.norm(a1 - a2)

    def main(
        self,
        job_output="output_relax/decoy",
        input_file="sequences.txt",
        distances_pkl=None,
        native_pdb=None,
    ):
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"Size: {size}")
        # size = 4

        # List of sequences to be relaxed
        sequences = []
        # Open the file in read mode
        with open(input_file, "r") as file:
            # Read the contents of the file
            lines = file.readlines()
            # Iterate over each line
            for line in lines:
                # Strip any leading/trailing whitespace and add to the list
                sequences.append(line.strip())

        # Initialize native pose
        native_pose = prs.pyrosetta.pose_from_pdb(native_pdb)

        # Initialize score function and relax mover
        sfxn = prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            "ref2015"
        )
        fastrelax_mover = prs.rosetta.protocols.relax.FastRelax()
        fastrelax_mover.set_scorefxn(sfxn)

        # Initialize filters and calculators
        hydro_filter = (
            prs.rosetta.protocols.denovo_design.filters.ExposedHydrophobicsFilter()
        )
        hydro_filter.set_sasa_cutoff(0)
        hydro_filter.set_threshold(-1)
        salt_bridges_calculator = (
            prs.rosetta.protocols.pose_metric_calculators.SaltBridgeCalculator()
        )

        # Read distances
        distance_dict = pickle.load(open(distances_pkl, "rb"))

        # Distribute sequences equally among processors
        num_sequences = len(sequences)
        sequences_per_proc = num_sequences // size
        remainder = num_sequences % size
        start_index = rank * sequences_per_proc + min(rank, remainder)
        end_index = start_index + sequences_per_proc + (1 if rank < remainder else 0)

        # Each processor relaxes its assigned sequences
        relaxed_energies = []
        interface_scores = []
        apo_scores = []
        hydrophobic_scores = []
        n_salt_bridges_iter = []
        distances_res = []
        for i in range(start_index, end_index):
            jd = prs.PyJobDistributor(
                f"{job_output}_R{rank}_I{i}", nstruct=1, scorefxn=sfxn
            )
            pose = self.mutate_native_pose(native_pose, sequences[i])
            test_pose, mean_energy = self.relax_sequence(
                pose=pose,
                jd=jd,
                fastrelax_mover=fastrelax_mover,
                sfxn=sfxn,
            )
            apo_score = self.calculate_Apo_Score(test_pose, sfxn, "L")
            interface_score = self.calculate_Interface_Score(test_pose, sfxn, "L")
            hydro_filter.apply(test_pose)
            hydrophobic_score = hydro_filter.compute(test_pose)
            n_salt_bridges = salt_bridges_calculator.get(
                key="salt_bridge", this_pose=test_pose
            )
            res_distance = distance_dict.copy()
            for key, value in distance_dict.items():
                res_distance[key] = self.distance(pose, value[0], value[1])
            # n_salt_bidges = self.get_salt_bridges(f"{job_output}_R{rank}_I{i}_0.pdb", "pdb")

            # Gather results
            relaxed_energies.append((i, mean_energy))
            interface_scores.append((i, interface_score))
            apo_scores.append((i, apo_score))
            hydrophobic_scores.append((i, hydrophobic_score))
            n_salt_bridges_iter.append((i, n_salt_bridges))
            distances_res.append((i, res_distance))

        # Delete the pdbs
        # Construct the pattern to match the files
        pattern = f"{job_output}_R{rank}_*.pdb"

        # Find all files matching the pattern
        files_to_remove = glob.glob(pattern)

        # Remove each file
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing file {file_path}: {e.strerror}")

        # Gather results from all processors
        all_relaxed_energies = comm.gather(relaxed_energies, root=0)
        all_interface_scores = comm.gather(interface_scores, root=0)
        all_apo_scores = comm.gather(apo_scores, root=0)
        all_hydrophobic_scores = comm.gather(hydrophobic_scores, root=0)
        all_salt_bridges = comm.gather(n_salt_bridges_iter, root=0)
        all_distances_res = comm.gather(distances_res, root=0)

        # If rank 0, process the gathered results
        if rank == 0:
            # Flatten the list of lists
            def flatten_and_merge(all_data, column_names):
                flattened_data = [item for sublist in all_data for item in sublist]
                return pd.DataFrame(flattened_data, columns=column_names)

            df_relaxed_energies = flatten_and_merge(
                all_relaxed_energies, ["Index", "Relax Energy"]
            )
            df_interface_scores = flatten_and_merge(
                all_interface_scores, ["Index", "Interface Score"]
            )
            df_apo_scores = flatten_and_merge(all_apo_scores, ["Index", "Apo Score"])
            df_hydrophobic_scores = flatten_and_merge(
                all_hydrophobic_scores, ["Index", "Hydrophobic Score"]
            )
            df_salt_bridges = flatten_and_merge(
                all_salt_bridges, ["Index", "Salt Bridges"]
            )
            flattened_dist = [item for sublist in all_distances_res for item in sublist]
            df_distances = pd.DataFrame(
                [item[1] for item in flattened_dist],
                index=[item[0] for item in flattened_dist],
            )
            df_distances = df_distances.rename(columns=lambda x: "dist_" + x)

            df = df_relaxed_energies
            df = df.merge(df_interface_scores, on="Index")
            df = df.merge(df_apo_scores, on="Index")
            df = df.merge(df_hydrophobic_scores, on="Index")
            df = df.merge(df_salt_bridges, on="Index")
            df = df.merge(df_distances, left_on="Index", right_index=True)

            df.to_csv("rosetta_scores.csv", index=False)


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="MPI Relaxation Script")
    parser.add_argument(
        "--job_output",
        type=str,
        default="output_relax/decoy",
        help="Output directory for the job",
    )
    parser.add_argument(
        "--native_pdb", type=str, required=True, help="Path to the native PDB file"
    )
    parser.add_argument(
        "--params_folder", type=str, required=True, help="Path to the parameters folder"
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed for the job"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="sequences.txt",
        help="Input file containing sequences",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default="distance.pkl",
        required=True,
        help="File containing distances between residues",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    options = f"-relax:default_repeats 1 -constant_seed true -jran {args.seed}"
    options += f" -extra_res_path {args.params_folder}"

    prs.pyrosetta.init(options=options)

    mpi_relax = Mpi_relax()

    if not os.path.exists(args.job_output.split("/")[0]):
        os.makedirs(args.job_output.split("/")[0])

    mpi_relax.main(
        job_output=args.job_output,
        input_file=args.input_file,
        native_pdb=args.native_pdb,
        distances_pkl=args.distances,
    )
