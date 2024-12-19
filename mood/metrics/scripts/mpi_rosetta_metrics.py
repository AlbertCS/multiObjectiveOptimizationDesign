import glob
import json
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

        return test_pose, energy_final

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
            if str(res.name1()) == "Z" or str(res.name1()) == "X":
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
        output_folder="output_relax",
        sequences_file="sequences.txt",
        distance_dict=None,
        native_pdb=None,
        cst_file=None,
        ligand_chain=None,
        atom_pair_constraint_weight=1,
    ):
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # size = 4

        # List of sequences to be relaxed
        sequences = []
        # Open the file in read mode
        with open(sequences_file, "r") as file:
            # Read the contents of the file
            lines = file.readlines()
            # Iterate over each line
            for line in lines:
                # Strip any leading/trailing whitespace and add to the list
                sequences.append(line.strip())

        # Initialize native pose
        native_pose = prs.pyrosetta.pose_from_pdb(native_pdb)

        if distance_dict is not None:
            # Get the distances dictionary
            for key, value in distances.items():
                if len(value[0]) == 3:
                    natom1 = native_pose.pdb_info().pdb2pose(value[0][0], value[0][1])
                    natom2 = native_pose.pdb_info().pdb2pose(value[1][0], value[1][1])
                    distances[key] = [(natom1, value[0][2]), (natom2, value[1][2])]

        # Initialize score function and relax mover
        sfxn = prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            "ref2015"
        )
        sfxn_scorer = (
            prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
                "ref2015"
            )
        )

        # Apply constraints to the energy function
        if cst_file != "None":
            sfxn.set_weight(rosetta.core.scoring.ScoreType.res_type_constraint, 1)
            # Define catalytic constraints
            set_constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
            # Add constraint file
            set_constraints.constraint_file(cst_file)
            # Add atom_pair_constraint_weight ScoreType
            sfxn.set_weight(
                rosetta.core.scoring.ScoreType.atom_pair_constraint,
                atom_pair_constraint_weight,
            )
            # Turn on constraint with the mover
            set_constraints.add_constraints(True)
            set_constraints.apply(native_pose)

        # Initialize the fastRelax mover
        fastrelax_mover = prs.rosetta.protocols.relax.FastRelax()
        fastrelax_mover.set_scorefxn(sfxn)

        # Initialize filters and calculators
        # initialize hydrophobic surface calculator
        hydro_filter = (
            prs.rosetta.protocols.denovo_design.filters.ExposedHydrophobicsFilter()
        )
        hydro_filter.set_sasa_cutoff(0)
        hydro_filter.set_threshold(-1)
        # initialize Salt bridge calculator
        salt_bridges_calculator = (
            prs.rosetta.protocols.pose_metric_calculators.SaltBridgeCalculator()
        )

        # Distribute sequences equally among processors
        num_sequences = len(sequences)
        sequences_per_proc = num_sequences // size
        remainder = num_sequences % size
        start_index = rank * sequences_per_proc + min(rank, remainder)
        end_index = start_index + sequences_per_proc + (1 if rank < remainder else 0)

        # Initialize all the datastructures
        relaxed_energies = []
        interface_scores = []
        apo_scores = []
        hydrophobic_scores = []
        n_salt_bridges_iter = []
        distances_res = []
        # Each processor relaxes its assigned sequences
        for i in range(start_index, end_index):
            jd = prs.PyJobDistributor(
                f"{output_folder}/decoy_R{rank}_I{i}",
                nstruct=1,
                scorefxn=sfxn_scorer,
                compress=True,
            )
            # Get the mutated sequence translated to a pose
            pose = self.mutate_native_pose(native_pose, sequences[i])
            # Relax the pose
            test_pose, mean_energy = self.relax_sequence(
                pose=pose,
                jd=jd,
                fastrelax_mover=fastrelax_mover,
                sfxn=sfxn_scorer,
            )
            if ligand_chain != "None":
                # Get the apo score
                apo_score = self.calculate_Apo_Score(
                    test_pose, sfxn_scorer, ligand_chain
                )
                # Get the binding score
                interface_score = self.calculate_Interface_Score(
                    test_pose, sfxn_scorer, ligand_chain
                )
            # Apply the hydrophobic filter and get the score
            hydro_filter.apply(test_pose)
            hydrophobic_score = hydro_filter.compute(test_pose)
            # Calculate the salt bridges
            n_salt_bridges = salt_bridges_calculator.get(
                key="salt_bridge", this_pose=test_pose
            )
            # Calculate the distances
            if distance_dict is not None:
                res_distance = distance_dict.copy()
                for key, value in distance_dict.items():
                    res_distance[key] = self.distance(test_pose, value[0], value[1])
            # n_salt_bidges = self.get_salt_bridges(f"{job_output}_R{rank}_I{i}_0.pdb", "pdb")

            # Gather results
            relaxed_energies.append((i, mean_energy))
            if ligand_chain != "None":
                print("ligand_chain is not None in append")
                interface_scores.append((i, interface_score))
                apo_scores.append((i, apo_score))
            hydrophobic_scores.append((i, hydrophobic_score))
            n_salt_bridges_iter.append((i, n_salt_bridges))
            if distance_dict is not None:
                distances_res.append((i, res_distance))

        # Gather results from all processors
        all_relaxed_energies = comm.gather(relaxed_energies, root=0)
        if ligand_chain != "None":
            all_interface_scores = comm.gather(interface_scores, root=0)
            all_apo_scores = comm.gather(apo_scores, root=0)
        all_hydrophobic_scores = comm.gather(hydrophobic_scores, root=0)
        all_salt_bridges = comm.gather(n_salt_bridges_iter, root=0)
        if distance_dict is not None:
            all_distances_res = comm.gather(distances_res, root=0)

        # Delete the pdbs
        # Construct the pattern to match the files
        pattern = f"{output_folder}/decoy_R{rank}_*.pdb"

        # Find all files matching the pattern
        # files_to_remove = glob.glob(pattern)
        # # Remove each file
        # for file_path in files_to_remove:
        #     try:
        #         os.remove(file_path)
        #     except OSError as e:
        #         print(f"Error removing file {file_path}: {e.strerror}")

        # Remove each file
        # for i in range(start_index, end_index):
        #     try:
        #         os.remove(f"{output_folder}/decoy_R{rank}_I{i}_0.pdb.gz")
        #     except OSError as e:
        #         print(
        #             f"Error removing file {output_folder}/decoy_R{rank}_I{i}_0.pdb: {e.strerror}"
        #         )

        # If rank 0, process the gathered results
        if rank == 0:
            # Flatten the list of lists
            def flatten_and_merge(all_data, column_names):
                flattened_data = [item for sublist in all_data for item in sublist]
                return pd.DataFrame(flattened_data, columns=column_names)

            df_relaxed_energies = flatten_and_merge(
                all_relaxed_energies, ["Index", "Relax_Energy"]
            )
            if ligand_chain != "None":
                df_interface_scores = flatten_and_merge(
                    all_interface_scores, ["Index", "Interface_Score"]
                )
                df_apo_scores = flatten_and_merge(
                    all_apo_scores, ["Index", "Apo_Score"]
                )
            df_hydrophobic_scores = flatten_and_merge(
                all_hydrophobic_scores, ["Index", "Hydrophobic_Score"]
            )
            df_salt_bridges = flatten_and_merge(
                all_salt_bridges, ["Index", "Salt_Bridges"]
            )
            if distance_dict is not None:
                flattened_dist = [
                    item for sublist in all_distances_res for item in sublist
                ]
                df_distances = pd.DataFrame(
                    [item[1] for item in flattened_dist],
                    index=[item[0] for item in flattened_dist],
                )
                df_distances = df_distances.rename(columns=lambda x: "dist_" + x)

            # Create the final df with all the values
            df = df_relaxed_energies
            if ligand_chain != "None":
                df = df.merge(df_interface_scores, on="Index")
                df = df.merge(df_apo_scores, on="Index")
            df = df.merge(df_hydrophobic_scores, on="Index")
            df = df.merge(df_salt_bridges, on="Index")
            if distance_dict is not None:
                df = df.merge(df_distances, left_on="Index", right_index=True)

            df.to_csv(f"{output_folder}/rosetta_scores.csv", index=False)


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="MPI Relaxation Script")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output_relax",
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
        "--sequences_file",
        type=str,
        default="sequences.txt",
        help="Input file containing sequences",
    )
    parser.add_argument(
        "--distances",
        default=None,
        type=str,
        required=True,
        help="File containing distances between residues",
    )
    parser.add_argument(
        "--cst_file",
        default=None,
        type=str,
        help="File containing constraints for the job",
    )
    parser.add_argument(
        "--ligand_chain",
        default=None,
        type=str,
        help="Indicate the chain of the ligand",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    distances = None

    params = []
    patches = []

    if args.params_folder is not None:
        patches = [
            args.params_folder + "/" + x
            for x in os.listdir(args.params_folder)
            if x.endswith(".txt")
        ]
        params = [
            args.params_folder + "/" + x
            for x in os.listdir(args.params_folder)
            if x.endswith(".params")
        ]

    # Prom a list of path files, create a string with all the paths separated by a space
    params = " ".join(params)
    patches = " ".join(patches)

    print(f"Params:\n{params}")
    print(f"Patches:\n{patches}")

    options = f"-relax:default_repeats 1 -constant_seed true -jran {args.seed}"
    if params:
        options += f" -extra_res_fa {params}"
    if patches:
        options += f" -extra_patch_fa {patches}"

    prs.pyrosetta.init(options=options)

    mpi_relax = Mpi_relax()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if not os.path.exists(args.native_pdb):
        raise ValueError(f"The native pdb file {args.native_pdb} does not exist")

    if args.distances is not None:
        if args.distances.endswith(".pkl"):
            with open(args.distances, "rb") as file:
                distances = pickle.load(file)
        elif args.distances.endswith(".json"):
            with open(args.distances, "r") as file:
                distances = json.load(file)

    if not os.path.exists(args.sequences_file):
        raise ValueError(f"The sequence file {args.sequences_file} does not exists")

    mpi_relax.main(
        output_folder=args.output_folder,
        sequences_file=args.sequences_file,
        native_pdb=args.native_pdb,
        distance_dict=distances,
        cst_file=args.cst_file,
        ligand_chain=args.ligand_chain,
    )
