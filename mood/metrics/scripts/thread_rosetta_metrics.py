import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyrosetta as prs
from pyrosetta import rosetta, toolbox


def initialize_pyrosetta(options):
    """Initialize PyRosetta with given options in each process."""
    prs.pyrosetta.init(options=options)


class ProcessRelax:
    def __init__(self, pyrosetta_options):
        self.pyrosetta_options = pyrosetta_options

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
            if counter > 10:
                jd.job_complete = True
                break

            jd.output_decoy(test_pose)

        energy = [float(x) for x in dE.values()]
        if not dE:
            raise ValueError("No energy values found")
        if len(energy) == 0:
            raise ValueError("No energy values found")

        return test_pose, energy_final

    def calculate_Apo_Score(self, pose, sfxn, ligand_chain) -> float:
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
        if bb_only and sc_only:
            raise ValueError("bb_only and sc_only cannot be given simultaneously!")

        coordinates = []
        for r in range(1, pose.total_residue() + 1):
            if residues is not None:
                if r not in residues:
                    continue

            residue = pose.residue(r)
            bb_indexes = residue.all_bb_atoms()
            for a in range(1, residue.natoms() + 1):
                if bb_only and a not in bb_indexes:
                    continue
                if sc_only and a in bb_indexes:
                    continue
                xyz = residue.xyz(a)
                xyz = np.array([xyz[0], xyz[1], xyz[2]])
                coordinates.append(xyz)

        return np.array(coordinates)

    def calculate_Interface_Score(self, pose, sfxn, peptide_chain) -> float:
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

    def process_sequence(self, args):
        """Process a single sequence with the given parameters."""
        # Initialize PyRosetta in this process
        initialize_pyrosetta(self.pyrosetta_options)

        # Unpack arguments
        (
            idx,
            sequence,
            pdb_path,
            output_folder,
            cst_file,
            distance_dict,
            ligand_chain,
            atom_pair_constraint_weight,
        ) = args

        # Initialize native pose
        native_pose = prs.pyrosetta.pose_from_pdb(pdb_path)

        # Initialize score function
        sfxn = prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            "ref2015"
        )

        # Apply constraints if provided
        if cst_file != "None" and cst_file is not None:
            sfxn.set_weight(rosetta.core.scoring.ScoreType.res_type_constraint, 1)
            set_constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
            set_constraints.constraint_file(cst_file)
            sfxn.set_weight(
                rosetta.core.scoring.ScoreType.atom_pair_constraint,
                atom_pair_constraint_weight,
            )
            set_constraints.add_constraints(True)
            set_constraints.apply(native_pose)

        # Initialize movers and filters
        fastrelax_mover = prs.rosetta.protocols.relax.FastRelax()
        fastrelax_mover.set_scorefxn(sfxn)
        hydro_filter = (
            prs.rosetta.protocols.denovo_design.filters.ExposedHydrophobicsFilter()
        )
        hydro_filter.set_sasa_cutoff(0)
        hydro_filter.set_threshold(-1)
        salt_bridges_calculator = (
            prs.rosetta.protocols.pose_metric_calculators.SaltBridgeCalculator()
        )

        jd = prs.PyJobDistributor(
            f"{output_folder}/decoy_{idx}",
            nstruct=1,
            scorefxn=sfxn,
            compress=False,
        )

        pose = self.mutate_native_pose(native_pose, sequence)
        test_pose, mean_energy = self.relax_sequence(
            pose=pose,
            jd=jd,
            fastrelax_mover=fastrelax_mover,
            sfxn=sfxn,
        )

        results = {"Index": idx, "Relax_Energy": mean_energy}

        if ligand_chain != "None" and ligand_chain is not None:
            results["Apo_Score"] = self.calculate_Apo_Score(
                test_pose, sfxn, ligand_chain
            )
            results["Interface_Score"] = self.calculate_Interface_Score(
                test_pose, sfxn, ligand_chain
            )

        hydro_filter.apply(test_pose)
        results["Hydrophobic_Score"] = hydro_filter.compute(test_pose)
        results["Salt_Bridges"] = salt_bridges_calculator.get(
            key="salt_bridge", this_pose=test_pose
        )

        if distance_dict != "None" and distance_dict is not None:
            res_distance = distance_dict.copy()
            for key, value in distance_dict.items():
                res_distance[key] = self.distance(test_pose, value[0], value[1])
            results.update({f"dist_{k}": v for k, v in res_distance.items()})

        return results

    def main(
        self,
        output_folder="output_relax",
        sequences_file="sequences.txt",
        distance_file=None,
        native_pdb=None,
        cst_file=None,
        ligand_chain=None,
        atom_pair_constraint_weight=1,
        n_processes=None,
    ):
        # Read sequences
        with open(sequences_file, "r") as file:
            sequences = [line.strip() for line in file.readlines()]

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        distance_dict = None
        if distance_file is not None:
            if distance_file.endswith(".pkl"):
                with open(distance_file, "rb") as file:
                    distance_dict = pickle.load(file)
            elif distance_file.endswith(".json"):
                with open(distance_file, "r") as file:
                    distance_dict = json.load(file)

        # Prepare arguments for each sequence
        args = [
            (
                i,
                seq,
                native_pdb,
                output_folder,
                cst_file,
                distance_dict,
                ligand_chain,
                atom_pair_constraint_weight,
            )
            for i, seq in enumerate(sequences)
        ]

        # Process sequences in parallel using ProcessPoolExecutor
        with Pool(
            processes=n_processes,
            initializer=initialize_pyrosetta,
            initargs=(self.pyrosetta_options,),
        ) as pool:
            results = pool.map(self.process_sequence, args)

        # Convert results to DataFrame and sort by Index
        df = pd.DataFrame(results)
        df = df.sort_values("Index").reset_index(drop=True)
        df.to_csv(f"{output_folder}/rosetta_scores.csv", index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Pool Relaxation Script")
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
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of processes to use (defaults to CPU count)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.params_folder is None:
        patches = []
        params = []
    elif not os.path.exists(args.params_folder):
        print(f"Warning: Directory {args.params_folder} does not exist")
        patches = []
        params = []
    else:
        patches = [
            os.path.join(args.params_folder, x)
            for x in os.listdir(args.params_folder)
            if x.endswith(".txt")
        ]
        params = [
            os.path.join(args.params_folder, x)
            for x in os.listdir(args.params_folder)
            if x.endswith(".params")
        ]

    # Build PyRosetta options string
    pyrosetta_options = f"-relax:default_repeats 1 -relax:range:cycles 5 -constant_seed true -jran {args.seed}"
    if params:
        params = " ".join(params)
        pyrosetta_options += f" -extra_res_fa {params}"
    if patches:
        patches = " ".join(patches)
        pyrosetta_options += f" -extra_patch_fa {patches}"

    if not os.path.exists(args.native_pdb):
        raise ValueError(f"The native pdb file {args.native_pdb} does not exist")

    if not os.path.exists(args.sequences_file):
        raise ValueError(f"The sequence file {args.sequences_file} does not exist")

    process_relax = ProcessRelax(pyrosetta_options)
    process_relax.main(
        output_folder=args.output_folder,
        sequences_file=args.sequences_file,
        native_pdb=args.native_pdb,
        distance_file=args.distances,
        cst_file=args.cst_file,
        ligand_chain=args.ligand_chain,
        n_processes=args.n_processes,
    )
