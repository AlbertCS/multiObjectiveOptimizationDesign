import random

import numpy as np
from pyrosetta import FoldTree, rosetta


class PyrosettaFunctions:
    """
    A class that represents a relaxation protocol for protein design using Rosetta.

    Parameters
    ----------
    sfxn : str, optional
        The name of the Rosetta score function to use (default is "ref2015").
    atom_pair_constraint_weight : int, optional
        The weight of the atom pair constraint score term (default is 1).
    minimization_steps : int, optional
        The maximum number of minimization steps to perform (default is 100).
    energy_threshold : float, optional
        The energy threshold for convergence during minimization (default is 2).

    Methods
    -------
    calculateApoScore(pose, peptide_chain)
        Calculate the energy of the protein in apo form.
    _getCoordinates(pose, residues=None, bb_only=False, sc_only=False)
        Get the atomic coordinates of the pose.
    calculateInterfaceScore(pose, peptide_chain)
        Calculate the interface score for a specified jump.
    define_constraint(cst_file)
        Define catalytic constraints from a constraint file.
    relax(pose)
        Perform relaxation of the pose using the defined protocol.
    """

    def __init__(
        self,
        sfxn="ref2015",
        atom_pair_constraint_weight=1,
        minimization_steps=100,
        energy_threshold=2,
    ):

        self.sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            sfxn
        )
        self.atom_pair_constraint_weight = atom_pair_constraint_weight
        self.minimization_steps = minimization_steps
        self.energy_threshold = energy_threshold

    def calculate_Apo_Score(self, pose, ligand_chain) -> float:
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

        return self.sfxn(apo_pose)

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

    def calculate_Interface_Score(self, pose, peptide_chain) -> float:
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

        energy_ini = self.sfxn(interface_pose)

        peptide_mover = rosetta.protocols.rigid.RigidBodyTransMover()
        peptide_mover.trans_axis(vector)
        peptide_mover.step_size(1000)
        peptide_mover.rb_jump(jump_id)
        peptide_mover.apply(interface_pose)

        energy_fin = self.sfxn(interface_pose)

        return energy_ini - energy_fin

    def define_constraint(self, cst_file):
        # Define catalytic constraints from cst_file
        set_constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
        # Add constraint file
        set_constraints.constraint_file(cst_file)
        # Add atom_pair_constraint_weight ScoreType
        self.sfxn.set_weight(
            rosetta.core.scoring.ScoreType.atom_pair_constraint,
            self.atom_pair_constraint_weight,
        )
        # Turn on constraint with the mover
        set_constraints.add_constraints(True)

    def relax(
        self,
        pose,
        minimization_cycles=10,
    ):
        fastrelax_mover_sampling = rosetta.protocols.relax.FastRelax()
        fastrelax_mover_sampling.set_scorefxn(self.sfxn)
        print("Running minimisation...")
        energy_ini = self.sfxn(pose)
        print(f"\tInitial energy: {energy_ini:.2f}")
        # need -relax:default_repeats 1 in init for pyrosetta
        for step in range(1, minimization_cycles + 1):
            fastrelax_mover_sampling.apply(pose)
            energy_final = self.sfxn(pose)
            d_energy = energy_final - energy_ini
            print(f"Finished minimization cycle {step} of {minimization_cycles}")
            print(f"Energy change (dE): {d_energy:.2f}")
            print(f"\tCurrent energy: {energy_final:.2f}")
            if abs(d_energy) < abs(self.energy_threshold):
                print("Energy convergence achieved.")
                break
            energy_ini = energy_final
        if step == self.minimization_steps - 1:
            print(
                f"Maximum number of relax iterations ({minimization_cycles}) reached."
            )
            print("Energy convergence failed!")

        print("Finished minimisation")

        # pose.dump_pdb("relaxed.pdb")

    def _getPeptideFoldTree(
        self, pose, anchoring_residue, peptide_chain="L", ft_type="m"
    ):
        """
        Change the Fold Tree to center on peptide propagation randomly by choosing
        among: N-terminal bound (nt), C-terminal bound (ct) and middle-out (m).

        Parameters
        ==========
        pose : pyrosetta.rosetta.core.pose.Pose
            Pose containing the peptide as the last chain.
        anchoring_residue : int
            Protein residue serving as jump to connect the peptide in the fold tree.
        ft_type : str
            Type of peptide fold tree to generate (ct, nt, or m)

        Returns
        =======
            fold_tree : pyrosetta.rosetta.core.kinematics.FoldTree
                The poses' fold tree type object. This can be given to the pose to set
                up the fold tree kinematics.
        """

        accepted_ft = ["ct", "nt", "m"]
        if ft_type not in accepted_ft:
            raise ValueError(
                "Unknown ft_type specified. Values allowed are: "
                + " ".join(accepted_ft)
            )

        # Get residues by chain
        chains = list(rosetta.core.pose.get_chains(pose))
        peptide_chain_id = rosetta.core.pose.get_chain_id_from_chain(
            peptide_chain, pose
        )
        residues = {}
        for r in range(1, pose.total_residue() + 1):
            chain = pose.residue(r).chain()
            if chain == peptide_chain_id:
                peptide_chain = chain
            if chain not in residues:
                residues[chain] = []
            residues[chain].append(r)

        last_residue = 0
        jump = 1
        fold_tree = FoldTree()
        for chain in residues:

            if chain != peptide_chain and anchoring_residue not in residues[chain]:
                if len(residues[chain]) > 1:
                    fold_tree.add_edge(residues[chain][0], residues[chain][-1], -1)
                    if last_residue != 0:
                        fold_tree.add_edge(last_residue, residues[chain][0], jump)
                        jump += 1
                    last_residue = residues[chain][-1]
                else:
                    fold_tree.add_edge(anchoring_residue, residues[chain][0], jump)
                    jump += 1

            elif chain != peptide_chain:
                if last_residue != 0:
                    fold_tree.add_edge(anchoring_residue, last_residue, jump)
                    jump += 1
                fold_tree.add_edge(anchoring_residue, residues[chain][0], -1)
                fold_tree.add_edge(anchoring_residue, residues[chain][-1], -1)

        if ft_type == "nt":
            fold_tree.add_edge(anchoring_residue, residues[peptide_chain][0], jump)
            jump += 1
            fold_tree.add_edge(
                residues[peptide_chain][0], residues[peptide_chain][-1], -1
            )
        elif ft_type == "ct":
            fold_tree.add_edge(anchoring_residue, residues[peptide_chain][-1], jump)
            jump += 1
            fold_tree.add_edge(
                residues[peptide_chain][-1], residues[peptide_chain][0], -1
            )
        elif ft_type == "m":
            if (len(residues[peptide_chain]) % 2) == 0:
                mr_index = int(len(residues[peptide_chain]) / 2) - 1
            else:
                mr_index = int(len(residues[peptide_chain]) / 2)

            fold_tree.add_edge(
                anchoring_residue, residues[peptide_chain][mr_index], jump
            )
            jump += 1
            fold_tree.add_edge(
                residues[peptide_chain][mr_index], residues[peptide_chain][0], -1
            )
            fold_tree.add_edge(
                residues[peptide_chain][mr_index], residues[peptide_chain][-1], -1
            )

        if not fold_tree.check_fold_tree():
            raise ValueError("Error in FoldTree construction.")

        return fold_tree

    def local_relax(
        self,
        pose,
        sfxn,
        residues=None,
        moving_chain=None,
        fold_tree=True,
        neighbour_distance=10,
        minimization_steps=100,
        min_energy_threshold=2,
        pymol=False,
    ):
        residues = residues or []
        min_pose = pose.clone()

        fastrelax_mover = rosetta.protocols.relax.FastRelax()
        fastrelax_mover.set_scorefxn(sfxn)

        if moving_chain is not None:
            chain_indexes = []
            for r in range(1, pose.total_residue() + 1):
                _, chain = pose.pdb_info().pose2pdb(r).split()
                if chain == moving_chain:
                    chain_indexes.append(r)
        else:
            chain_indexes = []

        ct_chain_selector = rosetta.core.select.residue_selector.ResidueIndexSelector()
        indexes = chain_indexes + residues
        ct_chain_selector.set_index(",".join([str(x) for x in indexes]))

        nbr_selector = (
            rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
        )
        nbr_selector.set_focus_selector(ct_chain_selector)
        nbr_selector.set_include_focus_in_subset(
            True
        )  # This includes the peptide residues in the selection
        nbr_selector.set_distance(neighbour_distance)

        enable_mm = rosetta.core.select.movemap.move_map_action(1)

        mmf_relax = rosetta.core.select.movemap.MoveMapFactory()
        mmf_relax.all_bb(False)
        mmf_relax.add_bb_action(enable_mm, nbr_selector)

        ## Deactivate side-chain except for the selected chain during relax + neighbours
        mmf_relax.all_chi(False)
        mmf_relax.add_chi_action(enable_mm, nbr_selector)

        # Define RLT to prevent repacking of residues (fix side chains)
        prevent_repacking_rlt = rosetta.core.pack.task.operation.PreventRepackingRLT()
        # Define RLT to only repack residues (movable side chains but fixed sequence)
        restrict_repacking_rlt = (
            rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        )

        # Prevent repacking of everything but CT, peptide and their neighbours
        prevent_subset_repacking = (
            rosetta.core.pack.task.operation.OperateOnResidueSubset(
                prevent_repacking_rlt, nbr_selector, True
            )
        )
        # Allow repacking of peptide and neighbours
        restrict_subset_to_repacking_for_sampling = (
            rosetta.core.pack.task.operation.OperateOnResidueSubset(
                restrict_repacking_rlt, nbr_selector
            )
        )

        tf_sampling = rosetta.core.pack.task.TaskFactory()
        tf_sampling.push_back(prevent_subset_repacking)
        tf_sampling.push_back(restrict_subset_to_repacking_for_sampling)

        fastrelax_mover.set_movemap_factory(mmf_relax)
        fastrelax_mover.set_task_factory(tf_sampling)

        ## Run minimization
        if pymol:
            pymover = rosetta.protocols.moves.PyMOLMover()
            pymover.keep_history(True)
            pymover.apply(min_pose)

        if fold_tree:
            ft_keys = ["ct", "nt", "m"]
            ft_values = {
                x: self._getPeptideFoldTree(
                    min_pose,
                    int(len(min_pose.residues) / 2),
                    ft_type=x,
                    peptide_chain=moving_chain,
                )
                for x in ft_keys
            }
            # Randomly define initial FoldTree
            current_ft = random.choice(ft_keys)
            min_pose.fold_tree(ft_values[current_ft])

        print("Running minimisation")
        initial_energy = sfxn(min_pose)
        print(f"\tInitial energy: {initial_energy:.2f}")
        for step in range(1, minimization_steps + 1):
            print(f"Running minimization step {step} of {minimization_steps}")
            if fold_tree:
                print(f"\tUsing FoldTree type: {current_ft}")
            fastrelax_mover.apply(min_pose)
            final_energy = sfxn(min_pose)
            d_energy = final_energy - initial_energy
            print(f"Finished minimization cycle {step} of {minimization_steps}")
            print(f"Energy change (dE):{d_energy:.2f}")
            print(f"\tCurrent energy: {final_energy:.2f}")
            if abs(d_energy) < abs(min_energy_threshold):
                print("Energy convergence achieved.")
                break
            initial_energy = final_energy

            if fold_tree:
                current_ft = random.choice(ft_keys)
                min_pose.fold_tree(ft_values[current_ft])

            if pymol:
                pymover.apply(min_pose)

        if step == minimization_steps - 1:
            print(f"Maximum number of relax iterations ({minimization_steps}) reached.")
            print("Energy convergence failed!")

        print("Finished minimisation:")

        return min_pose
