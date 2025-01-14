import concurrent.futures
import os
import random
from typing import Any, Dict

import numpy as np
import pandas as pd

from mood.base.log import Logger
from mood.base.sequence import Sequence
from mood.optimizers.optimizer import Optimizer, OptimizersType


class GeneticAlgorithm(Optimizer):
    def __init__(
        self,
        population_size: int = 100,
        init_mutation_rate: float = 0.1,
        seed: int = 12345,
        debug: bool = False,
        data: Any = None,
        # mutable_positions: List[int] = [],
        mutable_aa: Dict[int, Any] = {},
        eval_mutations: bool = False,  # Rosetta minimization mover to decide if the mutations are accepted
        eval_mutations_params: Dict[str, Any] = {},
        crossover_iterations=1,
        mutation_iterations=1,
        max_mutation_per_iteration=1,
        min_mutation_per_iteration=1,
        folder_name=None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            seed=seed,
            debug=debug,
            data=data,
            optimizerType=OptimizersType.GA,
        )

        self.rng = random.Random(seed)
        # self.mutable_positions = mutable_positions
        self.mutable_aa = mutable_aa
        self.init_mutation_rate = init_mutation_rate
        self.logger = Logger(debug).get_log()
        self.child_sequences = []
        self.crossoverTypes = ["uniform", "two_point", "single_point"]
        self.native = None
        self.crossover_iterations = crossover_iterations
        self.mutation_iterations = mutation_iterations
        self.cycle_length = self.crossover_iterations + self.mutation_iterations
        self.max_mutation_per_iteration = max_mutation_per_iteration
        self.min_mutation_per_iteration = min_mutation_per_iteration
        self.folder_name = folder_name
        self.sequence_to_file_frst = {}
        self.eval_mutations = eval_mutations
        self.eval_mutations_params = eval_mutations_params

    def init_pyrosetta(self):

        import pyrosetta as prs

        self.native_pdb = self.eval_mutations_params["native_pdb"]
        params = []
        patches = []
        if "cst_file" not in self.eval_mutations_params.keys():
            self.eval_mutations_params["cst_file"] = None

        if "params_folder" in self.eval_mutations_params.keys():

            if self.eval_mutations_params["params_folder"] != None and os.path.exists(
                self.eval_mutations_params["params_folder"]
            ):
                patches = [
                    self.eval_mutations_params["params_folder"] + "/" + x
                    for x in os.listdir(self.eval_mutations_params["params_folder"])
                    if x.endswith(".txt")
                ]
                params = [
                    self.eval_mutations_params["params_folder"] + "/" + x
                    for x in os.listdir(self.eval_mutations_params["params_folder"])
                    if x.endswith(".params")
                ]

        params = " ".join(params)
        patches = " ".join(patches)

        options = f"-relax:default_repeats 1 -constant_seed true -jran {self.eval_mutations_params["seed"]}"
        if params != "":
            options += f" -extra_res_fa {params}"
        if patches != "":
            options += f" -extra_patch_fa {patches}"

        prs.pyrosetta.init(options=options)

    def local_relax(
        self,
        residues=None,
        moving_chain=None,
        neighbour_distance=10,
        minimization_steps=10,
        starting_sequence=None,
        mutated_sequence=None,
        cst_file=None,
        energy_threshold=1,
    ):
        import pyrosetta as prs
        from pyrosetta import toolbox

        def mutate_native_pose(pose, seq):
            for res, aa in zip(pose.residues, seq):
                if res.name1() == "Z" and res.name1() == "X":
                    continue
                elif str(res.name1()) != str(aa):
                    toolbox.mutate_residue(pose, res.seqpos(), aa)
            return pose

        residues = residues or []
        native_pose = prs.pyrosetta.pose_from_pdb(self.native_pdb)

        sfxn = prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
            "ref2015"
        )
        sfxn_scorer = (
            prs.rosetta.core.scoring.ScoreFunctionFactory.create_score_function(
                "ref2015"
            )
        )
        atom_pair_constraint_weight = 1

        if cst_file is not None:
            sfxn.set_weight(prs.rosetta.core.scoring.ScoreType.res_type_constraint, 1)
            # Define catalytic constraints
            set_constraints = (
                prs.rosetta.protocols.constraint_movers.ConstraintSetMover()
            )
            # Add constraint file
            set_constraints.constraint_file(cst_file)
            # Add atom_pair_constraint_weight ScoreType
            sfxn.set_weight(
                prs.rosetta.core.scoring.ScoreType.atom_pair_constraint,
                atom_pair_constraint_weight,
            )
            # Turn on constraint with the mover
            set_constraints.add_constraints(True)
            set_constraints.apply(native_pose)

        starting_pose = mutate_native_pose(native_pose, starting_sequence)
        mutated_pose = mutate_native_pose(native_pose, mutated_sequence)

        fastrelax_mover = prs.rosetta.protocols.relax.FastRelax()
        fastrelax_mover.set_scorefxn(sfxn)

        if moving_chain is not None:
            chain_indexes = []
            for r in range(1, native_pose.total_residue() + 1):
                _, chain = native_pose.pdb_info().pose2pdb(r).split()
                if chain == moving_chain:
                    chain_indexes.append(r)
        else:
            chain_indexes = []

        # indexes seria mutations
        ct_chain_selector = (
            prs.rosetta.core.select.residue_selector.ResidueIndexSelector()
        )
        indexes = chain_indexes + residues
        ct_chain_selector.set_index(",".join([str(x) for x in indexes]))

        nbr_selector = (
            prs.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
        )

        # Residue of the mutation
        nbr_selector.set_focus_selector(ct_chain_selector)
        nbr_selector.set_include_focus_in_subset(
            True
        )  # This includes the peptide residues in the selection
        nbr_selector.set_distance(neighbour_distance)

        enable_mm = prs.rosetta.core.select.movemap.move_map_action(1)

        mmf_relax = prs.rosetta.core.select.movemap.MoveMapFactory()
        mmf_relax.all_bb(False)
        mmf_relax.add_bb_action(enable_mm, nbr_selector)

        ## Deactivate side-chain except for the selected chain during relax + neighbours
        mmf_relax.all_chi(False)
        mmf_relax.add_chi_action(enable_mm, nbr_selector)

        # Define RLT to prevent repacking of residues (fix side chains)
        prevent_repacking_rlt = (
            prs.rosetta.core.pack.task.operation.PreventRepackingRLT()
        )
        # Define RLT to only repack residues (movable side chains but fixed sequence)
        restrict_repacking_rlt = (
            prs.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        )

        # Prevent repacking of everything but CT, peptide and their neighbours
        prevent_subset_repacking = (
            prs.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                prevent_repacking_rlt, nbr_selector, True
            )
        )
        # Allow repacking of peptide and neighbours
        restrict_subset_to_repacking_for_sampling = (
            prs.rosetta.core.pack.task.operation.OperateOnResidueSubset(
                restrict_repacking_rlt, nbr_selector
            )
        )

        tf_sampling = prs.rosetta.core.pack.task.TaskFactory()
        tf_sampling.push_back(prevent_subset_repacking)
        tf_sampling.push_back(restrict_subset_to_repacking_for_sampling)

        fastrelax_mover.set_movemap_factory(mmf_relax)
        fastrelax_mover.set_task_factory(tf_sampling)

        initial_energy_mutated_pose = sfxn_scorer(mutated_pose)

        for step in range(1, minimization_steps + 1):
            fastrelax_mover.apply(starting_pose)
            final_energy_starting_pose = sfxn_scorer(starting_pose)
            fastrelax_mover.apply(mutated_pose)
            final_energy_mutated_pose = sfxn_scorer(mutated_pose)

            dE = final_energy_mutated_pose - initial_energy_mutated_pose
            if abs(dE) < abs(energy_threshold):
                break
            initial_energy_mutated_pose = final_energy_mutated_pose

        if step == minimization_steps - 1:
            self.logger.warning(
                f"Maximum number of relax iterations ({minimization_steps}) reached."
            )
            self.logger.warning("Energy convergence failed!")

        return final_energy_starting_pose - final_energy_mutated_pose

    def init_population(
        self, chain, sequences_initial, mutations_probabilities, max_attempts=1000
    ):
        self.logger.info("Initializing the population")
        # Initial checks
        if sequences_initial is None:
            self.logger.error("No initial sequences provided")
            raise ValueError("No initial sequences provided")
        if isinstance(sequences_initial, str):
            self.logger.info("Converting the initial sequence to a list")
            sequences_initial = [sequences_initial]
        if len(sequences_initial) == 0:
            self.logger.error("No initial sequences provided")
            raise ValueError("No initial sequences provided")
        child_sequences = []

        try:
            # Adding the initial sequences to the population
            i = self.data.nsequences(chain)
            sequences_to_add = []
            mutations = []
            starting_sequences = []
            n_seqs_added = 0
            for sequence in sequences_initial:
                sequences_to_add.append(
                    Sequence(
                        sequence=sequence,
                        chain=chain,
                        index=i,
                        active=True,
                    ),
                )
                i += 1
                child_sequences.append(sequence)
            self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)

            # Getting the number of missing sequences
            n_missing = self.population_size - len(child_sequences)
            self.logger.debug(
                f"Initial child sequences:\n{chr(10).join(child_sequences)}"
            )
            # Calculating the index of the next sequence to generate
            index = len(child_sequences)
            seq_index = self.data.nsequences(chain)
            if n_missing == 0:
                self.logger.info("Population already at the desired size")
            elif n_missing < 0:
                self.logger.error(
                    f"The initial sequences are more than the population size: {len(child_sequences)}\n Using only the first {self.population_size} sequences"
                )
                self.logger.error(
                    f"Using only the first {self.population_size} sequences"
                )
                child_sequences = child_sequences[: self.population_size]
            else:
                # Adding sequences by mutation until the desired percentage is reached
                while len(child_sequences) < self.population_size:
                    # Select a sequence to mutate, if we choose the native sequence, we will have a new sequence with only one mutation
                    # if we select a sequence from the child_sequences, we will have a new sequence with more than one mutation
                    sequence_to_start_from = self.rng.choice(sequences_initial)

                    mutated_sequence, mut = self.generate_mutation_sequence(
                        chain=chain,
                        sequence_to_mutate=sequence_to_start_from,
                        min_mutations=self.min_mutation_per_iteration,
                        max_mutations=self.max_mutation_per_iteration,
                        mutations_probabilities=mutations_probabilities,
                    )
                    # TODO may need to adapt to a more than one mutation per iteration
                    # Add the new sequence to the data object and iter_sequences
                    if (
                        not self.data.sequence_exists(chain, mutated_sequence)
                        and mutated_sequence not in child_sequences
                    ):
                        child_sequences.append(mutated_sequence)
                        mutations.append(mut)
                        starting_sequences.append(sequence_to_start_from)
                        sequences_to_add.append(
                            Sequence(
                                sequence=mutated_sequence,
                                chain=chain,
                                index=seq_index,
                                active=True,
                                mutations=mut,
                                native=self.native,
                            ),
                        )
                        seq_index += 1

                    if (
                        len(child_sequences) == self.population_size
                        and self.eval_mutations
                    ):
                        # Evaluate the mutation with rosetta
                        self.logger.info("Evaluating the mutations")
                        child_sequences, mutations, starting_sequences, n_seqs_added = (
                            self.evaluate_mutations_on_sequence(
                                mutated_sequences=child_sequences,
                                mutations=mutations,
                                chain=chain,
                                starting_sequences=starting_sequences,
                                n_seqs_added=n_seqs_added,
                            )
                        )
                        print(f"Number of sequences added: {n_seqs_added}")

                self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
                self.logger.info(
                    f"Population initialized with mutation: {len(child_sequences)}"
                )

            return child_sequences, sequences_to_add
        except Exception as e:
            self.logger.error(f"Error initializing the population: {e}")
            raise ValueError(f"Error initializing the population: {e}")

    def generate_mutation_sequence_old(self, sequence_to_mutate, mutation_rate, chain):
        self.logger.debug("Generating a mutant sequence")
        if not self.mutable_aa:
            self.logger.error("No mutable amino acids provided")
            raise ValueError("No mutable amino acids provided")
        new_aa = {}
        mut = []
        # Transform to a mutable sequence
        mutable_seq = list(sequence_to_mutate)
        self.logger.debug(f"Sequence_to_mutate: {sequence_to_mutate}")
        # Iterate over the aa in the mutable sequence
        for i, aa in enumerate(mutable_seq, start=0):
            # If the position is mutable and the mutation rate is met
            rng = self.rng.random()
            if i in self.mutable_aa[chain] and rng <= mutation_rate:
                new_residues = self.mutable_aa[chain][i]
                new_aa[i] = self.rng.choice(new_residues)
                # abans hi havia un mutable_seq[i - 1]
                mutable_seq[i] = new_aa[i]
                mut.append((aa, i, new_aa[i]))
        self.logger.debug(f"Mutated_sequence: {''.join(mutable_seq)}")
        return "".join(mutable_seq), mut

    def generate_crossover_sequence(
        self,
        sequence1=None,
        sequence2=None,
        crossover_type="uniform",
        sequences_pool=None,
        chain=None,
    ) -> str:
        # If no sequence were given, select two random sequences
        if sequence1 is None and sequence2 is None:
            sequence1 = random.choice(sequences_pool)
            sequence2 = random.choice(sequences_pool)
            # Check that the sequences are different
            while sequence1 == sequence2:
                sequence2 = random.choice(sequences_pool)

        if crossover_type not in self.crossoverTypes:
            raise ValueError(
                f"Invalid crossover type {crossover_type}. Allowed types: {self.crossoverTypes}"
            )
        if crossover_type == "two_point":
            return self.two_point_crossover(
                sequence1=sequence1, sequence2=sequence2, chain=chain
            )
        elif crossover_type == "single_point":
            return self.single_point_crossover(
                sequence1=sequence1, sequence2=sequence2, chain=chain
            )
        else:
            return self.uniform_crossover(
                sequence1=sequence1, sequence2=sequence2, chain=chain
            )

    def uniform_crossover(
        self,
        sequence1,
        sequence2,
        chain,
        percent_recomb=0.5,
    ) -> str:
        recombined_sequence = list(sequence1)

        mut = []
        for i in self.mutable_aa[chain].keys():
            if recombined_sequence[i] != sequence2[i]:
                if self.rng.random() < percent_recomb:
                    recombined_sequence[i] = sequence2[i]
                    mut.append((sequence1[i], i, sequence2[i]))

        return "".join(recombined_sequence)

    def two_point_crossover(
        self, sequence1, sequence2, start=None, end=None, chain=None
    ) -> str:
        self.logger.debug("Performing a two-point crossover")
        recombined_sequence = list(sequence1)
        if start is None:
            start = self.rng.randint(0, len(sequence1) + 1)
        if end is None:
            end = self.rng.randint(start, len(sequence1) + 1)
        for i in range(start, end + 1):
            if i in self.mutable_aa[chain].keys():
                recombined_sequence[i - 1] = sequence2[i - 1]
        return "".join(recombined_sequence)

    def single_point_crossover(
        self, sequence1, sequence2, crossover_point=None, chain=None
    ) -> str:
        self.logger.debug("Performing a single-point crossover")
        recombined_sequence = list(sequence1)
        if crossover_point is None:
            crossover_point = self.rng.randint(0, len(sequence1) + 1)
        for i in range(crossover_point, len(sequence1) + 1):
            if i in self.mutable_aa[chain].keys():
                recombined_sequence[i - 1] = sequence2[i - 1]
        return "".join(recombined_sequence)

    def non_dominated_sorting(df, maximize_metrics):
        """
        Perform non-dominated sorting on a DataFrame with multiple metrics.

        Parameters:
        - df (pd.DataFrame): DataFrame where each column is a metric to evaluate.
        - maximize_metrics (dict): A dictionary where keys are column names and values are booleans
                                indicating whether to maximize (True) or minimize (False) the corresponding metric.

        Returns:
        - list of lists: Each sublist represents a Pareto front, with the best front first.
        """
        n = len(df)
        fronts = [[] for _ in range(n)]
        domination_counts = [0] * n  # Number of times each point is dominated
        dominated_points = [[] for _ in range(n)]  # Points each point dominates
        ranks = [0] * n

        # Convert DataFrame index to list for easy access
        indices = df.index.tolist()

        for i in range(n):
            for j in range(n):
                if i != j:
                    a = df.iloc[i]
                    b = df.iloc[j]
                    dominates = True
                    strictly_better = False

                    for metric, maximize in maximize_metrics.items():
                        if maximize:
                            if a[metric] < b[metric]:
                                dominates = False
                            elif a[metric] > b[metric]:
                                strictly_better = True
                        else:
                            if a[metric] > b[metric]:
                                dominates = False
                            elif a[metric] < b[metric]:
                                strictly_better = True

                    # Check if point `i` dominates point `j`
                    if dominates and strictly_better:
                        dominated_points[i].append(j)
                        domination_counts[j] += 1

        # Identify the first front (non-dominated points)
        current_front = [i for i in range(n) if domination_counts[i] == 0]
        front_index = 0

        while current_front:
            next_front = []
            for p in current_front:
                ranks[p] = front_index + 1
                for dominated in dominated_points[p]:
                    domination_counts[dominated] -= 1
                    if domination_counts[dominated] == 0:
                        next_front.append(dominated)
            fronts[front_index] = [indices[idx] for idx in current_front]
            current_front = next_front
            front_index += 1

        # Remove empty fronts and return
        return [front for front in fronts if front]

    def calculate_non_dominated_rank(self, df, metric_states=None, objectives=None):
        """
        Calculate the non-dominated rank for each individual in the population.
        """
        df_to_empty = df.copy()

        df_to_empty = df_to_empty.drop(columns=["Sequence"])

        # for the df_to_empty, only keep the columns in the objectives list
        df_to_empty = df_to_empty[objectives]

        population_size = df_to_empty.shape[0]

        # Changes the values by state
        # Precompute the columns that need to be negated
        columns_to_negate = [s for s in metric_states if metric_states[s] == "Positive"]
        # Apply the negation using vectorized operations
        df_to_empty[columns_to_negate] = df_to_empty[columns_to_negate].map(
            lambda x: -x
        )

        values = df_to_empty.values
        ranks = np.zeros(population_size, dtype=int)

        for i in range(population_size):
            rank = 1
            for j in range(population_size):
                if (
                    i != j
                    and np.all(values[j] <= values[i])
                    and np.any(values[j] < values[i])
                ):
                    rank += 1
            ranks[i] = rank

        df["Ranks"] = ranks

        return df

    def eval_population(self, df, metric_states=None, objectives=None):
        """
        Evaluates the population to identify the parent population for the next generation.
        Returns the DataFrame with a the rank in a new column.
        """
        self.logger.info("Evaluating the population")
        # Calculate the Pareto front
        # ranked_df = self.rank_by_pareto(df, dimension)
        # TODO filter the df to only contain objective metrics
        ranked_df = self.calculate_non_dominated_rank(
            df=df, metric_states=metric_states, objectives=objectives
        )

        return ranked_df

    # TODO implement the following sort: https://github.com/smkalami/nsga2-in-python/blob/main/nsga2.py

    def add_frustration_files(
        self,
    ):
        frustrar_folder = (
            f"{self.folder_name}/{str(self.current_iteration -1).zfill(3)}/frustrar"
        )
        if os.path.exists(frustrar_folder):
            equivalence_file = f"{frustrar_folder}/results/equivalences.csv"
            equivalences = pd.read_csv(equivalence_file)
            # Save the equivalences
            for row in equivalences.iterrows():
                self.sequence_to_file_frst[row[1]["Sequence"]] = (
                    f"{frustrar_folder}/results/{row[1]['Names']}_singleresidue.csv"
                )

    def add_frustrationBias_to_mutations(self, sequence_to_mutate, chain):

        frustrar_folder = (
            f"{self.folder_name}/{str(self.current_iteration -1).zfill(3)}/frustrar"
        )
        # if the frustration calculation exists, add the frustration bias to the mutations
        if os.path.exists(frustrar_folder):

            if sequence_to_mutate in self.sequence_to_file_frst.keys():
                single_frst = pd.read_csv(
                    self.sequence_to_file_frst[sequence_to_mutate]
                )
            else:
                self.logger.error("No name found for the sequence")

            frst_index = single_frst["FrstIndex"]
            # Normalize the values of frst index, so they go between 0 and 1
            frst_index = 1 - (frst_index - frst_index.min()) / (
                frst_index.max() - frst_index.min()
            )
        else:
            frst_index = [1] * len(self.mutable_aa[chain].keys())

        return list(frst_index)

    def generate_mutation_sequence(
        self,
        chain,
        sequence_to_mutate,
        min_mutations=1,
        max_mutations=1,
        mutations_probabilities=None,
    ):
        sequence_to_mutate_list = list(sequence_to_mutate)
        mut = []

        mutable_positions = list(self.mutable_aa[chain].keys())
        # generate a list of 1 for each mutable position

        mutable_positions_probability = self.add_frustrationBias_to_mutations(
            sequence_to_mutate, chain
        )

        if self.mutable_aa == {}:
            raise ValueError("No mutable amino acids provided")
        for _ in range(self.rng.choice(list(range(min_mutations, max_mutations + 1)))):
            try:
                position = self.rng.choices(
                    mutable_positions, mutable_positions_probability, k=1
                )[0]
            except Exception as e:
                self.logger.error(f"Error selecting the position: {e}")
            if mutations_probabilities is None:
                new_aa = self.rng.choice(self.mutable_aa[chain][position])
            else:
                if position not in mutations_probabilities[chain].keys():
                    self.logger.error(
                        f"Position {position} not in the mutation probabilities"
                    )
                    self.logger.error(
                        f"Available probabilities: {mutations_probabilities[chain].keys()}"
                    )
                    self.logger.error(
                        f"Available positions: {self.mutable_aa[chain].keys()}"
                    )
                else:
                    new_aa = self.rng.choices(
                        self.mutable_aa[chain][position],
                        mutations_probabilities[chain][position],
                        k=1,
                    )[0]
            sequence_to_mutate_list[position] = new_aa
            mut.append((sequence_to_mutate[position], position, new_aa))

        return "".join(sequence_to_mutate_list), mut

    def mutation_on_crossover(self, parent_sequences, chain, mutations_probabilities):
        child_sequences = []
        sequences_to_add = []
        attemps = 0
        while len(child_sequences) < self.population_size:
            seq_index = self.data.nsequences(chain)
            sequence_to_start_from = self.rng.choice(parent_sequences)
            child_sequence, mut = self.generate_mutation_sequence(
                chain=chain,
                sequence_to_mutate=sequence_to_start_from,
                min_mutations=self.min_mutation_per_iteration,
                max_mutations_=self.max_mutation_per_iteration,
                mutations_probabilities=mutations_probabilities,
            )
            # TODO may need to adapt to a more than one mutation per iteration
            # Evaluate the mutation with rosetta
            if self.eval_mutations:
                dEnergy = self.local_relax(
                    residues=[mut[0][1]],
                    moving_chain=chain,
                    starting_sequence=sequence_to_start_from,
                    mutated_sequence=child_sequence,
                    cst_file=self.eval_mutations_params["cst_file"],
                )
                if dEnergy < self.eval_mutations_params["min_energy_threshold"]:
                    continue
            # If the sequence does not exist, add it to the list of sequences to add
            if (
                not self.data.sequence_exists(chain, child_sequence)
                and child_sequence not in child_sequences
            ):
                child_sequences.append(child_sequence)
                sequences_to_add.append(
                    Sequence(
                        sequence=child_sequence,
                        chain=chain,
                        index=seq_index,
                        active=True,
                        mutations=mut,
                        native=self.native,
                    ),
                )
                seq_index += 1
            if attemps > 1000:
                self.logger.error(
                    f"Exceeded the number of attempts to generate a new sequence"
                )
                raise ValueError(
                    f"Exceeded the number of attempts to generate a new sequence"
                )
            attemps += 1
        return sequences_to_add

    # Function to evaluate a single mutation
    def evaluate_single_mutation(self, arg):
        self.init_pyrosetta()
        dEnergy = 0
        child_sequence, mut, sequence_to_start_from, chain = arg
        if not os.path.exists("debug"):
            os.makedirs("debug", exist_ok=True)
        try:
            dEnergy = self.local_relax(
                residues=[mut[0][1] + 1],
                moving_chain=chain,
                starting_sequence=sequence_to_start_from,
                mutated_sequence=child_sequence,
                cst_file=self.eval_mutations_params["cst_file"],
            )
        except Exception as e:
            with open(f"debug/{chain}_{child_sequence}.txt", "w") as f:
                f.write(f"*** Evaluating mutation ***\n")
                f.write(f"Child sequence: {child_sequence}\n")
                f.write(f"Mutation: {mut}\n")
                f.write(f"Sequence to start from: {sequence_to_start_from}\n")
                f.write(f"Chain: {chain}\n")
                f.write(f"dEnergy: {dEnergy}\n")
                f.write(f"Error: {e}\n")
                f.write(f"*** Done envaluating mutation ***\n")

        return child_sequence, mut, sequence_to_start_from, dEnergy

    def evaluate_mutations_on_sequence(
        self,
        mutated_sequences,
        mutations,
        chain,
        starting_sequences,
        n_seqs_added=0,
    ):
        """
        Evaluate mutations on a sequence and retain the ones that meet energy criteria.

        Parameters:
        - mutated_sequences: List of mutated sequences to evaluate.
        - mutations: List of mutations applied to generate mutated_sequences.
        - chain: The chain on which mutations are being evaluated.
        - starting_sequences: List of sequences used as a basis for mutations.
        - n_seqs_added: Number of sequences that were already evaluated.

        Returns:
        - better_sequences: List of sequences that pass the energy criteria.
        - better_sequences_mut: Corresponding mutations for better_sequences.
        - better_sequences_starting: Starting sequences for better_sequences.
        - num_better_sequences: Number of sequences that passed the energy criteria.
        """
        # Initialize lists to store sequences that meet the criteria
        better_sequences = []
        better_sequences_mut = []
        better_sequences_starting = []

        # Prepare the arguments for parallel execution
        args_list = list(
            zip(
                mutated_sequences[n_seqs_added:],
                mutations[n_seqs_added:],
                starting_sequences[n_seqs_added:],
                [chain] * len(mutated_sequences[n_seqs_added:]),
            )
        )

        # Determine the number of threads
        num_threads = os.cpu_count() - 1

        # Use ProcessPoolExecutor to parallelize the evaluation
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads
        ) as executor:
            results = list(executor.map(self.evaluate_single_mutation, args_list))

        # Process the results
        for child_sequence, mut, sequence_to_start_from, dEnergy in results:
            # If the energy is below the threshold, skip this sequence
            if dEnergy < self.eval_mutations_params["min_energy_threshold"]:
                continue

            # Store the sequences that meet the energy threshold
            better_sequences.append(child_sequence)
            better_sequences_mut.append(mut)
            better_sequences_starting.append(sequence_to_start_from)
            self.logger.debug(f"Mutation energy: {dEnergy} accepted")

        # Return only the sequences that meet the criteria
        return (
            mutated_sequences[:n_seqs_added] + better_sequences,
            mutations[:n_seqs_added] + better_sequences_mut,
            starting_sequences[:n_seqs_added] + better_sequences_starting,
            len(mutated_sequences[:n_seqs_added]) + len(better_sequences),
        )

    def generate_child_population(
        self,
        parent_sequences,
        chain=None,
        current_iteration=None,
        mutations_probabilities=None,
    ):
        self.current_iteration = current_iteration
        child_sequences = []
        sequences_to_add = []
        mutations = []
        starting_sequences = []
        n_seqs_added = 0
        cycle_pos = current_iteration % self.cycle_length

        self.add_frustration_files()

        if cycle_pos < self.crossover_iterations:
            self.logger.info(f"Iteration {current_iteration} - Crossover")
            general_attempt = 0
            same_parents_attempts = 0
            while len(child_sequences) < self.population_size:
                seq_index = self.data.nsequences(chain)
                # Select the parents sequences
                sequence1 = random.choice(parent_sequences)
                sequence2 = random.choice(parent_sequences)
                # Check that the sequences are different
                while sequence1 == sequence2:
                    sequence2 = random.choice(parent_sequences)
                while same_parents_attempts < 10:
                    child_sequence = self.generate_crossover_sequence(
                        sequence1=sequence1, sequence2=sequence2, chain=chain
                    )
                    # If the sequence does not exist, add it to the list of sequences to add
                    if (
                        not self.data.sequence_exists(chain, child_sequence)
                        and child_sequence not in child_sequences
                    ):
                        child_sequences.append(child_sequence)
                        sequences_to_add.append(
                            Sequence(
                                sequence=child_sequence,
                                chain=chain,
                                index=seq_index,
                                active=True,
                                mutations=None,
                                native=self.native,
                            ),
                        )
                        seq_index += 1
                        general_attempt = 0
                        same_parents_attempts = 0
                        break

                    same_parents_attempts += 1

                if general_attempt > 100:
                    # If the number of attempts to generate a new sequence is exceeded, switch to mutation
                    self.logger.info(
                        f"Exceeded the number of attempts to generate a new sequence, switching to mutation"
                    )
                    sequences_to_add = self.mutation_on_crossover(
                        parent_sequences, chain
                    )
                    break

                general_attempt += 1
                same_parents_attempts = 0

            # Add sequences tot the data object
            self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
            self.logger.info(f"Population on crossover: {len(sequences_to_add)}")

        else:
            self.logger.info(f"Iteration {current_iteration} - Mutation")
            attemps = 0
            while len(child_sequences) < self.population_size:
                seq_index = self.data.nsequences(chain)
                sequence_to_start_from = self.rng.choice(parent_sequences)
                self.parent_sequences = parent_sequences
                child_sequence, mut = self.generate_mutation_sequence(
                    chain=chain,
                    sequence_to_mutate=sequence_to_start_from,
                    min_mutations=self.min_mutation_per_iteration,
                    max_mutations=self.max_mutation_per_iteration,
                    mutations_probabilities=mutations_probabilities,
                )
                # TODO may need to adapt to a more than one mutation per iteration
                # If the sequence does not exist, add it to the list of sequences to add
                if (
                    not self.data.sequence_exists(chain, child_sequence)
                    and child_sequence not in child_sequences
                ):
                    child_sequences.append(child_sequence)
                    mutations.append(mut)
                    starting_sequences.append(sequence_to_start_from)
                    sequences_to_add.append(
                        Sequence(
                            sequence=child_sequence,
                            chain=chain,
                            index=seq_index,
                            active=True,
                            mutations=mut,
                            native=self.native,
                        ),
                    )
                    seq_index += 1
                if attemps > 1000:
                    self.logger.error(
                        f"Exceeded the number of attempts to generate a new sequence"
                    )
                    raise ValueError(
                        f"Exceeded the number of attempts to generate a new sequence"
                    )
                attemps += 1
                if len(child_sequences) == self.population_size and self.eval_mutations:
                    # Evaluate the mutation with rosetta
                    self.logger.info("Evaluating the mutations")
                    child_sequences, mutations, starting_sequences, n_seqs_added = (
                        self.evaluate_mutations_on_sequence(
                            mutated_sequences=child_sequences,
                            mutations=mutations,
                            chain=chain,
                            starting_sequences=starting_sequences,
                            n_seqs_added=n_seqs_added,
                        )
                    )
                    print(f"Number of sequences added: {n_seqs_added}")

            self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
            self.logger.info(f"Population on Mutation: {len(sequences_to_add)}")

        return child_sequences, sequences_to_add

    def generate_child_population_with_recombination(
        self,
        parent_sequences,
        chain=None,
        current_iteration=None,
        mutations_probabilities=None,
        mutation_rate=0.4,
    ):
        child_sequences = []
        sequences_to_add = []

        self.logger.info(f"Iteration {current_iteration} - Crossover")
        general_attempt = 0
        same_parents_attempts = 0
        while len(child_sequences) < self.population_size:
            seq_index = self.data.nsequences(chain)
            # Select the parents sequences
            sequence1 = random.choice(parent_sequences)
            sequence2 = random.choice(parent_sequences)
            # Check that the sequences are different
            while sequence1 == sequence2:
                sequence2 = random.choice(parent_sequences)
            while same_parents_attempts < 10:
                child_sequence = self.generate_crossover_sequence(
                    sequence1=sequence1, sequence2=sequence2, chain=chain
                )
                if self.rng.random() < mutation_rate:
                    # See if a 50% of probability to mutate is too much
                    child_sequence = self.generate_mutation_sequence(
                        chain=chain,
                        sequence_to_mutate=child_sequence,
                        min_mutations=self.min_mutation_per_iteration,
                        max_mutations=self.max_mutation_per_iteration,
                        mutations_probabilities=mutations_probabilities,
                    )
                # If the sequence does not exist, add it to the list of sequences to add
                if (
                    not self.data.sequence_exists(chain, child_sequence)
                    and child_sequence not in child_sequences
                ):
                    child_sequences.append(child_sequence)
                    sequences_to_add.append(
                        Sequence(
                            sequence=child_sequence,
                            chain=chain,
                            index=seq_index,
                            active=True,
                            mutations=None,
                            native=self.native,
                        ),
                    )
                    seq_index += 1
                    general_attempt = 0
                    same_parents_attempts = 0
                    break

                same_parents_attempts += 1

            if general_attempt > 100:
                # If the number of attempts to generate a new sequence is exceeded, switch to mutation
                self.logger.info(
                    f"Exceeded the number of attempts to generate a new sequence, switching to mutation"
                )
                sequences_to_add = self.mutation_on_crossover(parent_sequences, chain)
                break

            general_attempt += 1
            same_parents_attempts = 0

        # Add sequences tot the data object
        self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
        self.logger.info(f"Population on crossover: {len(sequences_to_add)}")

        return child_sequences, sequences_to_add
