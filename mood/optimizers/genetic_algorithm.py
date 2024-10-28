import os
import random
from typing import Any, Dict, List

import numpy as np
from icecream import ic

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
        mutation_seq_percent: float = 0.5,
        # mutable_positions: List[int] = [],
        mutable_aa: Dict[int, Any] = {},
        eval_mutations: bool = False,  # Rosetta minimization mover to decide if the mutations are accepted
        eval_mutations_params: Dict[str, Any] = {},
        crossover_iterations=1,
        mutation_iterations=1,
        max_mutation_per_iteration=1,
    ) -> None:
        super().__init__(
            population_size=population_size,
            seed=seed,
            debug=debug,
            data=data,
            optimizerType=OptimizersType.GA,
        )

        # Number of mutated sequences to generate at the beginning
        self.mutation_seq_percent = mutation_seq_percent
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

        self.eval_mutations = eval_mutations
        if eval_mutations:
            self.eval_mutations_params = eval_mutations_params
            self.native_pdb = eval_mutations_params["native_pdb"]
            eval_mutations_params["cst_file"]

            import pyrosetta as prs

            if eval_mutations_params["params_folder"] != None:
                patches = [
                    eval_mutations_params["params_folder"] + "/" + x
                    for x in os.listdir(eval_mutations_params["params_folder"])
                    if x.endswith(".txt")
                ]
                params = [
                    eval_mutations_params["params_folder"] + "/" + x
                    for x in os.listdir(eval_mutations_params["params_folder"])
                    if x.endswith(".params")
                ]
                if patches == []:
                    patches = None
                if params == []:
                    raise ValueError(
                        f"Params files were not found in the given folder: {eval_mutations_params["params_folder"]}!"
                    )
            params = " ".join(params)
            patches = " ".join(patches)

            options = f"-relax:default_repeats 1 -constant_seed true -jran {eval_mutations_params["seed"]}"
            options += f" -extra_res_fa {params} -extra_patch_fa {patches}"

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

    def init_population(self, chain, sequences_initial, max_attempts=1000):
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
            else:
                if index > self.population_size * self.mutation_seq_percent:
                    self.logger.info(
                        f"Population already at {self.mutation_seq_percent * 100}%, no mutations will occur"
                    )
                else:
                    self.logger.info(
                        f"Populating {self.mutation_seq_percent * 100}% of the {self.population_size} total population"
                    )
                # Adding sequences by mutation until the desired percentage is reached
                while len(child_sequences) < self.population_size:
                    # Select a sequence to mutate, if we choose the native sequence, we will have a new sequence with only one mutation
                    # if we select a sequence from the child_sequences, we will have a new sequence with more than one mutation
                    sequence_to_start_from = self.rng.choice(sequences_initial)

                    mutated_sequence, mut = self.generate_mutation_sequence(
                        chain=chain,
                        sequence_to_mutate=sequence_to_start_from,
                        max_mutations_recomb=self.max_mutation_per_iteration,
                    )
                    # TODO may need to adapt to a more than one mutation per iteration
                    # Evaluate the mutation with rosetta
                    if self.eval_mutations:
                        dEnergy = self.local_relax(
                            residues=[mut[0][1]],
                            moving_chain=chain,
                            starting_sequence=sequence_to_start_from,
                            mutated_sequence=mutated_sequence,
                            cst_file=self.eval_mutations_params["cst_file"],
                        )
                        if abs(dEnergy) < abs(
                            self.eval_mutations_params["min_energy_threshold"]
                        ):
                            continue
                        self.logger.debug(f"Mutation energy: {dEnergy} accepted")
                    # Add the new sequence to the data object and iter_sequences
                    if (
                        not self.data.sequence_exists(chain, mutated_sequence)
                        and mutated_sequence not in child_sequences
                    ):
                        child_sequences.append(mutated_sequence)
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

                self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
                # self.logger.debug(
                #     f"Child population after mutation:\n{chr(10).join(child_sequences)}"
                # )
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

    def generate_mutation_sequence(
        self, chain, sequence_to_mutate, max_mutations_recomb=1
    ):
        sequence_to_mutate_list = list(sequence_to_mutate)
        mut = []
        if self.mutable_aa == {}:
            raise ValueError("No mutable amino acids provided")
        for _ in range(self.rng.choice(list(range(1, max_mutations_recomb + 1)))):
            position = self.rng.choice(list(self.mutable_aa[chain].keys()))
            new_aa = self.rng.choice(self.mutable_aa[chain][position])
            sequence_to_mutate_list[position] = new_aa
            mut.append((sequence_to_mutate[position], position, new_aa))

        return "".join(sequence_to_mutate_list), mut

    def mutation_on_crossover(self, parent_sequences, chain):
        child_sequences = []
        sequences_to_add = []
        attemps = 0
        while len(child_sequences) < self.population_size:
            seq_index = self.data.nsequences(chain)
            sequence_to_start_from = self.rng.choice(parent_sequences)
            child_sequence, mut = self.generate_mutation_sequence(
                chain=chain,
                sequence_to_mutate=sequence_to_start_from,
                max_mutations_recomb=self.max_mutation_per_iteration,
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
                if abs(dEnergy) < abs(
                    self.eval_mutations_params["min_energy_threshold"]
                ):
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

    def generate_child_population(
        self,
        parent_sequences,
        chain=None,
        current_iteration=None,
    ):
        child_sequences = []
        sequences_to_add = []
        cycle_pos = current_iteration % self.cycle_length

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
                child_sequence, mut = self.generate_mutation_sequence(
                    chain=chain,
                    sequence_to_mutate=sequence_to_start_from,
                    max_mutations_recomb=self.max_mutation_per_iteration,
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
                    if abs(dEnergy) < abs(
                        self.eval_mutations_params["min_energy_threshold"]
                    ):
                        continue
                    self.logger.debug(f"Mutation energy: {dEnergy} accepted")
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
            self.data.add_sequences(chain=chain, new_sequences=sequences_to_add)
            self.logger.info(f"Population on Mutation: {len(sequences_to_add)}")

        return child_sequences, sequences_to_add
