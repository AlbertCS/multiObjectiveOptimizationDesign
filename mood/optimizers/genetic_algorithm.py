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
        rosetta_mover: bool = False,  # Rosetta minimization mover to decide if the mutations are accepted
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
        self.child_sequences = []

        try:
            # Adding the initial sequences to the population
            for sequence in sequences_initial:
                i = self.data.nsequences(chain) + 1
                self.data.add_sequence(
                    chain,
                    Sequence(
                        sequence=sequence,
                        chain=chain,
                        index=i,
                        active=True,
                    ),
                )
                self.child_sequences.append(sequence)

            # Getting the number of missing sequences
            n_missing = self.population_size - len(self.child_sequences)
            self.logger.debug(f"Initial child sequences: {self.child_sequences}")
            # Calculating the index of the next sequence to generate
            index = len(
                self.child_sequences
            )  # Mejor: usar el número actual de secuencias
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
                while index < self.population_size * self.mutation_seq_percent:
                    self.logger.debug(f"Adding sequence {index} to the population")
                    # Select a sequence to mutate
                    sequence_to_start_from = self.rng.choice(self.child_sequences)

                    mutated_sequence, mut = self.generate_mutation_sequence(
                        sequence_to_start_from, self.init_mutation_rate, chain
                    )
                    # Add the new sequence to the data object and iter_sequences
                    added = self.data.add_sequence(
                        chain=chain,
                        new_sequence=Sequence(
                            sequence=mutated_sequence,
                            chain=chain,
                            index=self.data.nsequences(chain) + 1,
                            active=True,
                            mutations=mut,
                            native=self.native,
                        ),
                    )
                    attempt = 1
                    while not added and attempt <= max_attempts:
                        self.logger.warning(
                            f"Sequence {mutated_sequence} already in the data, generating a new one (Attempt {attempt})"
                        )
                        mutated_sequence, mut = self.generate_mutation_sequence(
                            sequence_to_start_from, self.init_mutation_rate, chain
                        )
                        added = self.data.add_sequence(
                            chain=chain,
                            new_sequence=Sequence(
                                sequence=mutated_sequence,
                                chain=chain,
                                index=self.data.nsequences(chain) + 1,
                                active=True,
                                mutations=mut,
                                native=self.native,
                            ),
                        )
                        attempt += 1
                    if not added:
                        self.logger.error(
                            f"Failed to add a unique mutated sequence after {max_attempts} attempts"
                        )
                        raise ValueError(
                            "Failed to add a unique mutated sequence after maximum attempts"
                        )

                    # TODO: Add mover for calculating the energy of the new sequence
                    # Ejemplo de cálculo de energía
                    # pose = pyrosetta.pose_from_sequence(mutated_sequence)
                    # scorefxn = pyrosetta.get_fa_scorefxn()
                    # energy = scorefxn(pose)
                    # self.data.set_energy(
                    #     chain, mutated_sequence, energy
                    # )  # Asumiendo que hay un método para esto

                    self.child_sequences.append(mutated_sequence)

                    self.logger.debug(
                        f"Child sequences after generate mutation: \n  {self.child_sequences}"
                    )
                    index += 1
                    number_of_sequences = len(self.child_sequences)
                    self.logger.debug(
                        f"Population size: {number_of_sequences} == {index} :index"
                    )

                # Adding sequences by crossover until the desired population size is reached
                while number_of_sequences < self.population_size:
                    self.logger.debug(
                        f"Adding sequence {number_of_sequences} to the population by Crossover"
                    )
                    # Get two random sequences to crossover
                    crossover_sequence = self.generate_crossover_sequence(
                        sequences_pool=self.child_sequences, chain=chain
                    )
                    n_tries = 0
                    # Add the new sequence to the data object
                    added = self.data.add_sequence(
                        chain=chain,
                        new_sequence=Sequence(
                            sequence=crossover_sequence,
                            chain=chain,
                            index=self.data.nsequences(chain) + 1,
                            active=True,
                            mutations=None,  # Corrección: no hay mutaciones específicas
                            native=self.native,
                        ),
                    )
                    while not added and n_tries < max_attempts:
                        self.logger.warning(
                            f"Sequence {crossover_sequence} already in the data, generating a new one (Attempt {n_tries + 1})"
                        )
                        crossover_sequence = self.generate_crossover_sequence(
                            sequences_pool=self.child_sequences, chain=chain
                        )
                        added = self.data.add_sequence(
                            chain=chain,
                            new_sequence=Sequence(
                                sequence=crossover_sequence,
                                chain=chain,
                                index=self.data.nsequences(chain) + 1,
                                active=True,
                                mutations=None,  # Corrección: no hay mutaciones específicas
                                native=self.native,
                            ),
                        )
                        n_tries += 1
                    if not added:
                        self.logger.error(
                            f"Failed to add a unique crossover sequence after {max_attempts} attempts"
                        )
                        raise ValueError(
                            "Failed to add a unique crossover sequence after maximum attempts"
                        )

                    # TODO: Add mover for calculating the energy of the new sequence
                    # Ejemplo de cálculo de energía
                    # pose = pyrosetta.pose_from_sequence(crossover_sequence)
                    # scorefxn = pyrosetta.get_fa_scorefxn()
                    # energy = scorefxn(pose)
                    # self.data.set_energy(
                    #     chain, crossover_sequence, energy
                    # )  # Asumiendo que hay un método para esto

                    self.child_sequences.append(crossover_sequence)
                    # self.logger.debug(
                    #     f"Child sequences after generate crossover: \n  {self.child_sequences}"
                    # )
                    number_of_sequences = len(self.child_sequences)
                    self.logger.debug(f"Population size: {number_of_sequences}")

            return self.child_sequences
        except Exception as e:
            self.logger.error(f"Error initializing the population: {e}")
            raise ValueError(f"Error initializing the population: {e}")

    def generate_mutation_sequence(self, sequence_to_mutate, mutation_rate, chain):
        self.logger.debug("Generating a mutant sequence")
        # if not self.mutable_positions:
        #     self.logger.error("No mutable positions provided")
        #     raise ValueError("No mutable positions provided")
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
                new_residues = [nuc for nuc in self.mutable_aa[chain][i] if nuc != aa]
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

        self.logger.debug("Generating a crossover sequence")

        if crossover_type not in self.crossoverTypes:
            raise ValueError(
                f"Invalid crossover type {crossover_type}. Allowed types: {self.crossoverTypes}"
            )
        if crossover_type == "two_point":
            return self.two_point_crossover(sequence1, sequence2)
        elif crossover_type == "single_point":
            return self.single_point_crossover(sequence1, sequence2)
        else:
            return self.uniform_crossover(sequence1, sequence2, chain)

    def uniform_crossover(self, sequence1, sequence2, chain, percent_recomb=0.3) -> str:
        self.logger.debug("Performing a uniform crossover")
        recombined_sequence = list(sequence1)
        for i in self.mutable_aa[chain].keys():
            if self.rng.random() < percent_recomb:
                recombined_sequence[i - 1] = sequence2[i - 1]
            # self.logger.debug(f"Initial_sequences: 1.{sequence1}")
            # self.logger.debug(f"Initial_sequences: 2.{sequence2}")
            # self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")

        return "".join(recombined_sequence)

    def two_point_crossover(
        self, sequence1, sequence2, start=None, end=None, chain=None
    ) -> str:
        self.logger.debug("Performing a two-point crossover")
        recombined_sequence = list(sequence1)
        if start is None:
            start = self.rng.randint(1, len(sequence1) + 1)
        if end is None:
            end = self.rng.randint(start, len(sequence1) + 1)
        for i in range(start, end + 1):
            if i in self.mutable_aa[chain].keys():
                recombined_sequence[i - 1] = sequence2[i - 1]
        # self.logger.debug(f"Initial_sequences: 1.{sequence1}")
        # self.logger.debug(f"Initial_sequences: 2.{sequence2}")
        # self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")
        return "".join(recombined_sequence)

    def single_point_crossover(
        self, sequence1, sequence2, crossover_point=None, chain=None
    ) -> str:
        self.logger.debug("Performing a single-point crossover")
        recombined_sequence = list(sequence1)
        if crossover_point is None:
            crossover_point = self.rng.randint(1, len(sequence1) + 1)
        for i in range(crossover_point, len(sequence1) + 1):
            if i in self.mutable_aa[chain].keys():
                recombined_sequence[i - 1] = sequence2[i - 1]
        # self.logger.debug(f"Initial_sequences: 1.{sequence1}")
        # self.logger.debug(f"Initial_sequences: 2.{sequence2}")
        # self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")
        return "".join(recombined_sequence)

    def calculate_pareto_front(self, df, selected_columns, dimension=1):
        """Calculate the Pareto front from a DataFrame for maximization problems."""
        values = df[selected_columns].values
        pareto_front_mask = np.ones(values.shape[0], dtype=bool)
        for i in range(values.shape[0]):
            if pareto_front_mask[
                i
            ]:  # the dominated will be turned to False so no need to check
                # dominated_mask = np.sum((values >= values[i]), axis=1) >= dimension
                dominated_mask = np.sum((values <= values[i]), axis=1) >= dimension

                pareto_front_mask &= dominated_mask

        return df[pareto_front_mask]

    def calculate_non_dominated_rank(self, df, metric_states=None):
        """
        Calculate the non-dominated rank for each individual in the population.
        """
        df_to_empty = df.copy()

        df_to_empty = df_to_empty.drop(columns=["Sequence"])
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

    def rank_by_pareto(self, df, dimension):
        """
        Functions that calculates the pareto front and assigns a rank to each individual in the population.
        Returns the DataFrame with a new column rank with the rank of each individual.
        """
        df_to_empty = df.copy()
        df_final = df.copy()
        df_to_empty = df_to_empty.drop(columns=["Sequence"])
        rank = 1
        while not df_to_empty.empty:
            pareto_front = self.calculate_pareto_front(
                df_to_empty, df_to_empty.columns, dimension
            )

            # Assign rank to the Pareto front rows
            df_final.loc[pareto_front.index, "Rank"] = int(rank)

            # Remove the Pareto front rows from df_to_empty
            df_to_empty = df_to_empty.drop(pareto_front.index)

            rank += 1

        return df_final

    def rank_by_pareto_old(self, df, dimension):
        pass

    def eval_population(self, df, dimension=1, metric_states=None):
        """
        Evaluates the population to identify the parent population for the next generation.
        Returns the DataFrame with a the rank in a new column.
        """
        self.logger.info("Evaluating the population")
        # Calculate the Pareto front
        # ranked_df = self.rank_by_pareto(df, dimension)
        ranked_df = self.calculate_non_dominated_rank(df, metric_states)

        return ranked_df

    # TODO implement the following sort: https://github.com/smkalami/nsga2-in-python/blob/main/nsga2.py

    def generate_child_population(
        self,
        parent_sequences,
        max_attempts=1000,
        mutation_rate=0.06,
        chain=None,
    ):
        self.logger.info("Generating the child population")
        # Initialize the child population list
        self.child_sequences = []
        number_of_sequences = len(self.child_sequences)
        n_tries = 0
        if len(parent_sequences) < 2:
            self.logger.error(
                "Not enough parent sequences to generate a child population"
            )
            raise ValueError(
                "Not enough parent sequences to generate a child population"
            )
        mutated_sequence = None
        # Adding sequences by crossover til the desired population size is reached
        while number_of_sequences < self.population_size:
            self.logger.debug(
                f"Adding sequence {number_of_sequences} to the population by CrossOver"
            )
            # Crossover
            crossover_sequence = self.generate_crossover_sequence(
                sequences_pool=parent_sequences, chain=chain
            )
            # Add the new sequence to the data object
            added = self.data.add_sequence(
                chain=chain,
                new_sequence=Sequence(
                    sequence=crossover_sequence,
                    chain=chain,
                    index=self.data.nsequences(chain) + 1,
                    active=True,
                    mutations=None,
                    native=self.native,
                ),
            )
            while not added:
                # self.logger.warning(
                #     f"Sequence {crossover_sequence} already in the data, generating a new one"
                # )
                crossover_sequence = self.generate_crossover_sequence(
                    sequences_pool=parent_sequences, chain=chain
                )
                # TODO add mover for calculating the energy of the new sequence

                added = self.data.add_sequence(
                    chain=chain,
                    new_sequence=Sequence(
                        sequence=crossover_sequence,
                        chain=chain,
                        index=self.data.nsequences(chain) + 1,
                        active=True,
                        mutations=None,
                        native=self.native,
                    ),
                )
                # Set a warning is not possible to create new sequences by recombination
                if n_tries > max_attempts:
                    self.logger.warning("Too many tries to generate a new sequence")
                    break
                print(f"n_tries: {n_tries}")
                n_tries += 1

            # If it's not possible to generate new sequences by recombination, try to mutate
            if n_tries > max_attempts and not added:
                mutated_sequence, mut = self.generate_mutation_sequence(
                    sequence_to_mutate=crossover_sequence,
                    mutation_rate=mutation_rate,
                    chain=chain,
                )
                n_tries = 0
                added = self.data.add_sequence(
                    chain=chain,
                    new_sequence=Sequence(
                        sequence=mutated_sequence,
                        chain=chain,
                        index=self.data.nsequences(chain) + 1,
                        active=True,
                        mutations=mut,
                        native=self.native,
                    ),
                )
                while not added:
                    self.logger.warning(
                        f"Sequence {crossover_sequence} already in the data, generating a new one"
                    )
                    mutated_sequence, mut = self.generate_mutation_sequence(
                        sequence_to_mutate=crossover_sequence,
                        mutation_rate=mutation_rate,
                        chain=chain,
                    )
                    added = self.data.add_sequence(
                        chain=chain,
                        new_sequence=Sequence(
                            sequence=mutated_sequence,
                            chain=chain,
                            index=self.data.nsequences(chain) + 1,
                            active=True,
                            mutations=mut,
                            native=self.native,
                        ),
                    )
                    if n_tries > max_attempts:
                        self.logger.error("Too many tries to generate a new sequence")
                        raise RuntimeError("Too many tries to generate a new sequence")
                    n_tries += 1

            if added:
                if mutated_sequence:
                    self.child_sequences.append(mutated_sequence)
                else:
                    self.child_sequences.append(crossover_sequence)

            self.logger.debug(
                f"Child sequences after generate crossover: \n  {self.child_sequences}"
            )
            number_of_sequences = len(self.child_sequences)
            self.logger.debug(f"Population size: {number_of_sequences}")

            # Reset counters
            n_tries = 0
            mutated_sequence = None

        return self.child_sequences
