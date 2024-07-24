import random
from typing import Any, Dict, List

from Bio.Seq import MutableSeq, Seq
from icecream import ic

from mood.base.log import Logger
from mood.optimizers.optimizer import Optimizer, OptimizersType


class GeneticAlgorithm(Optimizer):
    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.005,
        init_mutation_rate: float = 0.6,
        seed: int = 12345,
        debug: bool = False,
        data: Any = None,
        mutation_seq_percent: float = 0.5,
        mutable_positions: List[int] = [],
        mutable_aa: Dict[int, Any] = {},
    ) -> None:
        super().__init__(
            population_size=population_size,
            mutation_rate=mutation_rate,
            seed=seed,
            debug=debug,
            data=data,
            optimizerType=OptimizersType.GA,
        )

        # self.population_size = population_size
        # self.mutation_rate = mutation_rate
        # # self.seed = seed
        # self.debug = debug
        # self.data = data
        self.mutation_seq_percent = mutation_seq_percent
        self.rng = random.Random(seed)
        self.mutable_positions = mutable_positions
        self.mutable_aa = mutable_aa
        self.init_mutation_rate = init_mutation_rate
        self.logger = Logger(debug).get_log()
        self.child_sequences = []
        self.crossoverTypes = ["uniform", "two_point", "single_point"]

    def init_population(self, sequences_initial):
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

        try:
            # Adding the initial sequences to the population
            for sequence in sequences_initial:
                self.data.add_sequence(sequence)
                self.child_sequences.append(sequence)

            # Getting the number of missing sequences
            n_missing = self.population_size - len(self.child_sequences)
            self.logger.debug(f"Initial child sequences: {self.child_sequences}")
            # Calculating the index of the next sequence to generate
            index = self.population_size - n_missing
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
                # Adding sequences by mutation til the desired percentage is reached
                while index < self.population_size * self.mutation_seq_percent:
                    self.logger.debug(f"Adding sequence {index} to the population")
                    # Select a sequence to mutate
                    sequence_to_start_from = self.rng.choice(self.child_sequences)

                    mutated_sequence, _, _ = self.generate_mutation_sequence(
                        sequence_to_start_from, self.init_mutation_rate
                    )
                    # Add the new sequence to the data object and iter_sequences
                    added = self.data.add_sequence(mutated_sequence)
                    while not added:
                        self.logger.warning(
                            f"Sequence {mutated_sequence} already in the data, generating a new one"
                        )
                        mutated_sequence, _, _ = self.generate_mutation_sequence(
                            sequence_to_start_from, self.init_mutation_rate
                        )
                        added = self.data.add_sequence(mutated_sequence)
                    self.child_sequences.append(mutated_sequence)

                    self.logger.debug(
                        f"Child sequences after generate mutation: \n  {self.child_sequences}"
                    )
                    index += 1
                    number_of_sequences = len(self.child_sequences)
                    self.logger.debug(
                        f"Population size: {number_of_sequences} == {index} :index"
                    )

                # Adding sequences by crossover til the desired population size is reached
                while number_of_sequences < self.population_size:
                    self.logger.debug(
                        f"Adding sequence {number_of_sequences} to the population by CrossOver"
                    )
                    # Get two random sequences to crossover
                    crossover_sequence = self.generate_crossover_sequence()
                    # Add the new sequence to the data object
                    added = self.data.add_sequence(crossover_sequence)
                    while not added:
                        self.logger.warning(
                            f"Sequence {crossover_sequence} already in the data, generating a new one"
                        )
                        crossover_sequence = self.generate_crossover_sequence()
                        added = self.data.add_sequence(crossover_sequence)
                    self.child_sequences.append(crossover_sequence)
                    self.logger.debug(
                        f"Child sequences after generate crossover: \n  {self.child_sequences}"
                    )
                    number_of_sequences = len(self.child_sequences)
                    self.logger.debug(f"Population size: {number_of_sequences}")

        except Exception as e:
            self.logger.error(f"Error initializing the population: {e}")

    def generate_mutation_sequence(self, sequence_to_mutate, mutation_rate):
        self.logger.debug("Generating a mutant sequence")
        if not self.mutable_positions:
            self.logger.error("No mutable positions provided")
            raise ValueError("No mutable positions provided")
        if not self.mutable_aa:
            self.logger.error("No mutable amino acids provided")
            raise ValueError("No mutable amino acids provided")
        old_aa = {}
        new_aa = {}
        # Transform to a mutable sequence
        mutable_seq = MutableSeq(sequence_to_mutate)
        self.logger.debug(f"Sequence_to_mutate: {sequence_to_mutate}")
        # Iterate over the aa in the mutable sequence
        for i, aa in enumerate(mutable_seq, start=1):
            # If the position is mutable and the mutation rate is met
            if (
                i in self.mutable_positions
                and i in self.mutable_aa
                and self.rng.random() <= mutation_rate
            ):
                new_residues = [nuc for nuc in self.mutable_aa[i] if nuc != aa]
                old_aa[i] = aa
                new_aa[i] = self.rng.choice(new_residues)
                mutable_seq[i - 1] = new_aa[i]
        self.logger.debug(f"Mutated_sequence: {mutable_seq}")
        return Seq(mutable_seq), old_aa, new_aa

    def generate_crossover_sequence(
        self, sequence1=None, sequence2=None, crossover_type="uniform"
    ) -> Seq:
        # If no sequence were given, select two random sequences
        if sequence1 is None and sequence2 is None:
            sequence1 = random.choice(self.child_sequences)
            sequence2 = random.choice(self.child_sequences)
            # Check that the sequences are different
            while sequence1 == sequence2:
                sequence2 = random.choice(self.child_sequences)

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
            return self.uniform_crossover(sequence1, sequence2)

    def uniform_crossover(self, sequence1, sequence2) -> Seq:
        self.logger.debug("Performing a uniform crossover")
        recombined_sequence = MutableSeq(sequence1)
        for i, aa in enumerate(sequence2, start=1):
            if i in self.mutable_positions and self.rng.random() < 0.5:
                recombined_sequence[i - 1] = aa
        self.logger.debug(f"Initial_sequences: 1.{sequence1}")
        self.logger.debug(f"Initial_sequences: 2.{sequence2}")
        self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")
        return Seq(recombined_sequence)

    def two_point_crossover(self, sequence1, sequence2, start=None, end=None) -> Seq:
        self.logger.debug("Performing a two-point crossover")
        recombined_sequence = MutableSeq(sequence1)
        if start is None:
            start = self.rng.randint(1, len(sequence1) + 1)
        if end is None:
            end = self.rng.randint(start, len(sequence1) + 1)
        for i in range(start, end + 1):
            if i in self.mutable_positions:
                recombined_sequence[i - 1] = sequence2[i - 1]
        self.logger.debug(f"Initial_sequences: 1.{sequence1}")
        self.logger.debug(f"Initial_sequences: 2.{sequence2}")
        self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")
        return Seq(recombined_sequence)

    def single_point_crossover(self, sequence1, sequence2, crossover_point=None) -> Seq:
        self.logger.debug("Performing a single-point crossover")
        recombined_sequence = MutableSeq(sequence1)
        if crossover_point is None:
            crossover_point = self.rng.randint(1, len(sequence1) + 1)
        for i in range(crossover_point, len(sequence1) + 1):
            if i in self.mutable_positions:
                recombined_sequence[i - 1] = sequence2[i - 1]
        self.logger.debug(f"Initial_sequences: 1.{sequence1}")
        self.logger.debug(f"Initial_sequences: 2.{sequence2}")
        self.logger.debug(f"  Recombined_sequence: {recombined_sequence}")
        return Seq(recombined_sequence)

    def eval_population(self):
        print("Evaluating the population")

    def generate_child_population(self, parent_sequences):
        self.logger.info("Generating the child population")
        # Initialize the child population list
        self.child_sequences = []
