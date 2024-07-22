import logging
import random
from typing import Any, Dict, List

from Bio.Seq import MutableSeq, Seq
from icecream import ic

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

    @property
    def sequences(self):
        return self.data.sequences

    @property
    def data_frame(self):
        return self.data.data_frame

    def init_population(self, sequences_initial):
        logging.info("Initializing the population")
        try:
            if len(sequences_initial) == 0:
                logging.error("No initial sequences provided")
                raise ValueError("No initial sequences provided")
            # Getting the number of missing sequences
            n_missing = self.population_size - len(sequences_initial)
            ic(sequences_initial)
            # Calculating the index of the next sequence to generate
            index = self.population_size - n_missing
            if n_missing == 0:
                logging.info("Population already at the desired size")
            else:
                if index > self.population_size * self.mutation_seq_percent:
                    logging.info(
                        f"Population already at {self.mutation_seq_percent * 100}%, no mutations will occur"
                    )
                else:
                    logging.info(
                        f"Populating {self.mutation_seq_percent * 100}% of the {self.population_size} total population"
                    )
                # Adding sequences by mutation til the desired percentage is reached
                while (
                    index <= self.population_size * self.mutation_seq_percent
                    and index < 3
                ):
                    logging.debug(f"Adding sequence {index} to the population")
                    # Check if the input sequences of the algorithm are more than one, then select one random.
                    if len(sequences_initial) > 1:
                        sequence_to_start_from = sequences_initial[
                            self.rng.randrange(1, len(sequences_initial))
                        ]
                    else:
                        sequence_to_start_from = sequences_initial[0]
                    mutated_sequence, _, _ = self.generate_mutation_sequence(
                        sequence_to_start_from, self.init_mutation_rate
                    )
                    # Add the new sequence to the data object
                    ic(self.data.sequences)
                    added = self.data.add_sequence(mutated_sequence)
                    while not added:
                        mutated_sequence, _, _ = self.generate_mutation_sequence(
                            sequence_to_start_from, self.init_mutation_rate
                        )
                        added = self.data.add_sequence(mutated_sequence)
                    ic(self.data.sequences)
                    index += 1
                    number_of_sequences = len(self.data.sequences)
                    logging.debug(
                        f"Population size: {number_of_sequences} == {index} :index"
                    )

                # Adding sequences by crossover til the desired population size is reached
                number_of_sequences = len(self.data.sequences)
                while number_of_sequences < self.population_size:
                    logging.debug(
                        f"Adding sequence {number_of_sequences} to the population by CrossOver"
                    )
                    # Get two random sequences to crossover
                    crossover_sequence = self.generate_crossover_sequence()
                    # Add the new sequence to the data object
                    ic(self.data.sequences)
                    added = self.data.add_sequence(crossover_sequence)
                    while not added:
                        crossover_sequence = self.generate_crossover_sequence()
                        added = self.data.add_sequence(crossover_sequence)
                    ic(self.data.sequences)
                    number_of_sequences = len(self.data.sequences)
                    logging.debug(f"Population size: {number_of_sequences}")

        except Exception as e:
            logging.error(f"Error initializing the population: {e}")

    def generate_mutation_sequence(self, sequence_to_mutate, mutation_rate):
        logging.debug("Generating a mutation sequence")
        if not self.mutable_positions:
            logging.error("No mutable positions provided")
            raise ValueError("No mutable positions provided")
        if not self.mutable_aa:
            logging.error("No mutable amino acids provided")
            raise ValueError("No mutable amino acids provided")
        old_aa = {}
        new_aa = {}
        # Transform to a mutable sequence
        mutable_seq = MutableSeq(sequence_to_mutate)
        # Iterate over the aa in the mutable sequence
        for i, aa in enumerate(mutable_seq, start=1):
            # If the position is mutable and the mutation rate is met
            if i in self.mutable_positions and self.rng.random() <= mutation_rate:
                new_residues = [nuc for nuc in self.mutable_aa[i] if nuc != aa]
                old_aa[i] = aa
                new_aa[i] = random.choice(new_residues)
                mutable_seq[i - 1] = new_aa[i]
        return Seq(mutable_seq), old_aa, new_aa

    def generate_crossover_sequence(
        self, sequence1=None, sequence2=None, crossover_type="uniform"
    ) -> Seq:
        # If no sequence were given, select two random sequences
        if sequence1 is None and sequence2 is None:
            sequence_keys = list(self.data.sequences.keys())
            sequence1_id = random.choice(sequence_keys)
            sequence2_id = random.choice(sequence_keys)
            # Check that the sequences are different
            while sequence1_id == sequence2_id:
                sequence2_id = random.choice(sequence_keys)
            sequence1 = (self.data.sequences[sequence1_id],)
            sequence2 = (self.data.sequences[sequence2_id],)
        logging.debug("Generating a crossover sequence")
        crossoverTypes = ["uniform", "two_point", "single_point"]
        if crossover_type not in crossoverTypes:
            raise ValueError(
                f"Invalid crossover type {crossover_type}. Allowed types: {crossoverTypes}"
            )
        if crossover_type == "two_point":
            return self.two_point_crossover(sequence1, sequence2)
        elif crossover_type == "single_point":
            return self.single_point_crossover(sequence1, sequence2)
        else:
            return self.uniform_crossover(sequence1, sequence2)

    def uniform_crossover(self, sequence1, sequence2) -> Seq:
        logging.debug("Performing a uniform crossover")
        recombined_sequence = MutableSeq(sequence1)
        for i, aa in enumerate(sequence2):
            if i in self.mutable_positions and self.rng.random() < 0.5:
                recombined_sequence[i] = aa
        return Seq(recombined_sequence)

    def two_point_crossover(self, sequence1, sequence2) -> Seq:
        logging.debug("Performing a two-point crossover")
        recombined_sequence = MutableSeq(sequence1)
        start = self.rng.randint(0, len(sequence1))
        end = self.rng.randint(start, len(sequence1))
        for i in range(start, end):
            if i in self.mutable_positions:
                recombined_sequence[i] = sequence2[i]
        return Seq(recombined_sequence)

    def single_point_crossover(self, sequence1, sequence2) -> Seq:
        logging.debug("Performing a single-point crossover")
        recombined_sequence = MutableSeq(sequence1)
        crossover_point = self.rng.randint(0, len(sequence1))
        for i in range(crossover_point, len(sequence1)):
            if i in self.mutable_positions:
                recombined_sequence[i] = sequence2[i]
        return Seq(recombined_sequence)

    def get_sequences(self):
        print("Getting the sequences")

    def eval_population(self):
        print("Evaluating the population")

    def crossover_mutate(self):
        print("Crossing over and mutating the population")
