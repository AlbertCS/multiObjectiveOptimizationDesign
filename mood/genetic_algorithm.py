import datetime
import logging
import os
import random
import shutil
import subprocess
from math import floor

import numpy as np
import pandas as pd

from .base import Sequence, Silent
from .utils import analysis_geneticAlgorithm


class GeneticAlgorithm:

    # Good
    def __init__(
        self,
        job_folder="GA",
        sequences={},
        active_chains=[],
        population_size=100,
        mutation_rate=0.005,
        mutable_positions=None,
        elites_fraction=0.00,
        max_mutations=None,
        iteration=0,
        seed=None,
        debug=False,
    ):
        self.folders = {}
        self.folders["job"] = job_folder
        self.sequences = sequences
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elites_fraction = elites_fraction
        self.max_mutations = max_mutations
        self.seed = int(seed)
        self.rng = random.Random()
        self.rng.seed(seed)
        self.debug = debug
        self.iteration = iteration
        self.active_chains = active_chains
        self.mutable_positions = mutable_positions

        logging.basicConfig(
            filename=f"msd_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
            format="%(asctime)s %(message)s",
            filemode="w",
        )
        logger = logging.getLogger(__name__)
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        msg = f"Folders: {self.folders['job']}\n"
        msg += f"Population size: {self.population_size}\n"
        msg += f"Max mutations: {self.max_mutations}\n"
        msg += f"Mutation rate: {self.mutation_rate}\n"
        msg += f"Seed: {self.seed}\n"
        logger.info(msg)

        self.mutation_rate = {}
        self.max_mutations = {}
        for chain in self.active_chains:
            if mutation_rate is None:
                self.mutation_rate[chain] = 1 / len(mutable_positions[chain])
            else:
                self.mutation_rate[chain] = mutation_rate

            # If not given, set max mutation constraints to the length of mutable positions
            # This is equivalent to use the code without the max_mutations constraint.
            if max_mutations is None:
                self.max_mutations[chain] = len(mutable_positions[chain])
            elif isinstance(max_mutations, int):
                self.max_mutations[chain] = max_mutations

    # Good
    def _deactivateParents(self):
        for chain in self.sequences:
            for index in self.sequences[chain]:
                sequence = self.sequences[chain][index]
                if sequence.active and sequence.parent:
                    sequence.active = False

    def _select_parents_by_fitness(self):
        return 0

    def runGeneticAlgorithm(
        self,
        mutated_seq_pct=0.5,
    ):
        print("Seed is:", self.seed)

        if self.iteration == 0:
            for chain in self.active_chains:
                # Check if the population size is enough to start the genetic algorithm
                n_missing = self.population_size - len(self.sequences)
                if n_missing >= 0:
                    print(f"Missing {n_missing} sequences for chain {chain}")
                    print("Creating the first parent population")
                    self._create_first_population(
                        self.sequences, n_missing, mutated_seq_pct
                    )
                else:
                    print(
                        "Warning: more sequences (%s) than the population size were found for chain %s"
                        % (-n_missing)
                    )

        parents = self._select_parents_by_fitness()

        # Create the population by recombination
        applied_mutations = self.create_population_by_recombination()

        # Mutate the population
        self.mutatePopulation()

        return self.sequences

    # Good
    def _get_sequences_pool(
        self, chain, active=None, recombined=None, mutated=None, parent=None, child=None
    ):
        """
        Get a pool of sequences in the population of a specific chain as a dict
        of sequence strings or msdSequence objects. The keywords active, recombined,
        mutated, parent, and child can be used to select intersection of sequence
        populations. If the keyword is True it will select all sequences with the
        property turn to True, and vice versa if the keyword is False.

        Parameters
        ==========
        chain : str
            PDB ID of the active sequences to get.
        active : bool
            Do we select for active (True) or inactive (False) sequences?
        recombined : bool
            Do we select for recombined (True) or not recombined (False) sequences?
        mutated : bool
            Do we select for mutated (True) or not mutated (False) sequences?
        parent : bool
            Do we select for parent (True) or not parent (False) sequences?
        child : bool
            Do we select for child (True) or not child (False) sequences?

        Returns
        =======
        sequences_pool: dict
            Dictionary of selected sequences.

        """
        sequences_pool = {}
        for index in self.sequences[chain]:
            sequence = self.sequences[chain][index]

            # Check for active sequences
            if isinstance(active, bool):
                if active:
                    if not sequence.active:
                        continue
                else:
                    if sequence.active:
                        continue

            # Check for recombined sequences
            if isinstance(recombined, bool):
                if recombined:
                    if not sequence.recombined:
                        continue
                else:
                    if sequence.recombined:
                        continue

            # Check for mutated sequences
            if isinstance(mutated, bool):
                if mutated:
                    if not sequence.mutated:
                        continue
                else:
                    if sequence.mutated:
                        continue

            # Check for parent sequences
            if isinstance(parent, bool):
                if parent:
                    if not sequence.parent:
                        continue
                else:
                    if sequence.parent:
                        continue

            # Check for child sequences
            if isinstance(child, bool):
                if child:
                    if not sequence.child:
                        continue
                else:
                    if sequence.child:
                        continue

            sequences_pool[index] = sequence

        return sequences_pool

    def create_population_by_recombination(
        self, max_attempts=1000, verbose=True
    ) -> bool:

        # Check how many sequence to add when creating a new population
        # before there was a len(self.msd.getSequencesPool(chain, active=True, child=True) + 1

        if self.iteration == 0:
            skip_assert = True
        else:
            skip_assert = False

        for chain in self.active_chains:
            n_recombinations = self.population_size - (
                len(self._get_sequences_pool(chain, active=True, child=True))
            )
            if n_recombinations == 0:
                print("Population is full, no recombination steps will be carried out.")
                return

            # Gather pool of sequences only from parent sequences
            if self.iteration == 0:
                sequence_pool_child = self._get_sequences_pool(
                    chain, active=True, child=True
                )
                sequence_pool_parent = self._get_sequences_pool(
                    chain, active=True, parent=True
                )
                sequence_pool = {**sequence_pool_parent, **sequence_pool_child}
                sequence_pool_indexes = list(sequence_pool.keys())
            else:
                sequence_pool = self._get_sequences_pool(
                    chain, active=True, parent=True
                )
                sequence_pool_indexes = list(sequence_pool.keys())

            # Fill GA population by recombination of parent sequences
            recombined_parents = []
            n_recombined_children = len(
                self._get_sequences_pool(
                    chain, active=True, child=True, recombined=True
                )
            )
            n_tries = 0
            apply_mutations = False
            while n_recombined_children < n_recombinations:

                for i in range(n_recombinations - n_recombined_children):

                    # Select first parent to recombine
                    sequences_ids = [
                        x for x in sequence_pool_indexes if x not in recombined_parents
                    ]
                    sequence1_id = self.rng.choice(sequences_ids)
                    recombined_parents.append(sequence1_id)

                    # Empty recombined parent list if it is full
                    if len(recombined_parents) == len(sequence_pool_indexes):
                        recombined_parents = []

                    # Select second parent to recombine
                    sequences_ids = [
                        x for x in sequence_pool_indexes if x not in recombined_parents
                    ]
                    sequence2_id = self.rng.choice(sequences_ids)
                    recombined_parents.append(sequence2_id)

                    # Empty recombined parent list if it is full
                    if len(recombined_parents) == len(sequence_pool_indexes):
                        recombined_parents = []

                    # Skip recombination if both indexes are equal
                    if sequence1_id == sequence2_id:
                        continue

                    # Recombine the two sequences to obtain a new child sequence
                    sequence1 = sequence_pool[sequence1_id]
                    sequence2 = sequence_pool[sequence2_id]
                    if verbose:
                        message = "\tRecombining sequence %s and sequence %s" % (
                            sequence1_id,
                            sequence2_id,
                        )
                        print(message)
                    child_sequence = self._recombine_sequences(
                        sequence1, sequence2, skip_assert=skip_assert
                    )

                    if apply_mutations:
                        exists = True
                        while exists:
                            # logger.debug(
                            #    f"mutationRecombination child: {len(self.msd.getSequencesPool('A', active=True, child=True))}")
                            # Mutate sequence
                            mutated_sequence, oaa, naa = self._mutate_sequence(
                                child_sequence, return_aa_info=True
                            )
                            mutated_sequence.recombined_mutated = True
                            reverted = False
                            # Check reversions for the mutated sequence
                            n_reversions = max(
                                0,
                                len(mutated_sequence.mutations) - self.max_mutations,
                            )
                            if n_reversions > 0:
                                reverted_sequence = self._revert_sequence(
                                    mutated_sequence
                                )
                                mutated_sequence = reverted_sequence
                                mutated_sequence.reverted = True
                                reverted = True

                            exists = self._check_if_sequenceExists(mutated_sequence)

                        if verbose:
                            mutation_label = ",".join(
                                [oaa[i] + str(i) + naa[i] for i in oaa]
                            )
                            if mutation_label != "":
                                print(
                                    f"\tSequence with index {mutated_sequence.index} was mutated to {mutation_label}"
                                )
                            if reverted and n_reversions == 1:
                                print(
                                    f"\t{n_reversions} mutation was randomly reverted for sequence {mutated_sequence.index} to reach the maximum number of mutations"
                                )
                            elif reverted:
                                print(
                                    f"\t{n_reversions} mutations were randomly reverted for sequence {mutated_sequence.index} to reach the maximum number of mutations"
                                )
                        child_sequence = mutated_sequence

                    # Add child sequences to MSD (repeat until solutions are unique)
                    self._add_sequences(
                        child_sequence.sequence,
                        chain,
                        child=True,
                        recombined=True,
                        verbose=False,
                    )

                # Count added sequences
                n_recombined_children = len(
                    self._get_sequences_pool(
                        chain, active=True, child=True, recombined=True
                    )
                )

                n_tries += 1

                # logger.debug(f"Recom: {n_recombined_children}")
                # logger.debug(f"Tries: {n_tries}")
                if n_tries == max_attempts:
                    message = "The the maximum number of recombination attempts "
                    message += "was exceeded. No new variants are emerging from "
                    message += "recombining the current parents."
                    print(message)
                    print(
                        "Recombined sequences will be mutated until new variants emerge:"
                    )
                    apply_mutations = True

        # logger.debug(f"RNG calls (fin createPopulationByRecombination): {self.countrng}")
        return apply_mutations

    def mutatePopulation(self, verbose=True):
        pass

    def mutateSequence(self, sequence, return_aa_info=False) -> Sequence:
        pass

    def _get_number_of_mutations(self, reference, target_seq):
        if len(reference) != len(target_seq):
            message = "The given reference sequence is of different length than "
            message += "target sequence"
            raise ValueError(message)
        count = 0
        for r, s in zip(reference, target_seq):
            if r != s:
                count += 1
        return count

    def _recombine_sequences(self, sequence1, sequence2, skip_assert=False) -> Sequence:

        # logger.debug(f"RNG calls (ini recombineSequences): {self.countrng}")
        assert len(sequence1) == len(sequence2)
        chain = sequence1.chain
        native_sequence = sequence1.native

        # Initialise the child sequence to a list of the native sequence
        child_sequence = list(sequence1.native)

        # Set common positions into the child sequence
        for i, (s1, s2) in enumerate(zip(sequence1, sequence2)):
            if s1 == s2 and s1 != native_sequence[i]:
                child_sequence[i] = s1

        # Get recombination operations randomly in a list
        recombinations = []
        for p in self.mutable_positions[chain]:
            s1 = sequence1[p - 1]
            s2 = sequence2[p - 1]
            if s1 != s2:
                new_position = [s1, s2][self.rng.randint(0, 1)]
                # logger.debug(f"new positions: {new_position}")
                recombinations.append((p - 1, new_position))
        self.rng.shuffle(recombinations)

        # Apply recombinations
        for i, aa in recombinations:
            target = child_sequence.copy()
            target[i] = aa
            if child_sequence[i] != aa:
                assert child_sequence != target
            n_mutations = self._get_number_of_mutations(native_sequence, target)
            if n_mutations <= self.max_mutations[chain]:
                child_sequence = target

        # Convert child sequence back to string
        child_sequence = "".join(child_sequence)

        # Check the number of mutations on the new child sequence
        n_mutations = self._get_number_of_mutations(native_sequence, child_sequence)

        if not skip_assert:
            assert n_mutations <= self.max_mutations[chain]

        child_sequence = Sequence(
            child_sequence,
            sequence1.chain,
            self._get_new_sequence_index(chain),
            child=True,
            recombined=True,
            native=native_sequence,
        )

        # logger.debug(f"RNG calls (fin recombineSequences): {self.countrng}")
        return child_sequence

    def revertSequence(self, sequence) -> Sequence:
        pass

    def mutateResidue(self, pose, mutant_position, mutant_aa):
        pass

    def _getMutatedPositions(self, native_sequence, target_sequence) -> dict:
        pass

    def _create_first_population(self, sequence_init, n_missing, mutated_seq_pct):
        pass


def round_half_up(n, decimals=0):
    pass
