import numpy as np


class Sequence:
    """
    Class for containing an MOO sequence instance.

    Attributes
    ==========
    sequence : str
        The protein sequence of a specific chain in the MSD states.
    chain :
        The PDB id of the chain that the sequence represents.
    ...
    """
# TODO un dictionary de atributs, el boolen de child
    
    def __init__(
        self,
        sequence: str,
        chain: str,
        index: int,
        native=None,
        child=False,
        parent=False,
        active: bool = True,
        label: str = None,
        mutated: bool = False,
        recombined: bool = False,
        reverted: bool = False,
        recombined_mutated: bool = False,
    ):
        """
        Creates a new sequence object.

        Paramters
        =========
        sequence : str
            Amino acid sequence string
        chain : str
            PDB chain to which this sequence represents a variant.
        index : int
            The index of the sequence in the Genetic Algorithm
        child : bool
            Determines if this sequence is currently being treated as a child sequence
            in the Genetic Algorithm.
        parent : bool
            Determines if this sequence is currently being treated as a parent sequence
            in the Genetic Algorithm.
        active : bool
            Determines if this sequence is currently being treated as active in the
            Genetic Algorithm, i.e., active in the pool of sequences being evaluated.
        label : str
            The string representing the label (description) of the sequence.
        mutated : bool
            Determines if this sequence comes from a mutation of a parent sequence.
        recombined : bool
            Determines if this sequence comes from a recombination of parent sequences.
        reverted : bool
            Determines if this sequence has been reverted to fulfil the maximum number of mutations.
        """

        self._sequence = sequence
        self._chain = chain
        self._index = index
        self._parent = parent
        self._child = child
        self._active = active
        self._label = label
        self._native = native
        self._states_energy = {}
        self._recombined = recombined
        self._mutated = mutated
        self._reverted = reverted
        self._recombined_mutated = recombined_mutated
        self._mutations = []

    @property
    def sequence(self):
        return self._sequence

    @property
    def chain(self):
        return self._chain

    @property
    def index(self):
        return self._index

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    @property
    def active(self):
        return self._active

    @property
    def label(self):
        return self._label

    @property
    def native(self):
        return self._native

    @property
    def states_energy(self):
        return self._states_energy

    @property
    def recombined(self):
        return self._recombined

    @property
    def mutated(self):
        return self._mutated

    @property
    def reverted(self):
        return self._reverted

    @property
    def recombined_mutated(self):
        return self._recombined_mutated

    @property
    def mutations(self):
        return self._mutations

    @sequence.setter
    def sequence(self, value):
        self._sequence = value

    @mutations.setter
    def mutations(self, value):
        self._mutations = value

    @states_energy.setter
    def states_energy(self, value):
        """
        Set the optimised energy of the sequence in the context of a specific state.

        Parameters
        ==========
        state : int
            State index
        energy : float
            Score of the sequence when evaluated in the state.
        """
        state, energy = value
        self._states_energy[state] = energy

    @native.setter
    def native(self, native_sequence):
        """
        Define a sequence as the native sequence. This is used to calculate the
        list of mutations that the current sequence carries.

        Parameters
        ==========
        native_sequence : str
            The string representing the native sequence
        """
        # Check that sequences have the same length
        if len(native_sequence) != len(self):
            message = "The native sequence should have the same length as the current "
            message += "sequence object."
            raise ValueError(message)
        self.native = native_sequence

    @active.setter
    def active(self, value):
        self._active = value

    @recombined.setter
    def recombined(self, value):
        self._recombined = value

    @mutated.setter
    def mutated(self, value):
        self._mutated = value

    @reverted.setter
    def reverted(self, value):
        self._reverted = value

    @recombined_mutated.setter
    def recombined_mutated(self, value):
        self._recombined_mutated = value

    @parent.setter
    def parent(self, value):
        self._parent = value

    @child.setter
    def child(self, value):
        self._child = value

    @label.setter
    def label(self, value):
        self._label = value

    @chain.setter
    def chain(self, value):
        self._chain = value

    def contains_energies(self, n_states):
        """
        Return if sequence contain N energy values in its states_energy attribute
        for the N number of states specified.
        """
        # count = 0
        # for state in self.states_energy:
        #     if isinstance(self.states_energy[state], float):
        #         if not np.isnan(self.states_energy[state]):
        #             count += 1
        # if count == n_states:
        #     return True
        # else:
        #     return False
        count = 0
        for state, energy in self.states_energy.items():
            if isinstance(energy, float):
                if not np.isnan(energy):
                    count += 1
        if count == n_states:
            return True
        else:
            return False

    def get_mutations(self):
        """
        Get a list of mutations from a reference sequence

        Parameters
        =========
        reference : str
            Reference sequence upon which calculate the mutations.
        """
        self.mutations = []
        for i, (r, s) in enumerate(zip(self.native, self.sequence)):
            if r != s:
                self.mutations.append((r, i + 1, s))

    def mutate_position(self, index, aa):
        """
        Mutate a specific position in the sequence.

        Parameters
        ==========
        index : int
            Position in the sequence to mutate.
        aa : str
            Amino acid to mutate the position to.
        """
        sequence = list(self.sequence)
        sequence[index] = aa
        self.sequence = "".join(sequence)
        self.get_mutations()

    def __iter__(self):
        # returning __iter__ object
        self._iter_n = -1
        self._stop_inter = len(self.sequence)
        return self

    def __next__(self):
        self._iter_n += 1
        if self._iter_n < self._stop_inter:
            return self.sequence[self._iter_n]
        else:
            raise StopIteration

    def __getitem__(self, i):
        return self.sequence[i]

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        label = f"Chain :{self.chain}\n"
        label += f"Index : {self.index}\n"
        label += f"Active : {self.active}\n"
        label += f"Parent : {self.parent}\n"
        label += f"Child : {self.child}\n"
        label += f"Recombined : {self.recombined}\n"
        label += f"Mutated : {self.mutated}\n"
        label += f"Sequence : {self.sequence}\n"
        if self.mutations != []:
            label += "Mutations:\n"
            for m in self.mutations:
                label += "\t" + m[0] + str(m[1]) + m[2] + "\n"
        return label

    def debugPrint(self):
        label = f"Chain:{self.chain}/Index:{self.index}/mutated:{self.mutated}/recombined:{self.recombined}/reverted:{self.reverted}"
        label += f"/recombinedMutated:{self.recombined_mutated}/child:{self.child}/parent:{self.parent}/active:{self.active}"
        return label
