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
        mutations=[],
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
        metrics : dict
        """

        self._sequence = sequence
        self._chain = chain
        self._index = index
        self._parent = parent
        self._child = child
        self._active = active
        self._native = native
        self._states_energy = {}
        self._mutations = mutations

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
    def native(self):
        return self._native

    @property
    def mutations(self):
        return self._mutations

    @sequence.setter
    def sequence(self, value):
        self._sequence = value

    @mutations.setter
    def mutations(self, value):
        self._mutations = value

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

    @parent.setter
    def parent(self, value):
        self._parent = value

    @child.setter
    def child(self, value):
        self._child = value

    @chain.setter
    def chain(self, value):
        self._chain = value

    def calculate_mutations(self):
        """
        Get a list of mutations from a reference sequence

        Parameters
        =========
        reference : str
            Reference sequence upon which calculate the mutations.
        """
        # TODO redoo method as is possible not working, as nned to call the sequence property of the native
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
        label += f"Sequence : {self.sequence}\n"
        if self.mutations != []:
            label += "Mutations:\n"
            for m in self.mutations:
                label += "\t" + m[0] + str(m[1]) + m[2] + "\n"
        return label
