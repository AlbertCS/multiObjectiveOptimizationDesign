from mood.base import Sequence


def main():
    seq = Sequence(
        sequence="ATCG",
        chain="A",
        index=1,
        native="ATCG",
        child=False,
        parent=False,
        active=True,
        label="Test",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq.states_energy = (1, 2.0)
    print(seq.contains_energies(1))


if __name__ == "__main__":
    main()
