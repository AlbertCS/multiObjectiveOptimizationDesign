from multiobjective_design.base import Sequence


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
        recombinedMutated=False,
    )

    seq.mutatePosition(0, "G")


if __name__ == "__main__":
    main()
