from mood.base import Sequence
from mood.optimizers.genetic_algorithm import GeneticAlgorithm


def main():
    seq1 = Sequence(
        sequence="TLDVSRQDPRYNTLKHGFNLWWPSTDAQAAGRIALCEKADDVAPALSHIIDTGMRPTVRSGGHCYEDFVSANPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=False,
        parent=False,
        active=False,
        label="Test1",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq2 = Sequence(
        sequence="TLDVSRQDPRYNTLKHGFLLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTIRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVTIPAGTQNWNGYLELYKRHNLTL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=True,
        parent=False,
        active=True,
        label="Test2",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq3 = Sequence(
        sequence="TLDVSRQDRYNTQKHGFNLRRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSSNDGARIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=False,
        parent=False,
        active=True,
        label="Test3",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq4 = Sequence(
        sequence="TLDVSRQDPRYNTPKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSQNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRNLTLL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=False,
        parent=False,
        active=True,
        label="Test4",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq5 = Sequence(
        sequence="TLDVSRQDPRYNTLKHGFNPRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLCAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=True,
        parent=False,
        active=True,
        label="Test5",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    seq6 = Sequence(
        sequence="TLDVSRQDPRYNTLKHGFNLRWPSTDAAAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        chain="A",
        index=1,
        native="TLDVSRQDPRYNTLKHGFNLRWPSTDAQAAGRIALCEKADDVAPALQHIIDTGMRPTVRSGGHCYEDFVSNNPDGAIVDLSLLNAPEVRADGTVRIPAGTQNWNGYLELYKRHNLTL",
        child=False,
        parent=False,
        active=True,
        label="Test6",
        mutated=False,
        recombined=False,
        reverted=False,
        recombined_mutated=False,
    )
    sequences = {"A": {1: seq1, 2: seq2, 3: seq3, 4: seq4, 5: seq5, 6: seq6}}
    ga = GeneticAlgorithm(
        sequences=sequences,
        active_chains=["A"],
        population_size=6,
        mutation_rate=0.005,
        mutable_positions=[1, 2, 3, 4],
        elites_fraction=0.00,
        max_mutations=6,
        iteration=0,
        seed=1234,
        debug=True,
    )

    print("GA created")

    seqq = ga._get_sequences_pool(
        "A", active=True, recombined=False, mutated=False, parent=True, child=False
    )
    print("Sequences pool")

    ga._deactivateParents()

    seqq = ga._get_sequences_pool(
        "A", active=True, recombined=False, mutated=False, parent=True, child=False
    )
    print("Sequences pool")

    ga.create_population_by_recombination()

    print("Recombination")


if __name__ == "__main__":
    main()
