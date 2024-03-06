import unittest

from multiobjective_design.base.sequence import Sequence


class TestSequence(unittest.TestCase):
    def setUp(self):
        self.seq = Sequence(
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

    def test_init(self):
        self.assertEqual(self.seq.sequence, "ATCG")
        self.assertEqual(self.seq.chain, "A")
        self.assertEqual(self.seq.index, 1)
        self.assertEqual(self.seq.native, "ATCG")
        self.assertEqual(self.seq.child, False)
        self.assertEqual(self.seq.parent, False)
        self.assertEqual(self.seq.active, True)
        self.assertEqual(self.seq.label, "Test")
        self.assertEqual(self.seq.mutated, False)
        self.assertEqual(self.seq.recombined, False)
        self.assertEqual(self.seq.reverted, False)
        self.assertEqual(self.seq.recombinedMutated, False)

    def test_contains_energies(self):
        self.seq.state_energy = (1, 2.0)
        self.assertTrue(self.seq.contains_energies(1))

    def test_getMutations(self):
        self.seq.mutatePosition(0, "G")
        self.assertEqual(self.seq.mutations, [("A", 1, "G")])

    def test_mutatePosition(self):
        self.seq.mutatePosition(0, "G")
        self.assertEqual(self.seq.sequence, "GTCG")

    def test_len(self):
        self.assertEqual(len(self.seq), 4)

    def test_repr(self):
        self.assertEqual(
            self.seq.__repr__(),
            "Chain :AnIndex : 1\nActive : True\nParent : False\nChild : False\nRecombined : False\nMutated : False\nSequence : ATCG\n",
        )

    def test_debugPrint(self):
        self.assertEqual(
            self.seq.debugPrint(),
            "Chain:A/Index:1/mutated:False/recombined:False/reverted:False/recombinedMutated:False/child:False/parent:False/active:True",
        )


if __name__ == "__main__":
    unittest.main()
