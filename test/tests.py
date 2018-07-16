import unittest
import numpy as np
import pkgutil
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from SHMModels.simulate_mutations import make_aid_lesions
from SHMModels.simulate_mutations import c_bases_in_bubble
from SHMModels.simulate_mutations import deaminate_in_bubble
from SHMModels.simulate_mutations import Repair
from SHMModels.simulate_mutations import get_next_repair
from SHMModels.simulate_mutations import MutationRound
from SHMModels.mutation_processing import mutation_subset
from SHMModels.mutation_processing import mutation_subset_from_samm
from SHMModels.fitted_models import ContextModel


class testMutationSubset(unittest.TestCase):

    def setUp(self):
        pass

    def test_mutation_subset(self):
        naive_seq = Seq("AGCA", alphabet=IUPAC.unambiguous_dna)
        mutated_seq = Seq("ATTT", alphabet=IUPAC.unambiguous_dna)
        (_, ms1) = mutation_subset(naive_seq, mutated_seq, base="G")
        (_, ms3) = mutation_subset(naive_seq, mutated_seq, base="C")
        self.assertEqual(str(ms1), "ATCA")
        self.assertEqual(str(ms3), "AGTA")

    def test_mutation_subset_samm(self):
        ms_both = mutation_subset_from_samm('test_seqs.csv', 'test_genes.csv',
                                            base='c', strand='both')
        ms_fw = mutation_subset_from_samm('test_seqs.csv', 'test_genes.csv',
                                          base='c', strand='fw')
        ms_rc = mutation_subset_from_samm('test_seqs.csv', 'test_genes.csv',
                                          base='c', strand='rc')
        self.assertEqual(ms_both.shape[0], 2)
        self.assertEqual(ms_fw.shape[0], 1)
        self.assertEqual(ms_rc.shape[0], 1)


class testAIDLesion(unittest.TestCase):

    def setUp(self):
        pass

    def test_aid_lesion(self):
        cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
        # Input must be a seq object
        self.assertRaises(TypeError, make_aid_lesions,
                          "AGCT", cm)
        # Input must have an IUPAC.unambiguous_dna alphabet
        seq_wrong = Seq("AGCT", alphabet=IUPAC.protein)
        self.assertRaises(TypeError, make_aid_lesions, seq_wrong, cm)
        seq = Seq("AGCT", alphabet=IUPAC.unambiguous_dna)
        (lesions_fw, lesions_rc) = make_aid_lesions(seq, cm)
        # all the lesions should be at C positions
        self.assertTrue(all([seq[i] == "C" for i in lesions_fw]))
        self.assertTrue(all([seq.reverse_complement()[i] == "C"
                             for i in lesions_rc]))

    def test_c_in_bubble(self):
        seq = Seq("GCCCAG", alphabet=IUPAC.unambiguous_dna)
        # should get a value error if the stop site is outside the
        # range of the sequence
        self.assertRaises(ValueError, c_bases_in_bubble, seq, 2, 20, "fw")
        self.assertRaises(ValueError, c_bases_in_bubble, seq, 2, -5, "fw")
        # should get a value error if the strand is not either "fw" or "rc"
        self.assertRaises(ValueError, c_bases_in_bubble, seq, 2, 1, "wrong")
        (lesions_fw, lesions_rc) = c_bases_in_bubble(seq,
                                                     bubble_size=3,
                                                     stop_site=5,
                                                     strand="fw")
        # lesions should only be at C positions
        self.assertTrue(all([seq[i] == "C" for i in lesions_fw]))
        self.assertEqual(lesions_fw, [3])
        # since we're deaminating on the forward strand, we should
        # have no lesions on the rc strand
        self.assertEqual(len(lesions_rc), 0)
        (lesions_fw, lesions_rc) = c_bases_in_bubble(seq,
                                                     bubble_size=3,
                                                     stop_site=2,
                                                     strand="rc")
        self.assertTrue(all([seq.reverse_complement()[i] == "C"
                             for i in lesions_rc]))
        self.assertEqual(lesions_rc, [5])
        # deaminating on the rc strand means we should have no lesions
        # on the fw strand
        self.assertEqual(len(lesions_fw), 0)

    def test_bubble_deamination(self):
        context_model_string = pkgutil.get_data("SHMModels", "data/aid_goodman.csv")
        cm = ContextModel(context_length=3, pos_mutating=2, csv_string=context_model_string)
        s = Seq("C" * 10 + "A" * 10 + "C" * 10,
                alphabet=IUPAC.unambiguous_dna)
        (lesions_fw, lesions_rc) = deaminate_in_bubble(s, 20, len(s) - 1, "fw", cm, time=100)
        # no lesions on the rc strand
        self.assertEqual(len(lesions_rc), 0)
        # all the deaminations on the fw strand should be between 20 and 29
        if(len(lesions_fw) > 0):
            self.assertTrue(np.max(lesions_fw) <= len(s))
            self.assertTrue(np.min(lesions_fw) >= 20)

    def test_lesions(self):
        cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
        s = Seq("C" * 10 + "A" * 10 + "AACAGCAGCGACGTC",
                alphabet=IUPAC.unambiguous_dna)
        (lesions_fw, lesions_rc) = make_aid_lesions(s, cm, 10, 5)

    def test_sample_wait_times(self):
        return 0

    def test_pol_eta(self):
        # set a sequence, lesion, repair type, ber parameters, and
        # call sample_repaired_sequence
        mr = MutationRound(Seq("AAGCT", alphabet=IUPAC.unambiguous_dna))

    def test_ber(self):
        # set a sequence, lesion, repair type, ber parameters, and
        # call sample_repaired_sequence
        pass

    def test_create_repair_types(self):
        pass

    def test_next_repair(self):
        # test if one lesion is repaired by mmr and the other is
        # repaired in the process
        r1 = Repair(0, "mmr", .5, 0, 1)
        r2 = Repair(1, "ber", 1, None, None)
        (nr, new_repair_list) = get_next_repair([r1, r2])
        self.assertEqual(nr.repair_type, "mmr")
        self.assertEqual(nr.time, .5)
        self.assertEqual(len(new_repair_list), 0)
        # now suppose the first lesion is repaired by ber
        r1 = Repair(0, "mmr", 1, 0, 1)
        r2 = Repair(1, "ber", .5, None, None)
        (nr, new_repair_list) = get_next_repair([r1, r2])
        self.assertEqual(nr.repair_type, "ber")
        self.assertEqual(nr.time, .5)
        self.assertEqual(len(new_repair_list), 1)

    def test_get_context(self):
        cm = ContextModel(context_length=5, pos_mutating=2, csv_string=None)
        cm2 = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
        s = "AAGCT"
        self.assertEqual(cm.get_context(idx=2, sequence=s), "AAGCT")
        self.assertEqual(cm2.get_context(idx=2, sequence=s), "AAG")
        self.assertEqual(cm.get_context(idx=0, sequence=s), "NNAAG")
        self.assertEqual(cm.get_context(idx=4, sequence=s), "GCTNN")

    def test_in_flank(self):
        cm = ContextModel(context_length=5, pos_mutating=2, csv_string=None)
        self.assertTrue(cm.in_flank(idx=0, seq_len=10))
        self.assertTrue(cm.in_flank(idx=1, seq_len=10))
        self.assertFalse(cm.in_flank(idx=2, seq_len=10))
        self.assertTrue(cm.in_flank(idx=9, seq_len=10))
        self.assertTrue(cm.in_flank(idx=8, seq_len=10))
        self.assertFalse(cm.in_flank(idx=7, seq_len=10))
        cm2 = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
        self.assertTrue(cm2.in_flank(idx=0, seq_len=10))
        self.assertTrue(cm2.in_flank(idx=1, seq_len=10))
        self.assertFalse(cm2.in_flank(idx=2, seq_len=10))
        self.assertFalse(cm2.in_flank(idx=9, seq_len=10))
        self.assertFalse(cm2.in_flank(idx=8, seq_len=10))
        self.assertFalse(cm2.in_flank(idx=7, seq_len=10))

    def test_compute_marginal_prob(self):
        cm = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
        cm.context_dict = {}
        cm.context_dict["AAC"] = .5
        cm.context_dict["AGC"] = .1
        self.assertEqual(cm.compute_marginal_prob("NAC"), .5)
        self.assertEqual(cm.compute_marginal_prob("NNC"), .3)


if __name__ == '__main__':
    unittest.main()
