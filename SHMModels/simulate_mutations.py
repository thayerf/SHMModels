import numpy as np
import numpy.random
import pkgutil
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from SHMModels.summary_statistics import write_all_stats
from SHMModels.fitted_models import ContextModel
#from timeit import default_timer as timer


class MutationRound(object):
    """A round of deamination and repair, resulting in mutations. A MutationRound has the following properties:

    Attributes:
    start_seq -- The initial sequence.
    aid_lesions -- The locations of AID lesions.
    repair_types -- How each lesion was repaired.
    pol_eta_params -- A dictionary, keyed by nucleotide, each element
    describing the probability of Pol eta creating a mutation from
    that nucleotide to another nucleotide.
    ber_params -- A vector describing the probability that the BER
    machinery incorporates each nucleotide.
    exo_params -- The number of bases exo1 strips out to the left and
    to the right.
    replication_time -- Amount of time allowed for repair machinery to
    work. If machinery not recruited by this time, lesion is
    replicated over (C->T mutation).

    """

    def __init__(self, start_seq,
                 replication_time=1,
                 ber_lambda=1,
                 mmr_lambda=1,
                 bubble_size=20,
                 aid_time=1,
                 exo_params={'left': .2, 'right': .2},
                 pol_eta_params={
                     'A': [1, 0, 0, 0],
                     'G': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0],
                     'T': [0, 0, 0, 1]
                 },
                 p_fw=.5,
                 ber_params=[0, 0, 1, 0],
                 aid_context_model=None):
        """Returns a MutationRound object with a specified start_seq"""
        if not isinstance(start_seq, Seq):
            raise TypeError("The input sequence must be a Seq object")
        # we're going to need reverse complements, so the alphabet is important
        if not isinstance(start_seq.alphabet, type(IUPAC.unambiguous_dna)):
            raise TypeError("The alphabet must be IUPAC.unambiguous_dna")
        self.start_seq = start_seq
        self.replication_time = replication_time
        self.bubble_size = bubble_size
        self.aid_time = aid_time
        self.ber_lambda = ber_lambda
        self.mmr_lambda = mmr_lambda
        self.exo_params = exo_params
        self.pol_eta_params = pol_eta_params
        self.ber_params = ber_params
        self.p_fw = p_fw
        self.aid_context_model = aid_context_model
        self.NUCLEOTIDES = ["A", "G", "C", "T"]
        self.mmr_sizes = []

    def mutation_round(self):
        self.sample_lesions()
        self.sample_repair_types()
        self.sample_repaired_sequence()

    def sample_lesions(self):
        """Sample lesions induced by AID"""
        self.aid_lesions = make_aid_lesions(self.start_seq,
                                            context_model=self.aid_context_model,
                                            bubble_size=self.bubble_size,
                                            time=self.aid_time,
                                            p_fw=self.p_fw)

    def sample_repair_types(self):
        """Sample a repaired sequence given a base sequence and lesions."""
        # first get waiting times to recruit the BER/MMR machinery
        ber_wait_times = self.sample_ber_wait_times()
        mmr_wait_times = self.sample_mmr_wait_times()
        exo_positions = self.sample_exo_positions()
        self.repair_types = create_repair_types(self.aid_lesions,
                                                ber_wait_times,
                                                mmr_wait_times,
                                                exo_positions,
                                                self.replication_time)

    def sample_repaired_sequence(self):
        int_seq = self.start_seq
        # choose either the fw or rc strand to sample
        strand = np.random.choice([0, 1], size=1, p=[self.p_fw, 1-self.p_fw])[0]
        repairs = list(self.repair_types[strand])
        while len(repairs) > 0:
            # get the next lesion to repair and how to repair it,
            # update the list of resolutions to remove it and any
            # others that are resolved in the process.
            (rt, repairs) = get_next_repair(repairs)
            # add the info about exo length here
            if(rt.repair_type == "mmr"):
                self.mmr_sizes = self.mmr_sizes + [rt.exo_hi - rt.exo_lo]
            if strand == 0:
                int_seq = self.sample_sequence_given_repair(int_seq, rt)
            elif strand == 1:
                int_seq = self.sample_sequence_given_repair(int_seq.reverse_complement(), rt).reverse_complement()
            else:
                raise ValueError("Something went wrong, strand should be either 0 or 1, it was " + strand)
        self.repaired_sequence = int_seq

    def sample_sequence_given_repair(self, sequence, r):
        """Samples an intermediate sequence given input and repair type.

        Keyword arguments:
        sequence -- A Seq object containing the input sequence.
        r -- A Repair object describing the repair.

        Returns: A new sequence.
        """
        # so we can replace elements of the string
        s = list(str(sequence))
        if r.repair_type == "replicate":
            s[r.idx] = "T"
        elif r.repair_type == "ber":
            s[r.idx] = self.sample_ber()
        elif r.repair_type == "mmr":
            s = self.sample_pol_eta(s, r.exo_lo, r.exo_hi)
            
        s = "".join(s)
        return Seq(s, alphabet=IUPAC.unambiguous_dna)

    def sample_ber(self):
        """Samples a nucleotide as repaired by BER.

        Returns: A nucleotide.
        """
        return np.random.choice(self.NUCLEOTIDES, size=1, p=self.ber_params)[0]

    def sample_pol_eta(self, seq, lo, hi):
        """Samples sequences repaired by pol eta

        Keyword arguments:
        seq -- A list of characters describing the base sequence.
        lo -- The index of the most 5' nucleotide to be sampled.
        hi -- The index of the most 3' nucleotide to be sampled.

        Returns: A list of characters describing the sampled sequence.
        """
        new_seq = seq
        for i in range(lo, hi + 1):
            new_seq[i] = np.random.choice(self.NUCLEOTIDES, size=1, p=self.pol_eta_params[seq[i]])[0]
        return new_seq

    def sample_ber_wait_times(self):
        # for every lesion, sample a random exponential with rate
        # parameter ber_lambda
        return((np.random.exponential([1. / self.ber_lambda for
                                       _ in self.aid_lesions[0]]),
                np.random.exponential([1. / self.ber_lambda for
                                       _ in self.aid_lesions[1]])))

    def sample_mmr_wait_times(self):
        # for every lesion, sample a random exponential with rate parameter mmr_lambda
        return((np.random.exponential([1. / self.mmr_lambda for
                                       _ in self.aid_lesions[0]]),
                np.random.exponential([1. / self.mmr_lambda for
                                       _ in self.aid_lesions[1]])))

    def sample_exo_positions(self):
        l = len(self.start_seq)
        exo_positions = ([(max(0, a - np.random.geometric(self.exo_params['left'])),
                           min(a + np.random.geometric(self.exo_params['right']), l - 1)) for
                          a in self.aid_lesions[0]],
                         [(max(0, a - np.random.geometric(self.exo_params['left'])),
                           min(a + np.random.geometric(self.exo_params['right']), l - 1)) for
                          a in self.aid_lesions[1]])
        return(exo_positions)

class Repair(object):
    """Describes how a lesion is repaired. A Repair object has the following properties:

    Attributes:
    idx -- The location of the lesion.
    repair_type -- The type of repair machinery recruited first.
    repair_time -- The time at which the repair machinery is recruited.
    exo_lo -- The position of the most 3' base stripped out by EXO1.
    exo_hi -- The position of the most 5' base stripped out by EXO1.

    """
    def __init__(self, idx, repair_type, repair_time, exo_lo, exo_hi):
        self.idx = idx
        self.time = repair_time
        self.repair_type = repair_type
        if repair_type == "mmr":
            self.exo_lo = exo_lo
            self.exo_hi = exo_hi


def create_repair_types(aid_lesions, ber_wait_times, mmr_wait_times, exo_positions, replication_time):
    """Creates repair types from recruitment times for repair machinery.

    Keyword arguments:
    aid_lesions -- Two lists, first containing the indices of aid
    lesions on the fw strand, second containing the indices of aid
    lesions on the rc strand.
    ber_wait_times -- Two lists, giving recruitment times of ber
    machinery to each of the aid lesions.
    mmr_wait_times -- Two lists, giving recruitment times of the mmr
    machinery to each of the aid lesions.
    exo_positions -- Two lists, giving the indices of the 5'-most and
    3'-most bases that would be stripped out if each lesion were
    repaired by mmr.
    replication_time -- If no repair machinery is recruited by this
    time, the lesion gets replicated over.

    Returns: Two lists of Repair objects describing the first type of
    repair machinery recruited to each lesion and how it will act.

    """
    repairs = ([], [])
    for strand in [0, 1]:
        zipped = zip(aid_lesions[strand],
                     ber_wait_times[strand],
                     mmr_wait_times[strand],
                     exo_positions[strand])
        for (idx, bwt, mwt, el) in zipped:
            if replication_time < bwt and replication_time < mwt:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="replicate",
                                              repair_time=replication_time,
                                              exo_lo=None,
                                              exo_hi=None))
            elif bwt < mwt:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="ber",
                                              repair_time=bwt,
                                              exo_lo=None,
                                              exo_hi=None))
            else:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="mmr",
                                              repair_time=mwt,
                                              exo_lo=el[0],
                                              exo_hi=el[1]))
    return repairs


def get_next_repair(repair_list):
    """Describes repair types for a set of lesions

    Keyword arguments:
    repair_list -- A list of Repair objects.

    Returns: A tuple giving the index and repair type of the next
    lesion to repair along with the remaining lesions.

    """
    (next_repair_time, next_repair, next_repair_idx) = \
        min([(val.time, val, idx) for (idx, val) in enumerate(repair_list)])
    new_repair_list = list(repair_list)
    if next_repair.repair_type == "mmr":
        # we only keep repairs that are outside of the range of exo
        new_repair_list = [r for r in new_repair_list if
                           r.idx < next_repair.exo_lo or
                           r.idx > next_repair.exo_hi]
        return (next_repair, new_repair_list)
    else:
        new_repair_list.pop(next_repair_idx)
        return(next_repair, new_repair_list)


def make_aid_lesions(sequence, context_model, bubble_size=20, time=1, p_fw=.5):
    """Simulates AID lesions on a sequence

    Keyword arguments:
    sequence -- A Seq object using the IUPAC Alphabet

    Returns: A pair of vectors, the first giving the indices of lesions
    on the forward strand and the second giving the indices of lesions
    on the reverse complement.

    """
    if not isinstance(sequence, Seq):
        raise TypeError("The input sequence must be a Seq object")
    if not isinstance(sequence.alphabet, type(IUPAC.unambiguous_dna)):
        raise TypeError("The input sequence must have an IUPAC.unambiguous_dna alphabet")
    # choose a random position in the sequence and a strand
    stop_site = np.random.randint(0, len(sequence))
    strand = np.random.choice(["fw", "rc"], size=1, p=[p_fw, 1 - p_fw])
    lesions = deaminate_in_bubble(sequence, bubble_size, stop_site,
                                  strand, context_model, time)
    return lesions


def c_bases_in_bubble(sequence, bubble_size, stop_site, strand):
    """Identifies all the Cs in a transcription bubble

    Keyword arguments:
    sequence -- A Seq object.
    bubble_size -- The size of transcription bubble.
    stop_site -- The 3'-most nucleotide in the transcription bubble.
    strand -- The strand to be deaminated, can be "fw" or "rc".

    Returns: A pair of vectors, the first giving the indices of C's on
    the forward strand and the second giving the indices of C's on the
    reverse complement.

    """
    l = len(sequence)
    if stop_site > (l - 1) or stop_site < 0:
        raise ValueError("'stop_site' must be between 0 and l-1")
    if strand == "fw":
        idx_lo = np.max([0, stop_site - (bubble_size - 1)])
        idx_hi = stop_site
        cs_fw = [i for i in range(idx_lo, idx_hi + 1)
                 if sequence[i] == "C"]
        cs_rc = []
    elif strand == "rc":
        idx_lo = np.max([0, l - 1 - stop_site])
        idx_hi = np.min([l - 1 - stop_site + (bubble_size - 1), l - 1])
        cs_fw = []
        cs_rc = [i for i in range(idx_lo, idx_hi + 1)
                 if sequence.reverse_complement()[i] == "C"]
    else:
        raise ValueError("strand must be either 'fw' or 'rc'")
    return (cs_fw, cs_rc)


def deaminate_in_bubble(sequence, bubble_size, stop_site, strand,
                        context_model, time=1):
    """Simulates deamination of Cs in a transcription bubble

    Keyword arguments:
    sequence -- A Seq object.
    bubble_size -- The size of the region in which C's are available
    for deamination.
    stop_site -- The 3'-most nucleotide available for deamination,
    described in terms of the forward strand.
    strand -- The strand to be deaminated, can be "fw" or "rc".
    time -- The amount of time AID is bound.

    Returns: A pair of vectors, the first giving the indices of
    lesions on the forward strand and the second giving the indices of
    lesions on the reverse complement.

    """
    (c_idx_fw, c_idx_rc) = c_bases_in_bubble(sequence,
                                             bubble_size,
                                             stop_site,
                                             strand)
    c_rates_fw = get_aid_rates(c_idx_fw, sequence, strand="fw", context_model=context_model)
    c_rates_rc = get_aid_rates(c_idx_rc, sequence, strand="rc", context_model=context_model)
    waiting_times_fw = [np.random.exponential(scale=1/r) for r in c_rates_fw]
    waiting_times_rc = [np.random.exponential(scale=1/r) for r in c_rates_rc]
    deam_fw = [i for (i, wt) in zip(c_idx_fw, waiting_times_fw) if wt <= time]
    deam_rc = [i for (i, wt) in zip(c_idx_rc, waiting_times_rc) if wt <= time]
    return (deam_fw, deam_rc)


def get_aid_rates(idx, sequence, strand, context_model):
    """Computes deamination rates for positions in a sequence

    Keyword arguments:
    idx -- A vector containing indices of C's to be deaminated
    sequence -- The sequence the C's are found in.
    strand -- Whether idx refers to a position on the fw strand or the
    rc strand.
    context_model -- A ContextModel object describing deamination
    probabilities.

    Returns:
    A vector of the same size as idx containing deamination rates.

    """
    if strand == "fw":
        probs = [context_model.get_context_prob(i, sequence) for i in idx]
        if any([p is None for p in probs]):
            print idx, probs
        rates = [-np.log(1 - p) for p in probs]
    elif strand == "rc":
        probs = [context_model.get_context_prob(len(sequence) - i, sequence.reverse_complement()) for i in idx]
        if any([p is None for p in probs]):
            print idx, probs
        rates = [-np.log(1 - p) for p in probs]
    else:
        raise ValueError("strand must be either 'fw' or 'rc'")    
    return rates


def simulate_sequences_abc(germline_sequence,
                           aid_context_model,
                           context_model_length,
                           context_model_pos_mutating,
                           n_seqs,
                           n_mutation_rounds,
                           ss_file,
                           param_file,
                           n_sims):

    sequence = list(SeqIO.parse(germline_sequence, "fasta",
                                alphabet=IUPAC.unambiguous_dna))[0]
    aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
    aid_model = ContextModel(context_model_length,
                             context_model_pos_mutating,
                             aid_model_string)
    n_sum_stats = 310
    n_params = 9
    ss_array = np.zeros([n_sims, n_sum_stats])
    param_array = np.zeros([n_sims, n_params])
    for sim in range(n_sims):
        mutated_seq_list = []
        mmr_length_list = []
        # the prior specification
        #start_prior = timer()
        ber_lambda = np.random.uniform(0, 1, 1)[0]
        bubble_size = np.random.randint(5, 50)
        exo_left = 1 / np.random.uniform(1, 50, 1)[0]
        exo_right = 1 / np.random.uniform(1, 50, 1)[0]
        pol_eta_params = {
                     'A': [.9, .02, .02, .06],
                     'G': [.01, .97, .01, .01],
                     'C': [.01, .01, .97, .01],
                     'T': [.06, .02, .02, .9]
        }
        ber_params = np.random.dirichlet([1, 1, 1, 1])
        p_fw = np.random.uniform(0, 1, 1)[0]
        #end_prior = timer()
        #start_seqs = timer()
        for i in range(n_seqs):
            mr = MutationRound(sequence.seq,
                               ber_lambda=ber_lambda,
                               mmr_lambda=1 - ber_lambda,
                               replication_time=100,
                               bubble_size=bubble_size,
                               aid_time=10,
                               exo_params={'left': exo_left,
                                           'right': exo_right},
                               pol_eta_params=pol_eta_params,
                               ber_params=ber_params,
                               p_fw=p_fw,
                               aid_context_model=aid_model)
            for j in range(n_mutation_rounds):
                mr.mutation_round()
                mr.start_seq = mr.repaired_sequence
            mutated_seq_list.append(SeqRecord(mr.repaired_sequence, id=""))
            if(len(mr.mmr_sizes) > 0):
                mmr_length_list.append(np.mean(mr.mmr_sizes))
        #end_seqs = timer()
        #start_ss = timer()
        ss_array[sim, :] = write_all_stats(sequence,
                                           mutated_seq_list,
                                           np.mean(mmr_length_list),
                                           file=None)
        params = [ber_lambda, bubble_size, exo_left, exo_right, ber_params[0], ber_params[1], ber_params[2], ber_params[3], p_fw]
        param_array[sim, :] = params
        #end_ss = timer()
        #print("Draw the prior: {}, Simulate sequences: {}, Write summary statistics: {}".format(end_prior - start_prior, end_seqs - start_seqs, end_ss - start_ss))
    np.savetxt(param_file, param_array, delimiter=",")
    np.savetxt(ss_file, ss_array, delimiter=",")
    return (param_array, ss_array)
