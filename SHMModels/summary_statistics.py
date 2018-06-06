import numpy as np


def mutation_probs(gl_seq, mutated_seq_list):
    """Computes the fraction of sequences with a mutation at each position.

    Keyword arguments:
    gl_seq -- The germline sequence.
    mutated_seq_list -- A list of SeqRecord objects containing the mutated sequences.

    Returns: A list giving the fraction of mutations at each position.
    """
    mutation_counts = [0. for _ in range(len(gl_seq))]
    for seq in mutated_seq_list:
        for (i, b) in enumerate(seq):
            if b != gl_seq[i]:
                mutation_counts[i] += 1
    mutation_probs = [m / len(mutated_seq_list) for m in mutation_counts]
    return mutation_probs


def dist_between_mutations(gl_seq, mutated_seq_list):
    """Computes the average distance between mutations.

    Keyword arguments:
    gl_seq -- The germline sequence.
    mutated_seq_list -- A list of SeqRecord objects containing the mutated sequences.

    Returns: The average distance between mutations for each sequence.
    """
    mutation_dists = [0. for _ in range(len(mutated_seq_list))]
    for (seq_idx, seq) in enumerate(mutated_seq_list):
        mut_indices = []
        for (i, b) in enumerate(seq):
            if b != gl_seq[i]:
                mut_indices.append(i)
        total_dist = 0.
        for i in range(len(mut_indices) - 1):
            for j in range(i+1, len(mut_indices)):
                total_dist = total_dist + np.abs(mut_indices[i] - mut_indices[j])
        if len(mut_indices) > 1:
            mutation_dists[seq_idx] = total_dist / (len(mut_indices) * (len(mut_indices) - 1) / 2)
    return np.mean(mutation_dists)


def write_all_stats(gl_seq, mutated_seq_list, file=None):
    """Computes all the summary statistics

    Keyword arguments:
    gl_seq -- The germline sequence.
    mutated_seq_list -- A list of SeqRecord objects containing the mutated sequences.
    file -- If None, just returns a list of the summary statistics,
    otherwise returns the list of statistics and writes them to a
    file.

    Returns: A list of summary statistics.
    """
    ss = mutation_probs(gl_seq, mutated_seq_list)
    ss.append(dist_between_mutations(gl_seq, mutated_seq_list))
    if file is not None:
        with open(file, 'a') as output_file:
            output_file.write('\t'.join([str(m) for m in ss]))
            output_file.write('\n')
    return ss
