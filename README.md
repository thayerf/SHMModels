# SHMModels

This repo contains functions for simulating from mechanistically-informed models of somatic hypermutation.
The package can be installed using
```
pip install .
```

The primary function is `simulate_sequences_abc`, so for example,
```
from SHMModels.simulate_mutations import simulate_sequences_abc
simulate_sequences_abc("/Users/julia/GitHub/shmr/inst/extdata/gpt.fasta",
    "data/aid_logistic_3mer.csv",
    context_model_length = 3,
    context_model_pos_mutating = 2,
    n_seqs = 1,
    n_mutation_rounds = 3,
    ss_file = "for_nnet_ss.csv",
    param_file = "for_nnet_params.csv",
    sequence_file = "for_nnet_sequences.csv",
    n_sims = 100000,
    write_ss=False,
    write_sequences=True)
```
will simulate sequences from a 3-mer model, second position mutating, one sequence per parameter setting, with three rounds of mutation.
The sequences will be written to the file `for_nnet_sequences.csv` and the parameters used to generate those sequences will written to `for_nnet_params.csv`.
Summary statistics are a bit deprecated at this point.
