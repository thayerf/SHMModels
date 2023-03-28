# Load our stuff
import numpy as np
from Bio import SeqIO
from simulate_mutations import *
from SHMModels.fitted_models import ContextModel
import pkgutil
import logging
import os
import sys
import json
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import collections
from scipy.stats import norm
# Load options
import pandas as pd
import glob
from Bio.Alphabet.IUPAC import unambiguous_dna, ambiguous_dna
from random import sample
# Load df with all seqs
df = pd.read_pickle("/home/tfisher2/shm/SHMModels/SHMModels/full_edge_df.pk1")
parent_sequences = df['PARENT_SEQ']

cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
def sample_prior():
    # Sample from the prior for all parameters
    # Lengthscale
    ls = np.random.uniform(low = -12.0, high = -2.0)
    # Variance of the Gaussian Process
    sg = np.random.uniform(low = 5.0, high = 15.0)
    # Gaussian Process mean (unchanged)
    off = -10
    # Probability of the forward strand
    p_fw = np.random.uniform(low =0.0, high = 1.0)
    # Geometric prob for size of exo stripping region to left and right
    exo_left = np.random.uniform(low =1.0, high = 20.0)
    exo_right = np.random.uniform(low =1.0, high = 20.0)
    # Probability that BER is recruited
    ber_prob = np.random.uniform(low = 0.0, high = 1.0)
    # Probability of thinning a prelesion
    thinning_prob = norm.cdf(10.0/sg)
    # Base rate on each strand
    fw_br = 0.5
    rc_br = 0.5
    return {           "lengthscale" : ls,
                       "gp_sigma" : sg,
                       "gp_ridge" : .04,
            "gp_offset": off,
            "p_fw": p_fw,
            "fw_br": fw_br,
            "rc_br": rc_br,
            "exo_left": exo_left,
            "exo_right": exo_right,
            "ber_prob": ber_prob
            }

# Get batch (BER AND POL ETA DEFINED HERE)
def gen_batch_letters(seq,batch_size, params):
       # The prior specification
    ber_prob = params['ber_prob']
    ber_params = [0.25,0.25,0.25,0.25]
    
    bubble_size = 25.0
    pol_eta_params = {
        "A": [0.9, 0.02, 0.02, 0.06],
        "G": [0.01, 0.97, 0.01, 0.01],
        "C": [0.01, 0.01, 0.97, 0.01],
        "T": [0.06, 0.02, 0.02, 0.9],
    }
    prior_params = params
    exo_left = 1.0/prior_params['exo_left']
    exo_right = 1.0/prior_params['exo_right']
    mutated_seq_list = []
    for i in range(batch_size):
          mr = MutationRound(
          seq,
          ber_lambda=1.0,
          #mmr_lambda=(1 - ber_prob)/ber_prob,
          mmr_lambda= 0.001,
          replication_time=100,
          bubble_size=bubble_size,
          aid_time=10,
          exo_params={"left": exo_left, "right": exo_right},
          pol_eta_params=pol_eta_params,
          ber_params=ber_params,
          p_fw= 0.001,
          # p_fw= prior_params['p_fw'],
          aid_context_model=cm,
          log_ls = prior_params['lengthscale'],
          sg = prior_params['gp_sigma'],
          fw_br = prior_params['fw_br'],
          rc_br = prior_params['rc_br'],
          off = prior_params['gp_offset']
          )
          mr.mutation_round()
          mutated_seq_list.append(SeqRecord(mr.repaired_sequence, id=""))
    return [list(i.seq) for i in mutated_seq_list]
num_seqs = 1000
true_model_params = sample_prior()
true_params_array = np.array((true_model_params['lengthscale'],
                                true_model_params['gp_sigma'],
                                true_model_params['p_fw'],
                                true_model_params['exo_left'],
                                true_model_params['exo_right'],
                                true_model_params['ber_prob']))
obs_sample = []
parent_sample = sample(list(parent_sequences),num_seqs)


for i in range(num_seqs):
    t_seq = gen_batch_letters(Seq(parent_sample[i], unambiguous_dna),1, true_model_params)
    obs_sample.append(t_seq[0])

mut_orig = []
for i in range(num_seqs):
    for j in range(len(obs_sample[i])):
        if obs_sample[i][j] != parent_sample[i][j]:
            mut_orig.append(parent_sample[i][j])
            
plt.hist(mut_orig)