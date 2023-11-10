import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import numpy as np
import argparse
import pandas as pd
import random

# export KMP_DUPLICATE_LIB_OK=TRUE

def parse_args():
    parser = argparse.ArgumentParser(
                        description='Variational Combinatorial Sequential Monte Carlo')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use.',
                        default='cellphy_toy_data')
    parser.add_argument('--n_particles',
                        type=int,
                        help='number of SMC samples.',
                        default=10)
    parser.add_argument('--batch_size',
                        type=int,
                        help='number of sites on genome per batch.',
                        default=256)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='Learning rate.',
                        default=0.001)
    parser.add_argument('--num_epoch',
                        type=int,
                        help='number of epoches to train.',
                        default=100)
    parser.add_argument('--optimizer',
                       type=str,
                       help='Optimizer for Training',
                       default='GradientDescentOptimizer')
    parser.add_argument('--branch_prior',
                       type=float,
                       help='Hyperparameter for branch length initialization.',
                       default=np.log(10))
    parser.add_argument('--M',
                       type=int,
                       help='number of subparticles to compute look-ahead particles',
                       default=10)
    parser.add_argument('--nested', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--jcmodel', 
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--cellphy_model',  # either 'gt10' or 'gt16'
                       default=None, 
                       type=str)
    parser.add_argument('--cellphy_error',
                       default=False, 
                       type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--memory_optimization',
                       help='Use memory optimization?',
                       default='on')


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    primate_data = False
    corona_data = False
    hohna_data = False
    load_strings = False
    simulate_data = False
    hohna_data_1 = False
    hohna_data_2 = False
    hohna_data_3 = False
    hohna_data_4 = False
    hohna_data_5 = False
    hohna_data_6 = False
    hohna_data_7 = False
    hohna_data_8 = False
    primate_data_wang = False
    cellphy_toy_data = False
    ginkgo = False

    args = parse_args()

    exec(args.dataset + ' = True')

    Alphabet_dir = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1]}
    alphabet_dir = {'a': [1, 0, 0, 0],
                    'c': [0, 1, 0, 0],
                    'g': [0, 0, 1, 0],
                    't': [0, 0, 0, 1]}
    Alphabet_dir_blank = {'A': [1, 0, 0, 0],
                          'C': [0, 1, 0, 0],
                          'G': [0, 0, 1, 0],
                          'T': [0, 0, 0, 1],
                          '-': [1, 1, 1, 1],
                          '?': [1, 1, 1, 1]}
    Alphabet_dir_phased = {
                                #     AA CC GG TT AC AG AT CG CT GT CA GA TA GC TC TG
                                'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A/A
                                'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C/C
                                'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # G/G
                                'T': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # T/T
                                'M': [0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0], # A/C
                                'R': [0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0], # A/G
                                'W': [0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0, 0], # A/T
                                'S': [0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0, 0], # C/G
                                'Y': [0, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5, 0], # C/T
                                'K': [0, 0, 0, 0, 0, 0, 0, 0, 0,.5, 0, 0, 0, 0, 0,.5], # G/T
                                'N': [1/16] * 16, # blank
                            }
    Alphabet_dir_unphased = {
                                #     AA CC GG TT AC AG AT CG CT GT
                                'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A/A
                                'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # C/C
                                'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # G/G
                                'T': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # T/T
                                'M': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # A/C
                                'R': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # A/G
                                'W': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # A/T
                                'S': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # C/G
                                'Y': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # C/T
                                'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # G/T
                                'N': [1/10] * 10, # blank
                            }
    alphabet = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])


    def simulateDNA(nsamples, seqlength, alphabet):
        genomes_NxSxA = np.zeros([nsamples, seqlength, alphabet.shape[0]])
        for n in range(nsamples):
            genomes_NxSxA[n] = np.array([random.choice(alphabet) for i in range(seqlength)])
        return genomes_NxSxA


    def form_dataset_from_strings(genome_strings, alphabet_dir, alphabet_num=4):
        genomes_NxSxA = np.zeros([len(genome_strings), len(genome_strings[0]), alphabet_num])
        for i in range(genomes_NxSxA.shape[0]):
            for j in range(genomes_NxSxA.shape[1]):
                genomes_NxSxA[i, j] = alphabet_dir[genome_strings[i][j]]
        taxa = ['S' + str(i) for i in range(genomes_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': genomes_NxSxA}
        return datadict

    if hohna_data or hohna_data_1:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS1.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        # print(datadict['genome'].shape)
        
    if hohna_data_2:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS2.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_3:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS3.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_4:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS4.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_5:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS5.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_6:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS6.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_7:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS7.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if hohna_data_8:
        datadict_raw = pd.read_pickle('data/hohna_datasets/DS8.pickle')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)


    if corona_data:
        datadict = pd.read_pickle('data/coronavirus.p')


    if primate_data:
        datadict_raw = pd.read_pickle('data/primate.p')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_blank)
        
    if primate_data_wang:
        datadict_raw = pd.read_pickle('data/primates_small.p')
        genome_strings = list(datadict_raw.values())
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)


    if cellphy_toy_data:
        with open('data/cellphy_toy_set.phy') as f:
            phy_file_raw = f.readlines()
        genome_strings = [line[line.find(" "):].strip() for line in phy_file_raw[1:]]
        if args.cellphy_model == 'gt16':
            datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_phased, 16)
        elif args.cellphy_model == 'gt10':
            datadict = form_dataset_from_strings(genome_strings, Alphabet_dir_unphased, 10)
        else:
            raise ValueError('To use cellphy_toy_set.phy, you must specify --cellphy_model (either gt10 or gt16)')


    if simulate_data:
        data_NxSxA = simulateDNA(3, 5, alphabet)
        # print("Simulated genomes:\n", data_NxSxA)
        taxa = ['S' + str(i) for i in range(data_NxSxA.shape[0])]
        datadict = {'taxa': taxa,
                    'genome': data_NxSxA}


    if load_strings:
        genome_strings = ['ACTTTGAGAG', 'ACTTTGACAG', 'ACTTTGACTG', 'ACTTTGACTC']
        datadict = form_dataset_from_strings(genome_strings, Alphabet_dir)

    if ginkgo:
        data = pd.read_pickle('data/gingko/test_data_14.p')
        #print(np.swapaxes(data, 1, 2).shape)
        datadict = {
            'samples' : np.arange(data.shape[0]).astype(str).tolist(),
            'data' : np.swapaxes(data, 1, 2)
        }
        #print(datadict['samples'])
        #import pdb
        #pdb.set_trace()

    if args.nested == True:
        import vncsmc as vcsmc
    else:
        # import vcsmc_jet as vcsmc
        import vcsmc as vcsmc


    #pdb.set_trace()
    vcsmc = vcsmc.VCSMC(datadict, K=args.n_particles, args=args)

    vcsmc.train(epochs=args.num_epoch, batch_size=args.batch_size, learning_rate=args.learning_rate, memory_optimization=args.memory_optimization)
