# CPCNN
CPCNN: a tool for RNA coding potential prediction based on sequences information

Version: 1.0

Authors: Bingqian Cao and Kai Song

Maintainer: Kai Song songkai1987@126.com 

# Description

CPCNN predicts lncRNA using deep learning method. The method has good prediction accuracy for transcripts with short ORFs. CPCNN uses Convolution Neutral Network (CNN) technique to automatically learn nucleotide and codon patterns from transcript sequences and simultaneously build a predictive model based on the learned features. Given a query sequence, the framework outputs two scores which indicating the probability that the sequence should be identified as coding transcript or lncRNA.

# Dependencies

CPCNN requires Python 3.6 with the packages of numpy, pandas, theano, keras, scikit-learn, sklearn and Biopython. We recommand the use Conda to install all dependencies. 

# Usage 

The input of CPCNN is three fasta files containing two training files and one file with the sequences to predict. The output is a .txt file contain the predicted score for each of the input sequences. The first score represents the probability of the sequence being a lncRNA, and the second score represents the probability of the sequence being a coding RNA.

> python CPCNN.py training_lnRNA.fasta training_cds.fasta testing.fasta output_directory

There are four inputs in the command. The first input “training_lnRNA.fasta” is the lncRNA sequences used for training. The second input “training_cds.fasta” is the coding sequences used for training. The third input “testing.fasta” is the sequences to predict. The fourth input “output_directory” is the directory of the results being put.

As an example, the package provides two training data sets containing 1000 lncRNA sequences and 1000 coding sequences in the directory of “/test_data/”. Also, 1000 sequences to predict were provided in the directory of “/test_data/”. Then, one can make a directory “/test_data/” in your current directory, and used this command:

> python CPCNN.py ./test_data/training_lncRNA.fasta ./test_data/training_cds.fasta ./test_data/test.fasta ./test_data/
