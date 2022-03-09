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


