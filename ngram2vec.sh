#!/bin/sh

# brew install miniconda
# conda config --env --add channels conda-forge
# conda install numpy scipy
#
# source .save/active-miniconda.sh

if [ ! -f "data/corpus.txt" ]; then
	zig run src/sampling.zig -O ReleaseFast
fi

if [ ! -f "data/ngram2vec_pairs" ]; then
	python xxx2vec/ngram2vec/corpus2vocab.py --corpus_file data/corpus.txt --vocab_file data/ngram2vec_vocab --memory_size 4 --feature ngram --order 2 --min_count 3

	python xxx2vec/ngram2vec/corpus2pairs.py --corpus_file data/corpus.txt --pairs_file data/ngram2vec_pairs --vocab_file data/ngram2vec_vocab --processes_num 2 --cooccur ngram_ngram --input_order 2 --output_order 2 --win 6 --sub 0

	# Concatenate pair files. 
	if [ -f "data/ngram2vec_pairs" ]; then
		rm data/ngram2vec_pairs
	fi
	for i in $(seq 0 1)
	do
		cat data/ngram2vec_pairs_${i} >> data/ngram2vec_pairs
		rm data/ngram2vec_pairs_${i}
	done

	# Generate input vocab and output vocab, which are used as vocabulary files for all models
	python xxx2vec/ngram2vec/pairs2vocab.py --pairs_file data/ngram2vec_pairs --input_vocab_file data/ngram2vec_vocab.input --output_vocab_file data/ngram2vec_vocab.output
fi

# SGNS, learn representation upon pairs.
time zig run xxx2vec/ngram2vec.c -O ReleaseFast -- --pairs_file data/ngram2vec_pairs --input_vocab_file data/ngram2vec_vocab.input --output_vocab_file data/ngram2vec_vocab.output --input_vector_file data/ngram2vec_sgns.input --output_vector_file data/ngram2vec_sgns.output --threads_num 4 --size 256 -iter 8

# SGNS evaluation.
# python ngram2vec/similarity_eval.py --input_vector_file data/sgns.input  --test_file testsets/similarity/ws353_similarity.txt --normalize

# python ngram2vec/analogy_eval.py --input_vector_file data/sgns.input --test_file testsets/analogy/semantic.txt --normalize