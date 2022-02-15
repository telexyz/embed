echo "(( word2vec #nhà_làm ))"
# zig run src/sampling.zig -O ReleaseFast
zig run src/training.zig -O ReleaseFast

# word2vec
echo "(( word2vec nguyên bản ))"
if [ ! -f "data/wordvec.out" ]; then
time zig run xxx2vec/word2vec.c -O ReleaseFast -- -train data/corpus.txt -output data/wordvec.out -threads 4 -iter 8 -min-count 3 -negative 10 -size 256 -window 5 -read-vocab data/vocab.txt
fi

# ngram2vec
# ./ngram2vec.sh

# Xem kết quả của word2vec vs word2vec nguyên bản vs ngram2vec
zig run src/similar.zig -O ReleaseFast

# # - - - - -
# # Binarize
# # - - - - -
# # brew install openblas
cd xxx2vec/binarize && make && cd ../..

xxx2vec/binarize/binarize -input data/vocab.vec -output data/binary_vocab.vec -n-bits 256 -lr-rec 0.001 -lr-reg 0.001 -batch-size 75 -epoch 8
xxx2vec/binarize/topk_binary data/binary_vocab.vec 20 trí_tuệ thực_hành hạnh_phúc


xxx2vec/binarize/binarize -input data/wordvec.out -output data/binary_wordvec.out -n-bits 256 -lr-rec 0.001 -lr-reg 0.001 -batch-size 75 -epoch 8
xxx2vec/binarize/topk_binary data/binary_wordvec.out 20 trí_tuệ thực_hành hạnh_phúc

# # - - - - -
# # josh
# # - - - - -
# ./josh.sh
