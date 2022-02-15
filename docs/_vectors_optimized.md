## binary vectors

* The model architecture can transform any real-valued vectors to binary vectors of any size (e.g. 64, 128 or 256 bits to be in adequacy with CPU register sizes).

* The architecture has the ability to reconstruct original real-valued word vectors from the binary ones with high accuracy.

* Binary vectors use 97% less memory space than the original real-valued vectors with only a loss of ∼2% in accuracy on word semantic similarity, text classification and sentiment analysis tasks.

* A top-K query is 30 times faster with binary vectors than with real-valued vectors.

- - -

Classic real-valued word embeddings
- Millions of words in vocabulary
- Usually 300 dimensions per vector
- For 1M vectors, 1.2 gigabytes are required

Solution: binary word embeddings, i.e. associate a m-bits vector to each word of a vocabulary
* Faster vector operations
  - Cosine similarity requires O(n) additions and multiplications
  - Binary similarity requires a XOR and a popcount() operations

* Small memory size
  - 9600 bits per real-valued vector (300 dimensions, 32 bits per value)
  - 128 or 256 bits per binary vector
  - 16.1 MB to store 1M vectors of 128 bits (vs. 1.2 GB)
  - Computations can be done locally; no need of sending data to computing servers


### NLB: Near-lossless binarization of word embeddings

https://github.com/tca19/near-lossless-binarization

NHỎ HƠN, NHANH HƠN, THI THOẢNG TỐT HƠN => DÙNG LUÔN !!!

* Best average results achieved with 512 bits (18 times smaller)

* Binary vectors are on par or slightly outperform real-valued original vectors !!!

* Using 256-bit vectors is 30 times faster than using real-valued vectors
  - When the loading time is taken into account, they are 75 times faster

!!! REDUCE THE VECTOR SIZE BY 37.5 WITH ONLY A 2% PERFORMANCE LOSS !!!
