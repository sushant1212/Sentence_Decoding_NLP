# Sentence_Decoding_NLP

## Method Used

### Text PreProcessing
1. All the characters were converted to lower case
2. All punctuation marks were removed
3. Possessives were removed and replaced with an empty string
4. Using WordNet Lemmatizer, lemmatization was done on the train dataset.


### Embeddings
* I chose to use the Glove word embeddings.
* All the words were converted to their corresponding word embeddings using the pretrained Glove model. 
* The final embedding vector is a 50 dimensional vector

### Network Architecture
* The basic unit of the network is an GRU.
* The sentence would be fed into an GRU, and then the final output of the GRU will be stored.
* This final output is like an encoding of our input sentence.
* Using this encoding the model will predict the output.
* This model is on the same lines as the Seq2Seq model apart from the attention layer.

### Results

