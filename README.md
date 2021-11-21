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
* The basic unit of the network is an LSTM.
* The sentence would be fed into an LSTM, and then the final output of the LSTM will be stored.
* This final output is like an encoding of our input sentence.
* Using this encoding the model will predict the output.

##### Due to errors faced, I could not proceed further. This is all that I have been able to complete.
