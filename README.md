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

![Screenshot from 2021-11-25 18-23-18](https://user-images.githubusercontent.com/57453637/143446292-9b86df2b-f213-407e-9e0e-b7730a0a0957.png)
Results on Validation Set : Accuracy : 100%

Training Logs: 
![Screenshot from 2021-11-25 18-32-02](https://user-images.githubusercontent.com/57453637/143446595-08063f19-35f6-4abc-a987-5e31c0e96e7b.png)
![Screenshot from 2021-11-25 18-32-38](https://user-images.githubusercontent.com/57453637/143446605-0a2832dc-e6bd-466d-b479-b8b26cd94c8d.png)
![Screenshot from 2021-11-25 18-32-59](https://user-images.githubusercontent.com/57453637/143446615-fae60f59-f2bc-4ede-957a-d3c23bca1d2e.png)


## Instructions to run
``` python3 train.py --config <path_to_config.yaml>```

## In config file
* To change parameters for training : change the `lr` (learning rate), and `n_iter` (number of iterations)
* Before running give the correct paths to the training and validation csv files in the corresponding locations
