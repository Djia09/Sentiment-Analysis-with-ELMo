# ELMo-Embedding

We tested ELMo embedding on the Kaggle Challenge: Sentiment Analysis on Movie Reviews. 

## Dataset
The dataset is available on https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/.
It gives reviews in english, categorized in 5 classes: 
* 0 - negative
* 1 - somewhat negative
* 2 - neutral
* 3 - somewhat positive
* 4 - positive

## ELMo
ELMo is a deep contextualized word representation that models both complex characteristics of word use (e.g., syntax and semantics), and how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large Wikipedia corpus. 
ELMo embedding method is detailed on [Peters et al. (2018)](https://arxiv.org/abs/1802.05365) article.

The model is available on AllenNLP, an open-source NLP library built on PyTorch. More details on [AllenNLP](https://www.semanticscholar.org/paper/A-Deep-Semantic-Natural-Language-Processing-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d) article.

## Details
Our personal submissions were given with GloVe model with dimension 50 and 200, and also with ELMo model.

### Prerequisites
* Python 3
* Conda (for creating a clean environment, but you can also create it with virtualenv)

### Installing
Installation was tested on Ubuntu 18.04 and Windows 10, with Python 3.
For basic libraries:
```
pip install -r requirements.txt
```
For AllenNLP library, the detailed steps are given in https://github.com/allenai/allennlp. 
Basically, you just need to run on a terminal:
```
conda create -n allennlp python=3.6
source activate allennlp
pip install allennlp
```

### Pre-trained model 
ELMo Pre-trained model and their description are given on https://allennlp.org/elmo. 
To reproduce our results, you need to download [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5) and [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json).

GloVe pre-trained models can be found [here](https://nlp.stanford.edu/projects/glove/) and the details about GloVe models are given in the article [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

## Running the test
The file GloVe.py load and create a GloVe model.

To get the submission.csv in Kaggle format, just run:
```
python elmo-svm.py
```
## Author
David JIA
