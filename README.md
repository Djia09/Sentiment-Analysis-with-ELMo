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

### Prerequisites
Python 3
Conda (for creating a clean environment, but you can also create it with virtualenv)

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
## Running the test
To get the submission.csv in Kaggle format, just run:
```
python main.py
```
