# <span style="color:darkred"> Language Translation using Recurrent Neural Network </span>
In this project, a Recurrent Neural Network (RNN) is built and train as a sequence to sequence model. 
The network is trained on a dataset of English and their equivalent French sentences so that it can translate 
new sentences from English to French. 
To limit the training time and resources needed, the vocabulary is limited to a small number of words.

### <span style="color:darkred"> Sample translation </span>
_**Input**_ 

    Word Ids:      [42, 60, 49, 65, 62, 224, 212]
    English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']

_**Prediction**_

    Word Ids:      [225, 299, 334, 212, 42, 117, 244, 100, 1]
    French Words: il a vu le vieux camion jaune . <EOS>
