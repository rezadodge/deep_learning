
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.251908396946565
    Number of lines: 4258
    Average number of words in each line: 11.50164396430249
    
    The sentences 0 to 10:
    
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    

## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    vocab = set(text)
    int_to_vocab = {i:w for (i,w) in enumerate(vocab)}
    vocab_to_int = {value:key for (key, value) in int_to_vocab.items()}
    
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """

    symbols_dict = {'.': "||Period||", ',': "||Comma||", '"': "||Quotation_Mark||", ';': "||Semicolon||", 
                    '!': "||Exclamation_Mark||", '?': "||Question_Mark||", '(': "||Left_Parentheses||",
                    ')': "||Right_Parentheses||", '--': "||Dash||", '\n': "||Return||"}

    return symbols_dict

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.3.0
    

    C:\Users\mrd\Anaconda3\envs\tensorflow\lib\site-packages\ipykernel_launcher.py:14: UserWarning: No GPU found. Please use a GPU to train your neural network.
      
    

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None,None], name="input")
    targets = tf.placeholder(tf.int32, [None,None], name="target")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return inputs, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed
    


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    ''' ONLY FOR MORE COMPLICATED CELLS
    # 
    # cell parameters
    num_layers = 2
    dropout = 0.4

    def build_lstm_cell(lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        
        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([build_lstm_cell(rnn_size, 1-dropout) for _ in range(num_layers)])
    '''
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name="initial_state")
    
    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed
    

### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
# The module my_rnn is taken from tensorflow 1.3 and changed slightly so that the next function in this notebook, 
# i.e. build_nn() will pass the test! A conditional statement in the method _best_effort_input_batch_size(), 
# which is called from dynamic_rnn is removed, so that the batch_size falls back to the dynamic batch size 
# of its first input

import my_rnn

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    ###### produces final_state with wrong shape in tensorflow 1.3 ##########
    # output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) 
    #########################################################################
    output, final_state = my_rnn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")

    return output, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed
    

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embeded_inputs = get_embed(input_data, vocab_size, embed_dim)
    output, final_state = build_rnn(cell, embeded_inputs)
    logits = tf.contrib.layers.fully_connected(output, vocab_size, activation_fn=None)
    
    return logits, final_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed
    

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    ints_per_batch = batch_size * seq_length
    batches_number = len(int_text) // ints_per_batch
    
    # Keep only enough words to make full batches
    int_words = np.array(int_text[:batches_number * ints_per_batch])
    int_words = int_words.reshape([batch_size, -1])
    
    batches = np.ndarray(shape=[batches_number, 2, batch_size, seq_length], dtype=np.int32)
    for n in range(0, batches_number):
        ind1 = n * seq_length
        ind2 = ind1 + seq_length
        x = int_words[:, ind1:ind2]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        try:
            y[:, -1] = int_words[:, ind2]
        except IndexError:
            for j in range(batch_size-1):
                y[j,-1] = int_words[j+1,0]
            y[-1,-1] = int_words[0,0]
    
        batches[n][0] = x
        batches[n][1] = y
    
    return batches


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 50
# Batch Size
batch_size = 64
# RNN Size
rnn_size = 1000
# Embedding Dimension Size
embed_dim = 150
# Sequence Length
seq_length = 11
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 71

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forums](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/98   train_loss = 8.821
    Epoch   0 Batch   71/98   train_loss = 5.970
    Epoch   1 Batch   44/98   train_loss = 5.525
    Epoch   2 Batch   17/98   train_loss = 5.143
    Epoch   2 Batch   88/98   train_loss = 5.029
    Epoch   3 Batch   61/98   train_loss = 4.715
    Epoch   4 Batch   34/98   train_loss = 4.704
    Epoch   5 Batch    7/98   train_loss = 4.493
    Epoch   5 Batch   78/98   train_loss = 4.219
    Epoch   6 Batch   51/98   train_loss = 4.162
    Epoch   7 Batch   24/98   train_loss = 4.000
    Epoch   7 Batch   95/98   train_loss = 3.911
    Epoch   8 Batch   68/98   train_loss = 3.698
    Epoch   9 Batch   41/98   train_loss = 3.491
    Epoch  10 Batch   14/98   train_loss = 3.466
    Epoch  10 Batch   85/98   train_loss = 3.376
    Epoch  11 Batch   58/98   train_loss = 3.279
    Epoch  12 Batch   31/98   train_loss = 2.958
    Epoch  13 Batch    4/98   train_loss = 2.870
    Epoch  13 Batch   75/98   train_loss = 2.916
    Epoch  14 Batch   48/98   train_loss = 2.861
    Epoch  15 Batch   21/98   train_loss = 2.605
    Epoch  15 Batch   92/98   train_loss = 2.478
    Epoch  16 Batch   65/98   train_loss = 2.392
    Epoch  17 Batch   38/98   train_loss = 2.160
    Epoch  18 Batch   11/98   train_loss = 2.125
    Epoch  18 Batch   82/98   train_loss = 2.008
    Epoch  19 Batch   55/98   train_loss = 1.888
    Epoch  20 Batch   28/98   train_loss = 1.887
    Epoch  21 Batch    1/98   train_loss = 1.654
    Epoch  21 Batch   72/98   train_loss = 1.658
    Epoch  22 Batch   45/98   train_loss = 1.619
    Epoch  23 Batch   18/98   train_loss = 1.502
    Epoch  23 Batch   89/98   train_loss = 1.311
    Epoch  24 Batch   62/98   train_loss = 1.343
    Epoch  25 Batch   35/98   train_loss = 1.183
    Epoch  26 Batch    8/98   train_loss = 1.175
    Epoch  26 Batch   79/98   train_loss = 1.137
    Epoch  27 Batch   52/98   train_loss = 1.036
    Epoch  28 Batch   25/98   train_loss = 0.990
    Epoch  28 Batch   96/98   train_loss = 0.902
    Epoch  29 Batch   69/98   train_loss = 0.885
    Epoch  30 Batch   42/98   train_loss = 0.772
    Epoch  31 Batch   15/98   train_loss = 0.696
    Epoch  31 Batch   86/98   train_loss = 0.733
    Epoch  32 Batch   59/98   train_loss = 0.678
    Epoch  33 Batch   32/98   train_loss = 0.615
    Epoch  34 Batch    5/98   train_loss = 0.584
    Epoch  34 Batch   76/98   train_loss = 0.565
    Epoch  35 Batch   49/98   train_loss = 0.543
    Epoch  36 Batch   22/98   train_loss = 0.479
    Epoch  36 Batch   93/98   train_loss = 0.552
    Epoch  37 Batch   66/98   train_loss = 0.555
    Epoch  38 Batch   39/98   train_loss = 0.464
    Epoch  39 Batch   12/98   train_loss = 0.460
    Epoch  39 Batch   83/98   train_loss = 0.458
    Epoch  40 Batch   56/98   train_loss = 0.458
    Epoch  41 Batch   29/98   train_loss = 0.387
    Epoch  42 Batch    2/98   train_loss = 0.447
    Epoch  42 Batch   73/98   train_loss = 0.483
    Epoch  43 Batch   46/98   train_loss = 0.419
    Epoch  44 Batch   19/98   train_loss = 0.432
    Epoch  44 Batch   90/98   train_loss = 0.465
    Epoch  45 Batch   63/98   train_loss = 0.403
    Epoch  46 Batch   36/98   train_loss = 0.400
    Epoch  47 Batch    9/98   train_loss = 0.390
    Epoch  47 Batch   80/98   train_loss = 0.383
    Epoch  48 Batch   53/98   train_loss = 0.374
    Epoch  49 Batch   26/98   train_loss = 0.357
    Epoch  49 Batch   97/98   train_loss = 0.459
    Model Trained and Saved
    

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    initial_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    probs_tensor = loaded_graph.get_tensor_by_name("probs:0")
    
    return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed
    

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return int_to_vocab[np.argmax(probabilities)]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed
    

## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 300
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'homer_simpson'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[0][dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    INFO:tensorflow:Restoring parameters from ./save
    homer_simpson:(confused) hey, what's going on? that guy's fat tony.
    homer_simpson: why don't you come too?
    moe_szyslak:(serious, homer) ah, you little yellow-belly. what's the big person.
    moe_szyslak: oh, that's it... uh, that's just...
    moe_szyslak:(sighs) so you all hate me?
    moe_szyslak:(nods) okay, i got.
    carl_carlson: oh, i'm afraid lenny's dead.
    moe_szyslak:(friendly) really?
    homer_simpson:(murderously) i am the guys!
    
    
    barney_gumble:(to moe, in) hey, the? is that says big, uh... uh... oh, nuts. they're both friends.
    moe_szyslak: nope, homer.
    homer_simpson:(inanely) whoa, what's the best...
    chief_wiggum:(motorcycle noises) yeah, moe, i've been seen barney.
    moe_szyslak:(excited) i am so about this, moe.
    moe_szyslak:(struggling shotgun) i knew the booze.
    moe_szyslak:(under it, then clearly) the booze! and feeling...
    moe_szyslak:(excited) or i wouldn't.
    
    
    moe_szyslak:(to patrons) i won't tell if you and, your friend... swings to have about that.
    barney_gumble:(annoyed) you do it?!
    homer_simpson:(sings) i know, that's it. the costume's buried on the pickle.
    moe_szyslak: yeah, i did it was either cable the mechanical.
    moe_szyslak: you got to this friend!
    homer_simpson:(
    

# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the beginning of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
