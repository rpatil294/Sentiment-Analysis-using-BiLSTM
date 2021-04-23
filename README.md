# Sentiment-Analysis-using-BiLSTM
Employing BiLSTM to categorize movie reviews using PyTorch to get 90%+ accuracy 


We'll be building an RNN-based learning model to detect sentiment (i.e. detect if a sentence is positive or negative) using PyTorch and TorchText. This will be done on movie reviews, using the IMDb dataset.

We will use:

1. pre-trained word embeddings
2. different RNN architecture
3. bidirectional RNN
4. multi-layer RNN
5. regularization
6. optimizer

Instead of having our word embeddings initialized randomly, they are initialized with these pre-trained vectors. Here, we will use max vocab size of top 25000 most common words for quick training.

We get these vectors simply by specifying which vectors we want and passing it as an argument to `build_vocab`. `TorchText` handles downloading the vectors and associating them with the correct words in our vocabulary.

Here, we'll be using the `"fasttext.simple.300d" vectors"`. `300d` indicates these vectors are 100-dimensional.

The theory is that these pre-trained vectors already have words with similar semantic meaning close together in vector space, e.g. "terrible", "awful", "dreadful" are nearby. This gives our embedding layer a good initialization as it does not have to learn these relations from scratch.

We'll use a BucketIterator which is a special type of iterator that will return a batch of examples where each example is of a similar length, minimizing the amount of padding per example.

We also want to place the tensors returned by the iterator on the GPU (if you're using one). PyTorch handles this using torch.device, we then pass this device to the iterator.

Another thing for packed padded sequences all of the tensors within a batch need to be sorted by their lengths. This is handled in the iterator by setting sort_within_batch = True.

 Standard RNNs suffer from the vanishing gradient problem. LSTMs overcome this by having an extra recurrent state called a cell,  𝑐  - which can be thought of as the "memory" of the LSTM - and they use use multiple gates which control the flow of information into and out of the memory. For more information, go here. While RNN are can be written as a function of  𝑥𝑡  and  ℎ𝑡
 
 We use the Adam optimizer while training. This optimizer adapts the learning rate for each parameter, giving parameters that are updated more frequently lower learning rates and parameters that are updated infrequently higher learning rates.
 
 Our model currently outputs an unbound real number. As our labels are either 0 or 1, we want to restrict the predictions to a number between 0 and 1. We do this using the sigmoid or logit functions.

We then use this this bound scalar to calculate the loss using binary cross entropy.

The BCEWithLogitsLoss criterion carries out both the sigmoid and the binary cross entropy steps.

We define the criterion and place the model and criterion on the GPU (if available) by using .to

