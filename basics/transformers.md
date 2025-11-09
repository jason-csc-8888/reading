# Transformers

In most cases, transformers take high dimensional arrays (tensors) as inputs, progressively transformed into outputs through layers of attention and feedforward networks. The tunable parameters are weights and biases in these layers, that get matmul'd with the input tensors to produce output tensors. Weights are what defines the model while the data encodes what's being processed.

Let's run through the architecture with a single sentence like "The cat sat on the mat".

### Tokenization

First we can breakdown the sentence into tokens. Tokens are usually words or subwords. For example, "The cat sat on the mat" can be tokenized into ["The", "cat", "sat", "on", "the", "mat"]. In practice, tokenization is done using byte pair encoding (BPE) or similar algorithms that break down words into subwords based on frequency in the training corpus and it looks something like ["Th", "e", " cat", " sat", " on", " the", " mat"].

The model has a predefined vocabulary of tokens that it can recognize, usually like 50,000 words. The embedding matrix $W_E$ is a learned parameter of shape (vocab_size, embedding_dim), it has a single column for each token in the vocabulary, and each column is a learned vector representation of that token.

Just like how you can slice a 2d plane and map the 3d world onto it, the embedding matrix maps discrete tokens into a high dimensional space. If you map it to a 3d space you can see directions in that space that correspond to semantic meaning, like "king - man + woman = queen", and $E(aunt) - E(uncle) = E(woman) - E(man)$. For example if you do $E(Sushi) - E(Japan) + E(Germany) = E(Brahtwurst)$, as in the embedding space seems to have captured  cultural relationships. In 12,288 dimensions, these relationships are even more nuanced.

> Dot product $[v_1, v_2, ...] \cdot [w_1, w_2, ...] = v_1w_1 + v_2w_2 + ...$ shows similarity between vectors. Dot product is 0 if orthogonal, positive if similar, negative if opposite. Cosine similarity is dot product normalized by vector lengths, to get a measure of similarity that is independent of vector magnitude.

In GPT-3 Embedding space has 12,288 dimensions. So the embedding matrix has shape (50,000 vocab/token size, 12,288 embed dim). So there's ~617 million weights in the embedding matrix alone. 

> $W_E = \begin{bmatrix} | & | & & | \\ E(token_1) & E(token_2) & ... & E(token_{50000}) \\ | & | & & | \end{bmatrix}$

Next the goal is to empower this embedding matrix to encode meaning / context, but the goal of attention / transformers is to help words soak up meaning from surrounding words.

The network can only process a fixed vector size at a time. This is **context window size**. It limits how much text the model can incorporate when predicting the next token. For GPT-3, the context window size is 2048 tokens. This means the model can only consider 2048 tokens when generating the next token.

The unembedding matrix $W_U$ is another learned parameter of shape (embedding_dim, vocab_size). It maps the final output vectors back to token probabilities. It is often the transpose of the embedding matrix, i.e. $W_U = W_E^T$. so we have (12,288 embed dim, 50,000 vocab size). This means there's another ~617 million weights in the unembedding matrix.

> $W_U = \begin{bmatrix} - & E(token_1)^T & - \\ - & E(token_2)^T & - \\ & ... & \\ - & E(token_{50000})^T & - \end{bmatrix}$

Lastly before we move on, softmax:
> Softmax converts raw scores (logits) into probabilities that add up to 1 (probability distribution). It exponentiates each score (positive values) and normalizes by the sum of exponentials. This ensures all probabilities are positive and sum to 1. $softmax(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$.

**Temperature** is a hyperparameter that in the softmax $softmax(z_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$. Higher temperature (>1) makes the distribution more uniform (more random sampling), while lower temperature (<1) makes it peakier (setting it to 0 means all the weight goes to that value, we always get the same token).