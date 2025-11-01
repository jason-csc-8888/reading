# Neural Nets

Most things are layers over layers of neurons. 

### Neuron

> Each neuron is just a function that takes outputs of neurons from the previous layer and spits out an output between 0 and 1 or any range. 

It is made up of a weighted sum as in : activation(sum of weights*input activations + bias)

Input activations $a_1 ... a_n$ are just activations from the previous layer that looks like `output = activation_function(z)`. If it's the first layer then they are just raw input features $a_1 ... a_n$ === pixels / sensor readings / text embeddings etc.

Weights are dials and nobs that can adjust the activations: $$w_1a_1+w_2a_2+...+w_na_n$$

Bias is added to the weighted sum such that we only want the neuron activated when its activated to a degree > the bias. $w_1a_1+w_2a_2+...+w_na_n + bias$

These things are then squished into a normalized range. Activation ouputs are usually activated between range $[0, 1]$ or $[-1,1]$.

In the end we get:
`output = activation_function(z)`