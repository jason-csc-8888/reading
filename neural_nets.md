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

### Gradient descent

Weights and biases are randomized initially so you need a "cost" function to tell the model what is right / wrong. Most neural nets are differentiable unless they are discrete decisions or if/then logic. The non-differentiable functions are usually replaced with relaxations like softmax / surrogate losses.

> Differentiability =/= convexity or guaranteed global optimum. Deep learning optimizations are nonconvex (Convex optimization - linear regression or SVMs guarantees a unique global minimum while non-convex can have multiple local minimas). Convex functions are usually bowl shaped and GD guarantees a global minimum.

Cost function: $C(x,y)$ The idea of gradient descent is to:
$ compute \space \Delta C $ which is the gradient
and take a small step in the $ - \Delta C$ direction.

This thing $ - \Delta C$ will allow us to nudge the weights vector $W$ to the local minimum.

Here, `∇C = [∂C/∂w₁, ∂C/∂w₂, ..., ∂C/∂wₙ, ∂C/∂b₁, ∂C/∂b₂, ...]` is how we compute the gradient. 

### Back propagation