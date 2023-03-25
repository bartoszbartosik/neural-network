# FEEDFORWARD NEURAL NETWORK
## Hello!
In the files above you'll find the most basic feedforward neural network. It has been created without usage of external libraries, like tensorflow, so that you could delve into my interpretation of this wonderful tool's principle of operation.

## Maths
The neural network learns by a backpropagation algorithm. Its purpose is to determine the cost function gradient with respect to the network's both weights and biases (separately). Equations breathing life into plain neural network architecture have been listed below:
```math
\delta^{L} = \nabla_{a}C \odot \sigma ' (z^{L}) = (a^{l} - y) \odot \sigma ' (z^{L})
```
```math
\delta^{l} = ((w^{l+1})^{T}\delta^{l+1}) \odot \sigma ' (z^{l})
```
```math
\frac{\partial C}{\partial b_{j}^{l}} = \delta_{j}^{l} 
```
```math
\frac{\partial C}{\partial w_{jk}^{l}} = w_{k}^{l-1} \delta_{j}^{l} = a^{l-1} \otimes (\delta^{l} \sigma ' (z^{L}))
```

These nicely formed equations appears in the code in a following form:
```math
\delta^{L} = (a^{L} - y) \odot \phi ' (z^{L})
```
```math
\delta^{l} = ((w^{l+1})^{T}\delta^{l+1}) \odot \phi ' (z^{l})
```
```math
\frac{\partial C}{\partial b^{l}} = \delta^{l} 
```
```math
\frac{\partial C}{\partial w^{l}} = a^{l-1} \otimes (\delta^{l} \phi ' (z^{L}))
```

where:
- $\delta^{L}$: error in the output layer,
- $\delta^{l}$: error in the l-th layer,
- $z^{L}$: input values of the output layer,
- $z^{l}$: input values of the l-th layer,
- $a^{L}$: output vector of output layer,
- $a^{l}$: output vector of l-th layer,
- $y$: expected output,
- $\phi()$: activation function
- $\frac{\partial C}{\partial b^{l}}$: cost function gradient with respect to the netowork's biases
- $\frac{\partial C}{\partial w^{l}}$: cost function gradient with respect to the netowork's weights

Above equations are used in the Stochastic Gradient Descent, where weights and biases are updated.


