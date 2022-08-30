# Neural Networks: Learning

## 1. Notation and number of output units
Neural Networks are widely used to solve classification problems. For binary classification, the number of output units is equal to $1$. For multiclass classification with $K$ classes, there are $K$ output units. We consider the following notation:

* L = total number of layers in the network
* $s_{l}$ number of units (excluding the bias unit) in layer $l$
* K = number of output units/classes.

## 2. Cost function: 
### 2.1 Logistic Regression as starting point
We recall the final regularized form of the cost function for Logistic Regression:
$
\begin{aligned}
    J(\theta) = -\frac{1}{m} \cdot \sum _{i=1}^{m} [y ^ {(i)} \cdot \log(h_{\theta} (x)) + (1 - y) \cdot \log(1 - h_{\theta} (x))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j ^ 2 
\end{aligned}
$

For neural network, there might be several output units. We denote $h_{\Theta}(x)_j$ as the hypothesis resulting in the $j$-th output unit. The final forms is as follows:

$
\begin{aligned}
    J(\theta) = -\frac{1}{m} \cdot \sum _{i=1}^{m}
    \sum_{k=1}^{K} [y ^ {(i)} \cdot \log(h_{\Theta}(x)_k) + (1 - y) \cdot \log(1 - h_{\Theta}(x)_k)] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\Theta_{j,i}^{(l)})^2
\end{aligned}
$
* The first nested summation calculates the logistic regression cost function fo each of the output units.
*  To better understand the second nested summation, we recall that the matrix $\Theta^{(i)}$ is the matrix mapping the values of layer $i$ to the layer $i + 1$. Thus the first summation iterates through all the $L - 1$ mapping matrices.
* Additionally, $\Theta^{(i)}$ is of dimensions $s_{i + 1} * (s_{i} + 1)$ Therefore, two inners sums, calculate the element-wise square $\Theta^{(i)}$ without considering the terms corresponding to the bias unit.
