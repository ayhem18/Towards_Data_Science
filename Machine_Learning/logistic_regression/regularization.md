# Regularization

## 1. Intuition
There are two main advantages for smaller values of $\theta_1$, $\theta_2$... $\theta_n$: 
1. ***Simpler*** hypothesis
2. Less prone to ***overfitting***

### 1.1 The modified cost function
The new cost function can be expressed as follows:
$\begin{aligned}
J_{new}(\theta) = J_{old}(\theta) + \frac{\lambda}{2m} * \sum_{j = 1}^{n} \theta_j
\end{aligned}$ 
where $c$ is the constant term (assuming $J$ is a sum or a product).
### 1.2 Explanation
The sum introduced is from $1$ to $n$ where $n$ is the number of features par training example. According to the convention, the first parameter/weight $\theta_0$ is not panalized. Therefore the sum starts from $1$. The term $\lambda$ is referred to as the ***regularization*** parameter. Roughly speaking, with the regularization term, the optimazation algorithm tends to set a certain number of parameters $\theta$ to values close to $0$ resulting in slightly less fit to the training data but general enough for accurate future predictions.
### 1.3 Notes
It is note worthy that for extremely large values of $\lambda$, the model ends up ***underfitting*** the data. A large regularization parameter means a large penalty for all parameters. Therefore, the model  assigns infinitesimal values for most parameters which oversimplies the model.

## 2. Regularized Linear Regression
### 2.1 Gradient descent
The update part of the algorithm can be expressed as folows: 

$\begin{aligned}
\theta_0 := \theta_0 - \alpha * \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)x_0^i\\
\theta_j := \theta_j*(1 - \alpha * \frac{\lambda}{m}) - \alpha * \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)x_j^i 
\end{aligned}$ 

### 2.1 Analytical Solution: Normal Equation
The introduction of the regularization term slightly modifies the analytical solution

$\begin{aligned}
\theta = (X ^ T * X + \lambda * 
\begin{bmatrix} 
0 && .. && .. && 0 \\
0 && 1 && .. && 0 \\
0 && 0 && 1.. && 0 \\
0 && 0 && .. && 1
\end{bmatrix})^{-1} X ^ T  y
\end{aligned}$ 
It is possible to prove that the matrix is always invertible for all values $\lambda > 0$.

## 3. Regularized Logistic Regression
### 3.1 The Cost Function
The regularized cost function can be expressed as follows:
$\begin{aligned}
    J(\theta) = -\frac{1}{m} \cdot \sum _{i=1}^{m} [y ^ {(i)} \cdot \log(h_{\theta} (x)) + (1 - y) \cdot \log(1 - h_{\theta} (x))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j ^ 2 
\end{aligned}$ 

### 3.2 The Gradient descent
Similarly to linear regression, the update part of the gradient descent can be expressed as follows

$\begin{aligned}
\theta_0 := \theta_0 - \alpha * \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)x_0^i\\
\theta_j := \theta_j*(1 - \alpha * \frac{\lambda}{m}) - \alpha * \frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)x_j^i 
\end{aligned}$ 

