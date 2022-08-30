# Classification and Logistic Regression
## 1. Binary Classification
This is a type of problems where target values are either 1 or 0. For such problems, Linear Regression is not the perfect solution for several reasons:

1. Due to the binary nature of target values, a treshhold $t$ should be chosen: if $h_{\theta} (x) \ge t$ then it is intrepreted as $1$ and $0$ otherwise. Such treshhold is significantly affected by outliers. Therefore $t$ might get too large contradicting what most of the data indicates.

2. even if the feature data is between 0 and 1, the resulting prdecition might not fell in the range between 0 and 1.

## 2 Logistic Regression
Logistic regression computes weights. However, the hypothesis $h_{\theta} (x) $ is no longer $ \theta ^ T\cdot x$ but $g(\theta ^ T \cdot x)$ where $g$ is a mathematical function mapping $\mathbb{R}$ to the interval $(0,1)$

### 2.1 Sigmoid function
let's consider the function $f(x) = \frac{L}{1 + e^{- k \cdot (x - x_0)}}$. The parameters are defined as follows:

1. $L$: The maximum of the function
2. $x_0$: the midpoint of the function
3. $k$: The growth rate

For $L = 1$ , $x_0 = 0$, $k = 1$, we define the *sigmoid* function: 
$\begin{aligned}
    f(x) = \frac{1}{1 + e^{-x}}
\end{aligned}
$

### 2.3 Hypothersis interpretation
We have
$\begin{aligned}
h_{\theta} (x) = \sigma(\theta ^ T \cdot x) =P(y = 1 |x, \theta)
\end{aligned}$ 
In other words, the result is interpreted as the probability that $y = 1$ given the vector of features $x$, parameterized by $\theta$. 

Since $y$ is either $1$ or $0$, the following holds: 
$\begin{aligned}
    P(y = 1 |x, \theta) = 1 - P(y = 0 |x, \theta)
\end{aligned}$ 

Conventionally, classification takes place according as follows:

$h_{\theta} (x) \ge 0.5 \rightarrow y = 1$ and 
$h_{\theta} (x) \le 0.5 \rightarrow y = 0$. 

By analysing the sigmoid function we can see such statements are equivalent to:

$\theta \cdot x \ge 0 \rightarrow y = 1$ and 
$\theta \cdot x \le 0 \rightarrow y = 0$.

The model can be expanded to represent more complex shapes by introducing new features $x_k = x_{i_1} ^ {a_1} \cdot x_{i_2} ^ {a_2}... \cdot x_{i_n} ^ {a_n} $ for some natural numbers $a_1, a_2.. a_n$.

### 2.4 Cost function and mathematical background
#### 2.4.1 Linear Regression as a starting point
The *cost function* or (metric) used to evaluate a linear regression model can be expressed as 

$\begin{aligned}
    J = \frac{1}{2m} \sum _{i=1}^{m}(y_i - h_{\theta} (x^i))^2 
\end{aligned}$ 

let's consider $J(y, x, \theta) = (y - h_{\theta} (x))^2$. for $h_{\theta} (x) = \frac{1}{1 + e ^ {-\theta \cdot x}}$
The function $J$ is ***non-convex***. The numerical approach is likely to fail for such cases. Therefore, a slightly different
$ J(y, x, \theta) $ should be introduced.

#### 2.4.2 The final cost function
The final cost function can be expressed as follows:

$\begin{aligned}
    J(\theta) = -\frac{1}{m} \cdot \sum _{i=1}^{m} [y ^ {(i)} \cdot \log(h_{\theta} (x)) + (1 - y) \cdot \log(1 - h_{\theta} (x))]
\end{aligned}$ 
The mathematical derivation can be found in the following link: https://github.com/ayhem18/Towards_Data_science/blob/master/Machine%20Learning/classification_and_logistic_regression/math_parts/Logistic_Regression_math.pdf

## 3. Multiclass classification
### 3.1 Extending Logistic regression
The mechanism provided by Logistic regression can be manipulated using the ***One Vs all*** method.
#### 3.1.1 Ons Vs All Method
Let's consider $n$ labels, denoted by numbers from $1$ to $n$. We consider $n$ versions where at each we consider one class as $1$ while the rest of the classes as a single class represented as $0$. Through $n$ iterations, we produce $n$ hypothesis 

$\begin{aligned}
h^i(\theta)  = P(y = i|x, \theta) , ~ i = 1, 2 ....n \\
prediction = \mathop{max}_{i} h^i(\theta)
\end{aligned} 
$

## 4 The Overfitting problem
Among the model's performance metrics, is how well it fits the training data and how well it is expected to fit new unseen data. Therefore, a model might range from ***underfit*** to ***overfit***:
1. ***underfit*** or ***high bias*** is when the model does not fit the training data well enough. *bias* term denotes that the model might incorporate assumptions made by the model's designder.
2. ***overfit*** or ***high variance*** is when the model fits quite well with the data (perfectly possibly). Yet, it fails to be general enough to produce accurate predictions for unseen data. the term *variance* expresses the large variability required to fit the training data to a large extent.

This problem might be addressed by two approaches.
1. Reduce the number of features either manually or by a model selection algorithm
2. Apply ***regularization***: keeping all features while decreasing their magnitudes
