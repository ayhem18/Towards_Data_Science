# 1. Linear Regression in General
## 1.1 Main idea
Linear regression is, as the name suggests, a supervised learning algorithm: a regression. The main idea is to fit the provided data into one line. The algorithm mainly finds the line with the minimal numerical error.

## 1.2 Notation
**m** generally refers to the number of training examples, samples

**x's** refer to an input variable, a feature

**y's** refer to an output variable, a target. 
More specifically, (x^(i) , y^(i)) represent the ith feature and target in the given sample data.

# 2. One variable Linear Regression

## 2.1 Main idea
given a large set of features **x's**, we would like our algorithm to express **y =  a * x + b**. The coefficients are chosen such that
the cost function is minimized

## 2.2 cost function and mathematical background

Please refer to the following link: https://github.com/ayhem18/Towards_Data_science/blob/master/Machine%20Learning/Linear%20Regression/math_parts/multi_var_LR_math.pdf
### 2.2.1 Remarques 
In addition to the analytically calculated solution, we can use optimization algorithms such as Gradient Descent (This algorithm is quite usuful since the square error metric function is a *convex* function.)

# 3 Multi variable regression
## 3.1 Notation
Additional notation should be introduced in this case:

$n$ = number of features: number of $x$'s per sample

$x^{(i)}$: the vector of features of the $i$-th training sample

$x_j^{(i)}$: the value of the $j$-th vaylue in the $i$-th training sample.
## 3.2 Main idea
Given a set of data with $n$ features $x$'s and a target $y$, we would like to approximate the target $y$ to a linear combination of the features. In other words,  express the prediction $h$ (or the hypothesis) as 

$\begin{aligned}
    h(\theta) = \theta_0 + \theta_1 \cdot x_1 + .. \theta_n \cdot x_n
\end{aligned}$

## 3.3 Cost function and mathematical background
Please refer to the following link: https://github.com/ayhem18/Towards_Data_science/blob/master/Machine%20Learning/Linear%20Regression/math_parts/Multi_var_LR_math.pdf


# 4. Features preprocessing
In any dataset, features do not necessarily vary within the same range of values. For example, considering a dataset of home prices, the size (in meters squared) might varies within the range in [0, 400] while the number of bedrooms varies within range [1, 15]. There is no reason to believe that a larger value should have a larger corresponding parameter. In addition, having features with significantly different range of values might be slow down numerical optimization algorithms.Thus some preprocessing might reveal necessary
## 4.1 Feature scaling
a feature is mapped to the range [0, 1] (or [-1, 1]) where $x_{i~new} = \frac{x_{i~old}}{max(x)}$. Not all features need to be scaled as certain ranges such as [-3, 3] and [$-\frac{1}{3}$, $\frac{1}{3}$] are not much different. However, larger ranges might need to be scaled.

## 4.2 Mean Normalization
it is possible to replace every feature $x_i$ (apart from $x_0$ as $x_0 = 1$) to $\frac{x_i - \mu_i}{max(x)}$ where $\mu_i = \frac{max(x) - min(x)}{2}$

## 4.3 Feature Scaling and Mean Normalization
Additionally we can consider $x_i = \frac{x_i - \bar x}{max(x) - min(x)}$ where $\bar x$ is the mean value. 

