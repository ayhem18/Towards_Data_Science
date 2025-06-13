# Overview

This is my attempt to solve the problems of the first chapter of Linear Algebra Done Right by 


Any proof on vector spaces will have to start from these axioms

![](images/img1.png)

## Notes

Additional important results to keep in mind:

- uniqueness of the additive inverse: for a given vector $v$, if there are 2 vectors $a, b$ such that $a + v = b + v = 0$, then $a = b = (-v)$


- for $0 \in F$, $$\forall v \in V, \implies 0 \cdot v = 0$$

- for $0 \in V$
    $$\forall a \in F \implies a \cdot v = 0$$



## 1.B: Vector Space defintion

### Problem 1

- keep in mind that uniquess of the additive inverse

$$
\begin{align*}
v + (-v) &= 0 && \text{by definition of the additive inverse for $v$} \\
(-v) + -(-v) &= 0 && \text{by definition of the additive inverse for $-v$} \\
\implies v + (-v) &= -(-v) + (-v) = 0

\end{align*}
$$

using the uniqueness of the additive inverse $$-(-v) = v$$

### Problem 2

2   Suppose a ∈ F, v ∈ V and av = 0.  Show that a = 0 or v = 0.  


- Main idea: if $a \neq 0, a \in F$, then $\exists ~ a^{-1} \in F$ such that $a \cdot a^{-1} = 1$

Let's assume that $a \neq 0$

$$
\begin{align*}
a \cdot v &= 0 \\
a^{-1} \cdot (a \cdot v) &= a^{-1} \cdot 0 = 0 \\
(a^{-1} \cdot a) \cdot v &= 0 ~~ \text{by multiplicative associativity} \\
1 \cdot v = v &= 0
\end{align*}
$$

and hence the result


### Problem 3

Given $v,w ∈ V$, explain why there exists a unique x ∈ V such that $v + 3x = w.$

Step 1. existence

$x = 3^{-1} \cdot (w + (-v))$ is a solution

$$
\begin{align*}
3 \cdot x &= 3 \cdot 3^{-1} \cdot (w + (-v)) \\ 
\implies 3 \cdot x &= 1 \cdot (w + (-v)) \\
\implies 3 \cdot x &= w + (-v) \\
\implies v + 3 \cdot x &= w
\end{align*}
$$

Let's suppose that $x_1$ satisfies the equation, let's prove that $x_1 = x$ 

$$
\begin{align*}
\implies v + 3 \cdot x_1 &= w \\
\implies 3 \cdot x_1 &= w + (-v) \\
\implies 3^{-1} \cdot 3 \cdot x_1 &= 3^{-1} \cdot (w + (-v)) \\
\implies x_1 &= 3^{-1} \cdot (w + (-v)) \\
\end{align*}
$$

$3^{-1} \cdot (w + (-v))$ is indeed unique since $-v$ is unique for a given $v$


### Problem 4

The empty set is not a vector space.  It fails exactly one axiom— which? 

The additive identity requires the existence of an element $0 \in V$, since the empty set has no elements, it cannot contain the additive identity. Therefore, it is not a vector space.


### Problem 5

Show that the additive-inverse axiom can be replaced by

$$   0 v = 0 \text{for every} ~~ v ∈ V.  $$

The proof is 2 directions: 

1. if a set of vectors satisfy the axioms of a vector space, then the condition above is satisfied
2. if a set of vectors satisfy the axioms of a vector space (except the additive-inverse axiom) + this condition, then the additive inverse axiom is satisfied. 


- Dir 1:

$$
\begin{align*}
0 \cdot v &= (0 + 0) \cdot v = 0 \cdot v + 0 \cdot v \\
- (0 \cdot v) + 0 \cdot v &= - (0 \cdot v) + 0 \cdot v + 0 \cdot v \\
0 &= 0 \cdot v && \text{by the additive inverse axiom}
\end{align*}
$$

- Dir 2: 

Let's suppose that $0 \cdot v = 0 ~~ \forall v \in V$, we need to prove the existence of an additive inverse. 

My approach relies on the following assumptions of a field 

- existence of an multiplicative identity $1 \in F, 1 \cdot v = v, \forall v \in V$

- existence of an additive inverse for every scalar $(a +(-a) = 0), \forall a \in F$ 


$$
\begin{align*}
0 \cdot v &= 0 && \text{given} \\
\implies (1 + (-1)) \cdot v & = 0 && \text{using the assumptions above} \\
\implies 1 \cdot v + (-1)\cdot v & = 0 \\
\implies v + (-1)\cdot v & = 0 && \text{using the multiplicative identity} 
\end{align*}
$$

which means that for any vector $v \in F$, there exists a vector $(-1) \cdot v \in F$ such $v +  (-1) \cdot v = 0$

