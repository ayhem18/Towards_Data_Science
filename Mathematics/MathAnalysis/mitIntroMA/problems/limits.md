# Overview

This is my attempt to solve problems from the book "Introduction to Analysis" by Arthur Mattuck. Specifically, the first chapter.

# Chapter 1

## Exercice 1.2

For each of the \( a_n \) below, determine if the sequence \(\{a_n\}, n \geq 1\), is increasing (strictly?), decreasing (strictly?), or neither. Show reasoning.


$$
a_n = 1 - \frac{1}{2} + \frac{1}{3} - \cdots + (-1)^{n-1} \frac{1}{n}
$$

neither decreasing nor increasing. why ?

$$
a_{n + 1} = a_n + (-1)^{n} \frac{1}{n+1} 
$$ 

$$

\text{if n is even, } a_{n+1} = a_n + \frac{1}{n+1}  \implies a_{n+1} > a_n \\
\text{if n is odd, } a_{n+1} = a_n - \frac{1}{n+1} \implies a_{n+1} < a_n
$$

----------------------------------------- 

$$
a_n = \frac{n}{n+1} = 1 - \frac{1}{n+1} \\
\frac{1}{n + 1} < \frac{1}{n} \implies 1 - \frac{1}{n + 1} > 1 - \frac{1}{n} \implies a_{n+1} > a_n
$$

----------------------------------------- 

$$
a_n = \sum_{k=1}^{n} \sin^2 k \\
\sin^2 k \geq 0 \implies a_{n+1} = a_n + \sin^2 (n+1) > a_n
$$

----------------------------------------- 


$$
a_n = \sum_{k=1}^{n} \sin k \\
$$

if $k \in [2n \pi, 2n \pi + \pi]$, $\sin k \geq 0$ 

if we choose $n \in \mathbb{N}$ such that $2m \pi \leq n < 2m \pi + \frac{\pi}{2}$ which exists, since $\frac{\pi}{2} > 1$, 

then $n + 1 \in [2m \pi + 1, 2m \pi + \pi]$

then $\sin (n+1) \geq 0$

so $a_{n+1} = a_n + \sin (n+1) < a_n$

if we choose $n \in \mathbb{N}$ such that $2m \pi + \pi - 1 < n < 2m \pi + \pi$

then $n + 1 \in ]2m \pi + \pi, 2m \pi + 2\pi[$

then $\sin (n+1) \leq 0$

so $a_{n+1} = a_n + \sin (n+1) > a_n$


Hence the sequence is neither strictly increasing nor strictly decreasing.

----------------------------------------- 

$$
a_n = \sin\left(\frac{1}{n}\right) \\
$$

$\forall n \in \mathbb{N}, 1 \geq \frac{1}{n} > 0$, and $\sin x$ is increasing for $x > 0$, so $\sin\left(\frac{1}{n}\right)$ is strictly decreasing.

----------------------------------------- 

$$
a_n = \sqrt{1 + \frac{1}{n^2}} \\
$$


Since $\frac{1}{(n+1)^2} < \frac{1}{n^2}$, the sequence is strictly decreasing.


## Problem 1.3

Prove that $$ a_n = \dfrac{1 \cdot 3 \cdot \ldots \cdot (2n+1)}{2 \cdot 4 \cdot \ldots \cdot (2n)} $$    is strictly increasing and not bounded above.

---

Let us first write $ a_n $ more explicitly:
$$
a_n = \frac{1 \cdot 3 \cdot 5 \cdots (2n+1)}{2 \cdot 4 \cdot 6 \cdots (2n)}
$$


2. Prove that \( a_n \) is not bounded above

Let us estimate $ a_n $ from below. Notice that
$$
a_n = \prod_{k=1}^n \frac{2k+1}{2k}
$$      


we can show divergence more formally. Take logarithms:

$$
\log a_n = \sum_{k=1}^n \log\left(1 + \frac{1}{2k}\right)
$$

for numbers less than 1 and greater than 0, $ \log(1+x) > \frac{x}{2} $

$$
\log a_n = \log(\prod_{k=1}^n \frac{2k+1}{2k}) = \sum_{k=1}^n \log\left(1 + \frac{1}{2k}\right) > \sum_{k=1}^n \frac{1}{4k} = \frac{1}{4} \sum_{k=1}^n \frac{1}{k}
$$
The harmonic series diverges, so $ \log a_n \to \infty $ as $ n \to \infty $, and thus $ a_n \to \infty $.


## Chapter 2 

## Chapter 3

## Chapter 5















