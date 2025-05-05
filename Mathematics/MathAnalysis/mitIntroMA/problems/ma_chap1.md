# Overview

This is my attempt to solve problems from the book "Introduction to Analysis" by Arthur Mattuck. Specifically, the first chapter.



## Problem 1.2

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

if we choose $n \in \mathbb{N}$ such that $2m \pi + \frac{\pi}{2} \leq n < 2m \pi + \pi$

then $n + 1 \in [2m \pi, 2m \pi + \frac{\pi}{2}]$

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

Show increasing; find an upper bound, if it exists; give the limit if you can.


to prove this sequence is increasing, I'll show that $\frac{a_{n+1}}{a_n} > 1$ for all $n \geq 1$.

First, I'll compute the ratio:

$$
\frac{a_{n+1}}{a_n} = \frac{\frac{\sqrt{(n+1)^2 - 1}}{n+1}}{\frac{\sqrt{n^2 - 1}}{n}} = \frac{n\sqrt{(n+1)^2 - 1}}{(n+1)\sqrt{n^2 - 1}}
$$

To simplify the analysis, I'll examine $\frac{a_{n+1}^2}{a_n^2}$. If this is greater than 1, then $\frac{a_{n+1}}{a_n} > 1$ as well.

$$
\frac{a_{n+1}^2}{a_n^2} = \frac{n^2(n^2 + 2n)}{(n+1)^2(n^2 - 1)} = \frac{n^4 + 2n^3}{(n^2 + 2n + 1)(n^2 - 1)}
$$

Expanding the denominator:

$$
(n^2 + 2n + 1)(n^2 - 1) = n^4 + 2n^3 + n^2 - n^2 - 2n - 1 = n^4 + 2n^3 - 2n - 1
$$

Therefore:

$$
\frac{a_{n+1}^2}{a_n^2} = \frac{n^4 + 2n^3}{n^4 + 2n^3 - 2n - 1}
$$

Since $-2n - 1$ is negative for all $n \geq 1$, the denominator is less than the numerator, making the fraction greater than 1. Thus, $\frac{a_{n+1}}{a_n} > 1$ and the sequence is strictly increasing.

The sequence is clearly bounded above by 1.

-----------------------------------------   

$$ a_n = \left(2 - \frac{1}{n}\right)\left(2 + \frac{1}{n}\right) $$

Simplifying:

$$
a_n = 4 - \frac{1}{n^2}
$$

The sequence is clearly increasing as $n$ increases, since $-\frac{1}{n^2}$ becomes less negative. The upper bound is 4, and the limit as $n \to \infty$ is 4.

-----------------------------------------   

$$ a_n = \sum_{k=0}^{n} \sin^2(k\pi) $$

Since $\sin(k\pi) = 0$ for all integers $k$, each term in the sum is zero. Therefore, the sequence is constant, not increasing, and the sum is always 0.

-----------------------------------------   

$$ a_n = \sum_{k=0}^{n} \sin^2\left(\frac{k\pi}{2}\right) $$

The sequence of $\sin^2\left(\frac{k\pi}{2}\right)$ is periodic with period 4: $0, 1, 0, 1$.

The sum $a_n$ increases by 1 every two terms. Therefore, the sequence is increasing without an upper bound. The limit does not exist as $n \to \infty$.

-----------------------------------------   














