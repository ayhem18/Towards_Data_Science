Most of my notes are non-linear and saved on my Obsidian vault. 

# Chapter 1

- some properties of sequences: increasing, decreasing, bounded ... ?


- ***bounded and monotonic sequences are convergent***

 
- sequence manipulations tricks and ideas


## Chapter 3

- if a sequence has a limit, the limit is unique.

**Proof:**  

Suppose, for contradiction, that a sequence $a_n$ has two limits $L_1 < L_2$.  

By the definition of limit, 

for any $\epsilon > 0$, there exists $N_1$ such that for all 

$$n > N_1, \quad |a_n - L_1| < \epsilon$$

and there exists $N_2$ such that for all 

$$n > N_2, \quad |a_n - L_2| < \epsilon$$



Let $\epsilon < \frac{L_2 - L_1}{2}$ and let $N = \max(N_1, N_2)$. Then for all $n > N$:

$$  
|a_n - L_1| < \epsilon \implies a_n < L_1 + \epsilon < L_1 + \frac{L_2 - L_1}{2} = \frac{L_1 + L_2}{2}
$$

$$
|a_n - L_2| < \epsilon \implies a_n > L_2 - \epsilon > L_2 - \frac{L_2 - L_1}{2} = \frac{L_1 + L_2}{2}
$$

This is a contradiction, since $a_n$ cannot be both less than and greater than $\frac{L_1 + L_2}{2}$ for all large $n$.  
Therefore, the limit is unique.

---

- if a sequence has a limit, then the sequence is bounded.

**Proof:**  

Suppose $a_n \to L$ as $n \to \infty$.  

By the definition of limit 

$$
\forall \epsilon > 0 \\ 
\exists N, \forall n > N,  |a_n - L| < \epsilon 
$$ 

This implies $ |a_n| < |L| + \epsilon $ for all $n > N$. 

let $\epsilon = 1$

$$
\exists N, \forall n > N,  |a_n| < |L| + 1 
$$



For the finitely many terms $$a_1, a_2, \ldots, a_N$$

let $M_1 = \max\{ |a_1|, |a_2|, \ldots, |a_N| \}$

let $M = \max\{ M_1, |L| + 1 \}$

Then for all $n$, 

$$ |a_n| \leq M $$

Thus, the sequence $a_n$ is bounded.


## Chapter 5

### Limit Theorems

#### 1. Linearity of Limits

If $$ \lim_{n \to \infty} a_n = L_1 \\ \lim_{n \to \infty} b_n = L_2 $$

then

$$
\lim_{n \to \infty} (a_n + b_n) = L_1 + L_2
$$

**Proof:**  

Given $ \epsilon > 0 $, there exist $ N_1 $ and $ N_2 $ 

such that for all $$ n > N_1, |a_n - L_1| < \epsilon $$, and for all $$ n > N_2, |b_n - L_2| < \epsilon $$

Let $ N = \max(N_1, N_2) $ 

Then for all $$ n > N $:

$$
|a_n + b_n - (L_1 + L_2)| = |(a_n - L_1) + (b_n - L_2)| \leq |a_n - L_1| + |b_n - L_2| < 2\epsilon
$$

Since $ \epsilon $ is arbitrary, the result follows.

---

#### 2. Product of Limits

If $$ \lim_{n \to \infty} a_n = L_1 \\ \lim_{n \to \infty} b_n = L_2 $$

then

$$
\lim_{n \to \infty} (a_n b_n) = L_1 L_2
$$

**Proof:**  

We have:

$$
\begin{align*}
|a_n b_n - L_1 L_2| &= |a_n b_n - a_n L_2 + a_n L_2 - L_1 L_2| \\
&= |a_n (b_n - L_2) + L_2 (a_n - L_1)| \\
&\leq |a_n| \cdot |b_n - L_2| + |L_2| \cdot |a_n - L_1|
\end{align*}
$$

Since $ a_n \to L_1 $, 

for a given $ \epsilon > 0 $, there exists $ N_1 $ such that for all $ n > N_1 $, 
$$ 
|a_n - L_1| < \epsilon \\
\implies
|a_n| < |L_1| + \epsilon 
$$

and since $ b_n \to L_2 $,

for a given $ \epsilon > 0 $, there exists $ N_2 $ such that for all $ n > N_2 $, $$ |b_n - L_2| < \epsilon \\
\implies
|b_n| < |L_2| + \epsilon $$


Setting $\epsilon < 1$, we have:

$$ 
|a_n b_n - L_1 L_2| < (|L_1| + 1)\epsilon + |L_2| \epsilon = (|L_1| + |L_2| + 1)\epsilon
$$

Since $ \epsilon $ is arbitrary, the result follows.

The result above holds for any $\epsilon < 1$ however, if we choosen $\epsilon \geq 1$, then $\epsilon > 0.8$ and the result holds for $0.8$ which means it holds for all $\epsilon > 1$ (and all $\epsilon$ in general)


---

#### 3. Limit of the Reciprocal

If $$ \lim_{n \to \infty} a_n = L $$ with $$ L \neq 0 $$

then

$$
\lim_{n \to \infty} \frac{1}{a_n} = \frac{1}{L}
$$

**Proof:**  

Let $ \epsilon > 0 $. Choose $ \epsilon_1 = \min\left(\frac{|L|}{2}, \frac{\epsilon |L|^2}{2}\right) $.  

Since $ a_n \to L $, there exists $ N $ such that for all $ n > N $, $ |a_n - L| < \epsilon_1 $. 

It is important to note since $a_n \to L \neq 0$, then $a_n \neq 0$ for all $n > N$, which means we can divide by $a_n$ in the next step.


Then for $ n > N $:

$$
|a_n| \geq |L| - |a_n - L| > |L| - \frac{|L|}{2} = \frac{|L|}{2}
$$

Now,

$$
\left| \frac{1}{a_n} - \frac{1}{L} \right| = \left| \frac{L - a_n}{a_n L} \right| = \frac{|a_n - L|}{|a_n||L|}
$$

$$
< \frac{\epsilon_1}{\frac{|L|}{2} |L|} = \frac{2\epsilon_1}{|L|^2}
$$

By our choice of $ \epsilon_1 $, $ \frac{2\epsilon_1}{|L|^2} < \epsilon $.  
Therefore, $ \left| \frac{1}{a_n} - \frac{1}{L} \right| < \epsilon $ for all sufficiently large $ n $, proving the result.









