# Overview

This is  my attempt to solve the problems in the book "Introduction to Probability" by Dimitri P. Bertsekas and John N. Tsitsiklis, Massachusetts Institute of Technology. 

This file considers only the first chapter of the book.

# Problem 2

Let A and B be two sets.

(a) Show that  $A^c = (A^c \cap B) \cup (A^c \cap B^c)$ 

Solution:
We start by noting that $A^c$ can be written as the intersection of $A^c$ with the universal set $\Omega$:
$$A^c = A^c \cap \Omega$$

Since $B \cup B^c = \Omega$ (any set and its complement comprise the entire universal set), we can substitute:
$$A^c = A^c \cap (B \cup B^c)$$

Now, applying the distributive property of intersection over union:
$$A^c = (A^c \cap B) \cup (A^c \cap B^c)$$

This completes the proof.

(b) show that $(A \cap B) ^ c = (A^c \cap B) \cup (A^c \cap B^c)   \cup (A \cap B^c)$

Solution:

- De Morgan's Law:
$$(A \cap B)^c = A^c \cup B^c$$

- Replace $A^c$ using the result from part (a):
$$A^c = (A^c \cap B) \cup (A^c \cap B^c)$$

- Similarly, for $B^c$ (using the same logic as in part (a), but swapping A and B):
$$B^c = (B^c \cap A) \cup (B^c \cap A^c) = (A \cap B^c) \cup (A^c \cap B^c)$$

- substitute:

$$(A \cap B)^c = A^c \cup B^c = [(A^c \cap B) \cup (A^c \cap B^c)] \cup [(A \cap B^c) \cup (A^c \cap B^c)]$$

Distributing the union operation:
$$(A \cap B)^c = (A^c \cap B) \cup (A^c \cap B^c) \cup (A \cap B^c) \cup (A^c \cap B^c)$$

- $X \cup X = X$ for any set $X$:

$$(A \cap B)^c = (A^c \cap B) \cup (A^c \cap B^c) \cup (A \cap B^c)$$

This completes the proof.

# Problem 5

Out of the students in a class, 60% are geniuses, 70% love chocolate,
and 40% fall into both categories. Determine the probability that a randomly selected student is neither a genius nor a chocolate lover.
    
Solution: 

Let $G$ be the set of geniuses and $C$ be the set of chocolate lovers.

We are given:
- $P(G) = 0.6$
- $P(C) = 0.7$
- $P(G \cap C) = 0.4$

We want to find the probability that a student is neither a genius nor a chocolate lover, which is denoted by $P(G^c \cap C^c)$.

By De Morgan's Law, we know that:
$G^c \cap C^c = (G \cup C)^c$

So we need to find $P((G \cup C)^c) = 1 - P(G \cup C)$

Using the inclusion-exclusion principle for the union of two sets:

$$P(G \cup C) = P(G) + P(C) - P(G \cap C)$$

$$P(G \cup C) = 0.6 + 0.7 - 0.4 = 0.9$$

Therefore:

$$P(G^c \cap C^c) = 1 - P(G \cup C) = 1 - 0.9 = 0.1$$

Thus, the probability that a randomly selected student is neither a genius nor a chocolate lover is 0.1 or 10%.

# Problem 6

A six-sided die is loaded in a way that each even face is twice as likely
as each odd face. All even faces are equally likely, as are all odd faces. Construct a probabilistic model for a single roll of this die and find the probability that the outcome is less than 4.

Let $p_i$ be the probability of the outcome $i$. The outcomes are 1, 2, 3, 4, 5, 6 and they are disjoint. 

$p_i = 2 \cdot p_j$ if $i$ is even and $j$ is odd. 
$p_i = p_j$ if $i$ and $j$ are of the same parity. 

Using the fact that the sum of the probabilities of all outcomes is 1, we have:

$$p_1 + p_2 + p_3 + p_4 + p_5 + p_6 = 1$$   

which means:

$p_i = \frac{1}{9}$ for $i$ odd and $p_i = \frac{2}{9}$ for $i$ even. 

The probability that the outcome is less than 4 is:

$$P(X < 4) = P(X = 1) + P(X = 2) + P(X = 3)$$

$$P(X < 4) = \frac{1}{9} + \frac{2}{9} + \frac{1}{9} = \frac{4}{9}$$   


# Problem 8

You enter a special kind of chess tournament, in which you play one game
with each of three opponents, but you get to choose the order in which you play your opponents, knowing the probability of a win against each. You win the tournament if you win two games in a row, and you want to maximize the probability of winning. Show that it is optimal to play the weakest opponent second, and that the order of playing the other two opponents does not matter.

## Solution

Let's denote the probability of winning against the three opponents as $p_1$, $p_2$, and $p_3$ based on their order of play (not their strength).

To win the tournament, you need to win at least two consecutive games. This can happen in three ways:

1. Win games 1 and 2 (regardless of game 3)
2. Win games 2 and 3, after losing game 1

The total probability of winning the tournament is therefore:

$$P(\text{win tournament}) = p_1 \cdot p_2 + (1-p_1) \cdot p_2 \cdot p_3$$

We can factorize this expression:

$
\begin{align}
P(\text{win tournament}) &= p_1 \cdot p_2 + p_2 \cdot p_3 - p_1 \cdot p_2 
\cdot p_3 \\
&= p_2 \cdot (p_1 + p_3 - p_1 \cdot p_3) 
\end{align}
$ 

From this factorized form, we can make two important observations:

1. The probability of winning is directly proportional to $p_2$, which means to maximize our chance of winning, we should assign our highest winning probability (i.e., play the weakest opponent) in the second position.

2. $p_1$ and $p_3$ appear in a symmetric form in the expression $p_1 + p_3 - p_1p_3$. In fact, if we were to interchange $p_1$ and $p_3$, the expression remains unchanged. This symmetry proves that the order of the first and third opponents doesn't matter.

Therefore, it is optimal to play the weakest opponent second, and the order of playing the other two opponents does not affect the probability of winning the tournament.



# Problem 9

A partition of the sample space Ω is a collection of disjoint events
$S_1, ..., S_n$ such that $\Omega = \cup_{i=1}^n S_i$.

(a) Show that for any event A, we have

$$P(A) = \sum_{i=1}^n P(A \cap S_i)$$

Solution for (a):

Since $S_1, ..., S_n$ form a partition of the sample space, we can express $A$ as:
$$A = A \cap \Omega = A \cap (\cup_{i=1}^n S_i) = \cup_{i=1}^n (A \cap S_i)$$

The sets $A \cap S_i$ are disjoint because the $S_i$ are disjoint. Therefore, by the additivity of probability:

$$P(A) = P(\cup_{i=1}^n (A \cap S_i)) = \sum_{i=1}^n P(A \cap S_i)$$

This completes the proof for part (a).

(b) Show that

$$P(A) = P(A \cap B) + P(A \cap C) + P(A \cap B^c \cap C^c) - P(A \cap B \cap C)$$

Solution for (b):

- consider the following partition of the sample space $\Omega$:

$\Omega$:
- $B \cap C$
- $B \cap C^c$
- $B^c \cap C$
- $B^c \cap C^c$

They are clearly disjoint since any intersection of two of them will contain either $B$ and $B^C$ or $C$ and $C^C$ and the intersection of both is an empty set.


Now let's verify their union equals $\Omega$:

$$
\begin{align*}
(B \cap C) \cup (B \cap C^c) \cup (B^c \cap C) \cup (B^c \cap C^c) &= (B \cap (C \cup C^c)) \cup (B^c \cap (C \cup C^c)) \\
&= (B \cap \Omega) \cup (B^c \cap \Omega) \\
&= B \cup B^c \\
&= \Omega
\end{align*}
$$

Now, applying the result from part (a) with this partition:

$$
P(A) = P(A \cap (B \cap C)) + P(A \cap (B \cap C^c)) + P(A \cap (B^c \cap C)) + P(A \cap (B^c \cap C^c))
$$

We can regroup these terms:

$$
\begin{align*}
P(A \cap B) &= P(A \cap B \cap C) + P(A \cap B \cap C^c) \\
P(A \cap C) &= P(A \cap B \cap C) + P(A \cap B^c \cap C)
\end{align*}
$$

This gives us:

$$
\begin{align*}
P(A) &= P(A \cap (B \cap C)) + P(A \cap (B \cap C^c)) + P(A \cap (B^c \cap C)) + P(A \cap (B^c \cap C^c)) \\

&= [P(A \cap B \cap C) + P(A \cap B \cap C^c)] + P(A \cap B^c \cap C) + P(A \cap B \cap C) - P(A \cap B \cap C) + P(A \cap B^c \cap C^c) \\ 

&= P(A \cap B) + P(A \cap C) + P(A \cap B^c \cap C^c) - P(A \cap B \cap C)
\end{align*}
$$

# Problem 10

Show that

$$P((A \cap B^c) \cup (A^c \cap B)) = P(A) + P(B) - 2P(A \cap B)$$  


$$
\begin{align*}
(A \cap B^c) \cap (A^c \cap B) &= A \cap B^c \cap A^c \cap B \\
&= A \cap A^c \cap B \cap B^c \\
&= \emptyset
\end{align*}
$$


Since the sets are disjoint, we can apply the additivity property:

$$P((A \cap B^c) \cup (A^c \cap B)) = P(A \cap B^c) + P(A^c \cap B)$$

$$P(A \cap B^c) = P(A) - P(A \cap B)$$

$$P(A^c \cap B) = P(B) - P(A \cap B)$$

Therefore:

Substituting these expressions back into our equation from Step 2:

$$
\begin{align*}
P((A \cap B^c) \cup (A^c \cap B)) &= P(A \cap B^c) + P(A^c \cap B) \\
&= [P(A) - P(A \cap B)] + [P(B) - P(A \cap B)] \\
&= P(A) + P(B) - 2P(A \cap B)
\end{align*}
$$

This completes the proof.




 







