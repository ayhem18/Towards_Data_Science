# Overview

These are my solutions of several problems in the chapter 2 to 3 of the book "A Modern Introduction to Probability".

# Chapter 2

* Few important points:     
    * Probability is associated with an experiment. The sample space is simply a set of results / outcomes of the experiments. The outcomes are mutually exclusive. 
    * An event can be expressed as a subset of the sample space. 
    
    $$ P(A) = \frac{|A|}{|\Omega|} $$

    * Perceiving Probability as a cardinality of a set enables us to use principles from naive set theory: 
    $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$



## Exercice 2.1

Using the formula     $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
 
, it is possible to compute 

$$P(A \cup B) = \frac{2}{3} + \frac{1}{6} - \frac{1}{9} = \frac{13}{18}$$ 


## Exercice 2.2

1. use Morgan's law to compute the complement of $E \cup F$

$$(E^c \cap F^c) = (E \cup F)^c$$

2. use the complement formula to compute the probability of the complement of $E \cup F$

$$P((E^c \cap F^c)) = P((E \cup F)^c) = 1 - P(E \cup F) = 1 - \frac{3}{4} = \frac{1}{4}$$


## Exercice 2.4
Using a ven diagram, the answer is clearly yes.


## Exercice 2.8
The question is about estimating the probability of the union of two events.

P(A ∩ B) is always less than P(A) or P(B), since A ∩ B is contained in both A and B. Mathematically:
$$P(A \cap B) \leq P(A) \text{ and } P(A \cap B) \leq P(B)$$

Using the formula:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

We can derive the following estimations:

$$P(A) + P(B) \geq P(A \cup B) \geq \max(P(A), P(B))$$
$$\min(P(A), P(B)) \geq P(A \cap B) \geq 0$$

## Exerice 2.11

Since these are the only two outcomes in the sample space, their probabilities must sum to 1:

$$p + p^2 = 1$$

This gives us a quadratic equation:
$$p^2 + p - 1 = 0$$

Using the quadratic formula:
$$p = \frac{-1 \pm \sqrt{1 + 4}}{2} = \frac{-1 \pm \sqrt{5}}{2}$$

This gives us two possible values:
$p = \frac{-1 + \sqrt{5}}{2} \approx 0.618$
$p = \frac{-1 - \sqrt{5}}{2} \approx -1.618$

Since p represents a probability, it must be between 0 and 1. Therefore, the only valid solution is:
$$p = \frac{-1 + \sqrt{5}}{2} \approx 0.618$$


## Exercice 2.16

Three events E, F, and G cannot occur simultaneously. Further it is known that P(E ∩ F) = P(F ∩ G) = P(E ∩ G) = 1/3. We want to determine P(E).

Since E, F, and G cannot occur simultaneously, we have P(E ∩ F ∩ G) = 0. This means that the pairwise intersections (E ∩ F), (F ∩ G), and (E ∩ G) are disjoint from each other:

- $(E \cap F) \cap (F \cap G) = E \cap F \cap G = \emptyset$
- $(E \cap F) \cap (E \cap G) = E \cap F \cap G = \emptyset$
- $(E \cap G) \cap (F \cap G) = E \cap F \cap G = \emptyset$

Additionally, the sum of probabilities of these pairwise intersections equals 1:

$$P(E \cap F) + P(F \cap G) + P(E \cap G) = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1$$

This means that the entire sample space Ω is covered by these three pairwise intersections:

$$\Omega = (E \cap F) \cup (F \cap G) \cup (E \cap G)$$

Now, to find P(E):
$$P(E) = P(E \cap \Omega) = P(E \cap ((E \cap F) \cup (F \cap G) \cup (E \cap G)))$$

Applying the distributive property:

$$ 
\begin{align*}
P(E) &= P(E \cap ((E \cap F) \cup (F \cap G) \cup (E \cap G))) \\
     &= P((E \cap (E \cap F)) \cup (E \cap (F \cap G)) \cup (E \cap (E \cap G)))
\end{align*}
$$ 

Since P(E ∩ F ∩ G) = 0:
$$P(E) = P((E \cap F) \cup (E \cap G))$$

Since (E ∩ F) and (E ∩ G) are disjoint:
$$P(E) = P(E \cap F) + P(E \cap G) = \frac{1}{3} + \frac{1}{3} = \frac{2}{3}$$

Therefore, P(E) = 2/3. 

The answer can be verified in the book.

# Chapter 3

## Exercice 3.2

A fair die is thrown twice. A is the event "sum of the throws equals 4," B is "at least one of the throws is a 3."

a. Calculate P(A | B).

b. Are A and B independent events? 

P(A | B) = P(A ∩ B) / P(B) 

$$ P(B) = 1 - P(B ^ c) = 1 - (\frac{5}{6} \times \frac{5}{6}) = \frac{11}{36} $$

$$ P(A \cap B) = \frac{2}{36} = \frac{1}{18} $$

$$ P(A | B) = \frac{P(A \cap B)}{P(B)} = \frac{\frac{1}{18}}{\frac{11}{36}} = \frac{2}{11} $$

$$P(A | B) = \frac{2}{11}$$

b. Are A and B independent events? 


$$ P(A) = \frac{3}{36} = \frac{1}{12} $$ 

$$ P(A) \neq P(A | B) $$

Therefore, A and B are not independent events.

## Exercice 3.3

A Dutch cow is tested for BSE, using Test A as described in Section 3.3,
with $P(T | B)=0.70$ and $P(T | B^c)=0.10$. 

Assume that the BSE risk for the Netherlands is the same as in 2003, when it was estimated to be $P(B) = 1.3 \cdot 10^{-5}$. 

Compute $P(B | T)$ and $P(B | T^c)$ 

Using Bayes' theorem and the law of total probability, we can compute the following:

$$P(B | T) = \frac{P(T | B)P(B)}{P(T)}$$

$$P(T) = P(B \cap T) + P(B^c \cap T)$$

all the values to compute $P(B | T)$ are given in the question.

# Exercice 3.7

Calculate:

a. P(A ∪ B) if it is given that P(A)=1/3 and P(B | A^c)=1/4.

Using conditional probability and set theory principles:

First, we can express the intersection of B and A^c using conditional probability:
$$P(B \cap A^c) = P(B | A^c) \cdot P(A^c)$$

Since $P(A^c) = 1 - P(A)$, we have:
$$P(B \cap A^c) = P(B | A^c) \cdot (1 - P(A))$$

Now, we can decompose $P(A \cup B)$ into disjoint events:
$$P(A \cup B) = P(A) + P(B \cap A^c)$$

Substituting our expression:
$$P(A \cup B) = P(A) + P(B | A^c) \cdot (1 - P(A))$$

Using the given values $P(A) = \frac{1}{3}$ and $P(B | A^c) = \frac{1}{4}$:

Therefore, $P(A \cup B) = \frac{1}{2}$.

b. $P(B)$ if it is given that $P(A \cup B) = \frac{2}{3}$ and $P(A^c | B^c) = \frac{1}{2}$

First, we note that:
$$P(A^c \cap B^c) = P((A \cup B)^c) = 1 - P(A \cup B)$$

From the conditional probability formula:
$$P(A^c | B^c) = \frac{P(A^c \cap B^c)}{P(B^c)}$$

Rearranging to solve for $P(B^c)$:
$$P(B^c) = \frac{P(A^c \cap B^c)}{P(A^c | B^c)} = \frac{1 - P(A \cup B)}{P(A^c | B^c)}$$

Finally, we can find $P(B)$ using the complement rule:
$$P(B) = 1 - P(B^c) = 1 - \frac{1 - P(A \cup B)}{P(A^c | B^c)}$$

Substituting the given values $P(A \cup B) = \frac{2}{3}$ and $P(A^c | B^c) = \frac{1}{2}$:
$$P(B) = 1 - \frac{1 - \frac{2}{3}}{\frac{1}{2}} = 1 - \frac{\frac{1}{3}}{\frac{1}{2}} = 1 - \frac{2}{3} = \frac{1}{3}$$

Therefore, $P(B) = \frac{1}{3}$.


## Exerice 3.9 

A certain grapefruit variety is grown in two regions in southern Spain.
Both areas get infested from time to time with parasites that damage the
crop. Let A be the event that region R1 is infested with parasites and B that
region R2 is infested. Suppose $P(A) = \frac{3}{4}$, $P(B) = \frac{2}{5}$ and $P(A \cup B) = \frac{4}{5}$.
If the food inspection detects the parasite in a ship carrying grapefruits from
R1, what is the probability region R2 is infested as well? 

We are asked to find $P(B | A)$, the probability that region R2 is infested given that region R1 is infested.

Using the conditional probability formula:
$$P(B | A) = \frac{P(A \cap B)}{P(A)}$$

To find $P(A \cap B)$, we can use the set theory formula for the union of two events:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

Rearranging to isolate $P(A \cap B)$:
$$P(A \cap B) = P(A) + P(B) - P(A \cup B)$$

Substituting this into our conditional probability formula:
$$P(B | A) = \frac{P(A) + P(B) - P(A \cup B)}{P(A)}$$

$$P(B | A) = \frac{7}{15}$$

Therefore, the probability that region R2 is infested given that region R1 is infested is $\frac{7}{15}$.


# Exercice 3.12

The events A, B, and C satisfy: $P(A | B \cap C) = \frac{1}{4}$, $P(B | C) = \frac{1}{3}$,
and $P(C) = \frac{1}{2}$. Calculate $P(A^c \cap B \cap C)$ 

To solve this problem, I'll follow these steps:

1. First, note that $P(A^c \cap B \cap C) = P(B \cap C) - P(A \cap B \cap C)$

2. Calculate $P(A \cap B \cap C)$ using the chain rule of conditional probabilities:
   $$P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B \cap C)$$

3. Calculate $P(B \cap C)$ using conditional probability:
   $$P(B \cap C) = P(B | C) \cdot P(C)$$

Substituting step 3 into step 2:
$$P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B | C) \cdot P(C)$$

Now I can calculate both parts using the given values:

$$P(B \cap C) = P(B | C) \cdot P(C) = \frac{1}{3} \cdot \frac{1}{2} = \frac{1}{6}$$

$$P(A \cap B \cap C) = P(A | B \cap C) \cdot P(B | C) \cdot P(C) = \frac{1}{4} \cdot \frac{1}{3} \cdot \frac{1}{2} = \frac{1}{24}$$

Finally, using step 1:
$$P(A^c \cap B \cap C) = P(B \cap C) - P(A \cap B \cap C) = \frac{1}{6} - \frac{1}{24} = \frac{4}{24} - \frac{1}{24} = \frac{3}{24} = \frac{1}{8}$$

Therefore, $P(A^c \cap B \cap C) = \frac{1}{8}$.




