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
We can verify this answer by checking that $p + p^2 = 1$:
If $p = \frac{-1 + \sqrt{5}}{2}$, then $p^2 = \frac{3 - \sqrt{5}}{2}$
$p + p^2 = \frac{-1 + \sqrt{5}}{2} + \frac{3 - \sqrt{5}}{2} = \frac{2}{2} = 1$ ✓


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
$$P(E) = P((E \cap (E \cap F)) \cup (E \cap (F \cap G)) \cup (E \cap (E \cap G)))$$
     = P((E \cap F) \cup (E \cap F \cap G) \cup (E \cap G))$$

Since P(E ∩ F ∩ G) = 0:
$$P(E) = P((E \cap F) \cup (E \cap G))$$

Since (E ∩ F) and (E ∩ G) are disjoint:
$$P(E) = P(E \cap F) + P(E \cap G) = \frac{1}{3} + \frac{1}{3} = \frac{2}{3}$$

Therefore, P(E) = 2/3. 

The answer can be verified in the book.

# Chapter 3


