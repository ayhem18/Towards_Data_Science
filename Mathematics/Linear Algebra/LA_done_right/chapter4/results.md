# Polynomial Results

---

## Result 1 — Derivative of a Product of Linear Factors
Let
$$
Q(x)=\prod_{i=1}^{n}(x-\lambda_i),\qquad \lambda_i\in\mathbb F.
$$
Then
$$
Q'(x)=\sum_{i=1}^{n}\;\prod_{\substack{j=1\\ j\ne i}}^{n}(x-\lambda_j).
$$

### Proof

Base case \(n=2\) is immediate from the product rule.  
Assume the statement holds for some \(n\ge2\); prove it for \(n+1\).

$$
\begin{align*}
Q(x)&=\prod_{i=1}^{n+1}(x-\lambda_i) \;{\text{set }} Y(x)=\prod_{i=1}^{n}(x-\lambda_i)\\[4pt]
Q'(x)&=Y'(x)\,(x-\lambda_{n+1})+Y(x) \;{\text{(product rule)}}\\[6pt]
\text{Induction }&\text{gives }Y'(x)=\sum_{i=1}^{n}\prod_{\substack{j=1\\ j\ne i}}^{n}(x-\lambda_j).\\[4pt]
\therefore\;Q'(x)&=\sum_{i=1}^{n}\Bigl[\prod_{\substack{j=1\\ j\ne i}}^{n}(x-\lambda_j)\Bigr](x-\lambda_{n+1})+\prod_{j=1}^{n}(x-\lambda_j)\\[6pt]
&=\sum_{i=1}^{n}\prod_{\substack{j=1\\ j\ne i}}^{n+1}(x-\lambda_j)+\prod_{\substack{j=1\\ j\ne n+1}}^{n+1}(x-\lambda_j)\\[4pt]
&=\sum_{i=1}^{n+1}\prod_{\substack{j=1\\ j\ne i}}^{n+1}(x-\lambda_j)
\end{align*}
$$

completing the induction step and the proof.

---

## Result 2 — Real-valued on \(\mathbb R\) \(\Leftrightarrow\) Real Coefficients

$$
\text{Polynomial }p\text{ has real coefficients }\Longleftrightarrow p(x)\in\mathbb R\;\forall x\in\mathbb R.
$$

### Proof
Write
$$
p(x)=\sum_{k=0}^{m}\alpha_k x^{k},\qquad\alpha_k\in\mathbb C.
$$

1. **Real coefficients \(\Rightarrow\) real-valued.**  Trivial: for real $x$, every term $\alpha_k x^k$ is real.

2. **Real-valued \(\Rightarrow\) real coefficients.**  Define

$$
 q(x)=\sum_{k=0}^{m}(\alpha_k-\overline{\alpha_k})x^{k}. \tag{1}
$$

For a real number $z$ we have
$$
 q(z)=\sum_{k=0}^{m}(\alpha_k-\overline{\alpha_k})z^{k}=p(z)-\overline{p(z)}. \tag{2}
$$

By hypothesis $p(z)\in\mathbb R$ when $z\in\mathbb R$, hence $p(z)=\overline{p(z)}$ and (2) gives $q(z)=0$ for **all** real $z$.

A non-zero polynomial of degree $\le m$ has at most $m$ real roots; $q$ has infinitely many, so $q\equiv0$.  From (1) this forces $\alpha_k=\overline{\alpha_k}$ for every $k$, i.e. each coefficient is real.

---


# Result 3 

$$
\text{for a polynomial }p\text{ with real coefficients, }p(\lambda) = 0 \iff p(\overline{\lambda}) = 0
$$

### Proof

Let $p(x)=\sum_{k=0}^{m}\alpha_k x^{k}$ with each $\alpha_k\in\mathbb R$.  For any $\lambda\in\mathbb C$ compute
$$
\begin{align*}
p(\overline{\lambda}) &= \sum_{k=0}^{m}\alpha_k \overline{\lambda}^{\;k} &&\{\alpha_k\text{ real}\}\\[4pt]
\overline{p(\lambda)} &= \overline{\sum_{k=0}^{m}\alpha_k \lambda^{k}}=\sum_{k=0}^{m}\alpha_k\,\overline{\lambda^{k}}=\sum_{k=0}^{m}\alpha_k\overline{\lambda}^{\;k}.
\end{align*}
$$
Thus $p(\overline{\lambda})=\overline{p(\lambda)}$.  If $p(\lambda)=0$ then the right-hand side is $0$, hence $p(\overline{\lambda})=0$ as well.  The converse follows by symmetry.

---

