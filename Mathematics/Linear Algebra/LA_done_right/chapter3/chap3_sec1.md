![](./images/img1.png)

## Problem 1

Let $b,c\in\mathbb{R}$ and define $T:\mathbb{R}^3\to\mathbb{R}^2$ by
$$
T(x,y,z)\;=\;(2x-4y+3z+b,\;6x+cxyz).
$$
We prove that $T$ is linear **iff** $b=c=0$.

---
### ($\Leftarrow$)  If $b=c=0$ then $T$ is linear
With $b=c=0$ we have $T(x,y,z)=(2x-4y+3z,6x)$, clearly a linear combination of the coordinates, hence linear.

---
### ($\Rightarrow$)  If $T$ is linear then $b=c=0$

1. **Scalar-multiplication test for $b$.**  
   For any $\alpha\in\mathbb{R}$ and $(x,y,z)\in\mathbb{R}^3$, linearity requires
   $T(\alpha x,\alpha y,\alpha z)=\alpha T(x,y,z)$.  
   Taking $(x,y,z)=(0,0,1)$ gives
   $$
   T(0,0,\alpha)=\bigl(3\alpha+b,\,0\bigr),\qquad
   \alpha T(0,0,1)=\alpha(3+b,0).
   $$
   Equating first components yields $3\alpha+b=3\alpha+\alpha b$, so $b=\alpha b$ for every $\alpha$.  Choosing $\alpha\neq1$ forces $b=0$.

2. **Additivity test for $c$.**  
   By linearity
   $$
   T(1,0,0)+T(1,1,1)=T\bigl((1,0,0)+(1,1,1)\bigr)=T(2,1,1).
   $$
   Compute each term:

   $$
   \begin{align*}
   T(1,0,0)&=(2+b,6),\\
   T(1,1,1)&=(2-4+3+b,6+c)= (1+b,6+c),\\
   T(2,1,1)&=(4-4+3+b,12+2c)= (3+b,12+2c).
   \end{align*}
   $$ 

   Summing the first two vectors and comparing with the third shows the second components satisfy $6+(6+c)=12+2c$, which implies $c=0$.

Hence $b=c=0$, completing the proof.


---
## Problem 2

Let $b,c\in\mathbb{R}$ and define $T:\mathcal{P}(\mathbb{R})\to\mathbb{R}^2$ by
$$
Tp=\Bigl(3p(4)+5p'(6)+b\,p(1)p(2),\;\displaystyle\int_{-1}^{2}x^{3}p(x)\,dx+c\,\sin p(0)\Bigr).
$$
We show $T$ is linear **iff** $b=c=0$.

---
### Sufficiency
If $b=c=0$ then each component of $Tp$ is a linear functional of $p$: evaluation at points, derivative evaluation, and a definite integral against a fixed weight. Thus $T$ is linear.

---
### Necessity
Suppose $T$ is linear.

1. **Condition $b=0$.**  
   Linearity in scalar multiplication demands $T(\alpha p)=\alpha Tp$.  Take a polynomial $p$ such that $p(1)p(2)\neq0$ (e.g. $p(x)=1$).  The first component yields
   $$3\alpha p(4)+5\alpha p'(6)+b\,\alpha^2 p(1)p(2)=\alpha\bigl(3p(4)+5p'(6)+b p(1)p(2)\bigr).
   $$
   Comparing coefficients of $\alpha^2$ gives $b p(1)p(2)=0$, hence $b=0$.

2. **Condition $c=0$.**  
   Additivity requires $T(p+q)=Tp+Tq$.  Choose $p,q$ such that $p(0)=q(0)=\pi/2$ so that $\sin p(0)=\sin q(0)=1$ but $(p+q)(0)=\pi$, whence $\sin(p+q)(0)=0$.  The second component gives
   $$\int_{-1}^{2}x^{3}(p+q)\,dx+c\sin(p+q)(0)=\int_{-1}^{2}x^{3}p\,dx+\int_{-1}^{2}x^{3}q\,dx+2c.
   $$
   The integrals cancel, leaving $0=2c$, so $c=0$.

Therefore $b=c=0$.

---
## Problem 3

Let $T\in\mathcal{L}(\mathbb{F}^n,\mathbb{F}^m)$.  Denote by $e^{(n)}_1,\dots,e^{(n)}_n$ the standard basis of $\mathbb{F}^n$ and by $e^{(m)}_1,\dots,e^{(m)}_m$ the standard basis of $\mathbb{F}^m$.

For any $x=(x_1,\dots,x_n)\in\mathbb{F}^n$ we have the decomposition
$$
 x=\sum_{k=1}^{n}x_k e^{(n)}_k.
$$
Linearity gives
$$
T(x)=\sum_{k=1}^{n}x_k\,T(e^{(n)}_k).
$$
Write each image $T(e^{(n)}_k)$ in the $e^{(m)}_j$ basis:
$$
T(e^{(n)}_k)=\sum_{j=1}^{m}A_{j,k}\,e^{(m)}_j,\qquad A_{j,k}\in\mathbb{F}.
$$
Substituting,
$$
T(x)=\sum_{k=1}^{n}x_k\Bigl(\sum_{j=1}^{m}A_{j,k}e^{(m)}_j\Bigr)=\sum_{j=1}^{m}\Bigl(\sum_{k=1}^{n}A_{j,k}x_k\Bigr)e^{(m)}_j.
$$
Thus
$$
T(x_1,\dots,x_n)=\bigl(A_{1,1}x_1+\dots+A_{1,n}x_n,\;\dots,\;A_{m,1}x_1+\dots+A_{m,n}x_n\bigr),
$$
which is precisely the matrix–vector product with the $m\times n$ matrix $A=(A_{j,k})$.

---
## Problem 4

Let $T\in\mathcal{L}(V,W)$ and suppose $v_1,\dots,v_m\in V$ with $Tv_1,\dots,Tv_m$ linearly independent in $W$.  We prove $v_1,\dots,v_m$ are independent in $V$.

Assume a relation
$$
\sum_{i=1}^{m}\alpha_i v_i=0.
$$
Apply $T$ and use linearity together with $T(0)=0$:
$$
0=T(0)=T\Bigl(\sum_{i=1}^{m}\alpha_i v_i\Bigr)=\sum_{i=1}^{m}\alpha_i T(v_i).
$$
Because the family $\{T(v_i)\}$ is linearly independent, all coefficients must vanish, i.e. $\alpha_i=0$ for every $i$.  Therefore the original vectors are linearly independent.


## Problem 7

Show that every linear map on a one-dimensional vector space is multiplication by a scalar.

Let $\dim V = 1$ and $T\in\mathcal{L}(V)$.  
Choose a non-zero vector $a\in V$.  Because $V=\operatorname{span}\{a\}$ we can write every $v\in V$ uniquely as $v=\alpha a$ with $\alpha\in\mathbb{F}$.

---
### Step 1 – $T(a)$ is proportional to $a$
The image $T(a)$ lies in $V$, hence there exists a unique scalar $\lambda\in\mathbb{F}$ such that

$$
T(a)=\lambda a. \tag{1}
$$

### Step 2 – $T(v)=\lambda v$ for every $v\in V$
Let $v=\alpha a$.  Using linearity of $T$ and equation (1),
$$
T(v)=T(\alpha a)=\alpha T(a)=\alpha\lambda a=\lambda(\alpha a)=\lambda v.
$$
Thus $T$ acts as multiplication by the fixed scalar $\lambda$ on **all** vectors of $V$.


Consequently every linear operator on a one-dimensional space is a scalar multiple of the identity.

---

## Problem 11

Let $V$ be finite-dimensional and $T\in\mathcal{L}(V)$.  We prove that
$$
ST = TS\quad\text{for all }S\in\mathcal{L}(V)\;\Longleftrightarrow\;T=\lambda I\text{ for some }\lambda\in\mathbb{F}.
$$

---
### ($\Leftarrow$)  Scalar multiples commute with everything
If $T=\lambda I$ then $ST=\lambda S=TS$ for every $S$, so the forward implication is immediate.

---
### ($\Rightarrow$)  Commutation implies $T$ is a scalar multiple of $I$
Assume $ST=TS$ for every linear operator $S$.  Choose a basis $(v_1,\dots,v_n)$ of $V$ (if $n=1$ the statement follows from Problem 7).

#### Step 1 – Coordinate projections commute with $T$
For each index $i$ define $S_i\in\mathcal{L}(V)$ by
$$
S_i\Bigl(\sum_{k=1}^{n}\alpha_k v_k\Bigr)=\alpha_i v_i. \tag{2}
$$
It is **easy to verify** that $S_i$ is linear: additivity and homogeneity follow component-wise from (2).  By hypothesis $S_iT=TS_i$.

Apply both sides to a basis vector $v_k$:
$$
S_iT(v_k)=TS_i(v_k).
$$
If $k\neq i$ then $S_i(v_k)=0$, hence the right-hand side vanishes and we obtain $S_iT(v_k)=0$.

Write $T(v_k)=\sum_{j=1}^{n}\beta_{j,k}v_j$.  Using definition (2),
$$
S_iT(v_k)=\beta_{i,k}v_i.
$$
Therefore $\beta_{i,k}=0$ whenever $i\neq k$.  When $k=i$ we have
$$
TS_i(v_i)=T(v_i)=S_iT(v_i)=\beta_{i,i}v_i.
$$
Hence
$$
T(v_k)=\alpha_k v_k \quad\text{with }\alpha_k:=\beta_{k,k}. \tag{3}
$$

So $T(v_k)$ is a scalar multiple of $v_k$ for each $k$.

#### Step 2 – All scalar coefficients coincide
Pick indices $i\neq j$.  Define a linear map $P_{ij}\in\mathcal{L}(V)$ that swaps $v_i$ and $v_j$ and leaves the other basis vectors unchanged:
$$
P_{ij}(v_i)=v_j,\;P_{ij}(v_j)=v_i,\;P_{ij}(v_k)=v_k\;(k\notin\{i,j\}).
$$
Again $P_{ij}$ is linear by inspection.  By hypothesis $P_{ij}T=TP_{ij}$.  Apply both sides to $v_i$:
$$
P_{ij}T(v_i)=TP_{ij}(v_i).
$$
Using (3) and the definition of $P_{ij}$,
$$
P_{ij}(\alpha_i v_i)=\alpha_i v_j,\qquad T(v_j)=\alpha_j v_j.
$$
Equality of these vectors forces $\alpha_i v_j = \alpha_j v_j$, and because $v_j\neq0$ we deduce $\alpha_i=\alpha_j$.  Thus all $\alpha_k$ are equal to a common scalar $\lambda$.

#### Step 3 – Conclusion
Using (3) with $\alpha_k=\lambda$ gives $T(v_k)=\lambda v_k$ for every basis vector.  By linearity, $T(v)=\lambda v$ for **all** $v\in V$; hence $T=\lambda I$.

---

## Problem 12

Let $U$ be a proper subspace of $V$ (so $U\neq V$) and let $S\in\mathcal{L}(U,W)$ be non-zero.  Define $T:V\to W$ by
$$
T(v)=\begin{cases}
Sv & v\in U,\\[4pt]
0 & v\notin U.
\end{cases}
$$
We prove $T$ is **not** linear.

---
### Step 1 – Choose witnesses outside $U$
Select a vector $v_1\in V\setminus U$ and a vector $a\in U$ with $Sa\neq0$ (such an $a$ exists because $S\neq0$).  Define
$$
x=a+v_1,\qquad y=a-v_1.
$$
Both $x$ and $y$ lie outside $U$, for otherwise $v_1=(x-a)$ or $-v_1=(y-a)$ would belong to $U$, contradicting the choice of $v_1$.

### Step 2 – Evaluate $T$ on $x,y,x+y$
Because $x,y\notin U$ we have $T(x)=T(y)=0$.  Their sum, however, equals
$$
x+y=2a\in U\quad\Longrightarrow\quad T(x+y)=S(2a)=2Sa\neq0.
$$
### Step 3 – Additivity fails
If $T$ were linear, additivity would give $T(x+y)=T(x)+T(y)=0$, contradicting the computation above. Hence $T$ is not linear.

---

## Problem 13

Let $U$ be a subspace of a finite-dimensional vector space $V$, and let $S\in\mathcal{L}(U,W)$.  We construct an extension $T\in\mathcal{L}(V,W)$ satisfying $Tu=Su$ for all $u\in U$.

---
### Step 1 – Complete a basis
Choose a basis $(u_1,\dots,u_m)$ of $U$.  Since $U\neq V$ in the interesting case, extend this list to a basis of $V$ by adding vectors $(v_1,\dots,v_{n-m})$ so that
$$
(u_1,\dots,u_m,v_1,\dots,v_{n-m})
$$
forms a basis of $V$.

### Step 2 – Define $T$ on the basis
Set
$$
Tu_i = Su_i \quad (1\le i\le m),\qquad Tv_j = 0 \quad (1\le j\le n-m). \tag{4}
$$
### Step 3 – Extend linearly and prove linearity
For an arbitrary vector
$$
 x = \sum_{i=1}^{m}\alpha_i u_i + \sum_{j=1}^{n-m}\beta_j v_j
$$
**define**
$$
T(x)= \sum_{i=1}^{m}\alpha_i Su_i. \tag{5}
$$
This rule is well-defined because the coefficients in the chosen basis are unique.  Expression (5) shows $T$ is linear: additivity and homogeneity follow directly from properties of the scalars.

### Step 4 – Check the extension property
If $x\in U$ then the $\beta$-coefficients vanish, so (5) reduces to $T(x)=\sum\alpha_i Su_i=S(\sum\alpha_i u_i)=Sx$ by linearity of $S$.  Therefore $T$ indeed extends $S$.

---

## Problem 15

Let $(v_1,\dots,v_m)$ be a linearly **dependent** list in $V$ and suppose $W\neq\{0\}$.  We show that there exist vectors $w_1,\dots,w_m\in W$ such that **no** linear map $T\in\mathcal{L}(V,W)$ satisfies $Tv_k=w_k$ for all $k$.

---
### Step 1 – Use a dependence relation
Because the $v_k$ are dependent, there exist scalars $\alpha_1,\dots,\alpha_m$, not all zero, with
$$
\sum_{k=1}^{m}\alpha_k v_k=0. \tag{6}
$$
Pick an index $j$ for which $\alpha_j\neq0$ (such $j$ exists).

### Step 2 – Choose the $w_k$
Select a non-zero vector $a\in W$.  Define
$$
 w_j = a,\qquad w_k = 0 \; (k\neq j). \tag{7}
$$
### Step 3 – Derive a contradiction
Assume, for contradiction, that there exists $T\in\mathcal{L}(V,W)$ with $Tv_k=w_k$.  Apply $T$ to the dependence relation (6):
$$
0 = T(0)=T\Bigl(\sum_{k=1}^m\alpha_k v_k\Bigr)=\sum_{k=1}^m\alpha_k Tv_k=\sum_{k\neq j}\alpha_k w_k + \alpha_j w_j=\alpha_j a. \tag{8}
$$
Since $\alpha_j\neq0$ and $a\neq0$, equation (8) is impossible.  Hence no such linear map $T$ exists.

---

## Problem 16

Let $\dim V=n\ge2$.  We construct $S,T\in\mathcal{L}(V)$ with $ST\neq TS$.

---
### Step 1 – Define the operators on a basis
Choose a basis $(v_1,\dots,v_n)$ of $V$ and set
$$
T(v_1)=v_1,\quad T(v_k)=0\;(k\ge2); \tag{9}
$$
$$
S(v_1)=v_2,\quad S(v_k)=0\;(k\ge2). \tag{10}
$$
It is **easy to check** that both $S$ and $T$ are linear: they are defined on a basis and extended by linearity.

### Step 2 – Compute $ST$ and $TS$
For the vector $v_1+v_2$ we have
$$
\begin{align*}
ST(v_1+v_2)&=S\bigl(T(v_1)+T(v_2)\bigr)=S(v_1)=v_2,\\
TS(v_1+v_2)&=T\bigl(S(v_1)+S(v_2)\bigr)=T(v_2)=0.
\end{align*}
$$

Because $v_2\neq0$ we conclude $ST\neq TS$.









