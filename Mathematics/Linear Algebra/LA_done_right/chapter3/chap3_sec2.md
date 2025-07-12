![](./images/img4.png)


# Results: Fundamental Theorem of Linear Maps and Key Consequences

Let $T:V\to W$ be a linear map between finite–dimensional vector spaces.

---

### Fundamental Theorem of Linear Maps (FTLM)
For every such $T$ we have
$$
 d(V)
 \,=\, d\bigl(N(T)\bigr) + d\bigl(R(T)\bigr).
$$
Here $d(\cdot)$ denotes dimension, $N(T)$ the null space (kernel) and $R(T)$ the range (image).

---

### Result 1 (Injectivity preserves independence)
If $v_1,\dots,v_n$ are linearly independent **and** $T$ is injective, then the images $T(v_1),\dots,T(v_n)$ are linearly independent.

**Proof.**  Assume a relation
$$
\begin{align*}
\sum_{i=1}^n \alpha_i T(v_i) &= 0 \\ 
T\Bigl( \sum_{i=1}^n \alpha_i v_i \Bigr) &= 0 \\
\implies \sum_{i=1}^n \alpha_i v_i = 0 && \text{because $T$ is injective}
\end{align*} \\
$$

Independence of the $v_i$ forces every $\alpha_i=0$, so the images are independent.

---

### Result 2 (Independence of images implies independence of pre-images)
If $T(v_1),\dots,T(v_n)$ are linearly independent, then so are $v_1,\dots,v_n$.

**Proof.**  Suppose
$$
 \sum_{i=1}^n \alpha_i v_i = 0.
$$
Apply $T$:
$$
 T\Bigl( \sum_{i=1}^n \alpha_i v_i \Bigr) = \sum_{i=1}^n \alpha_i T(v_i) = 0.
$$
Independence of the $T(v_i)$ gives $\alpha_i=0$ for every $i$.

---

### Result 3 (Spanning sets map onto the range)
If $v_1,\dots,v_n$ span $V$, then $T(v_1),\dots,T(v_n)$ span $R(T)$.

**Proof.**  

Inclusion $\operatorname{span}\{T(v_i)\}\subseteq R(T)$ is clear.  

For the reverse inclusion take any $x\in R(T)$.  There is some $u\in V$ with $x=T(u)$.  Because the $v_i$ span $V$ we can write:

$$
 u = \sum_{i=1}^n \alpha_i v_i \quad \Longrightarrow \quad x = T(u) = \sum_{i=1}^n \alpha_i T(v_i).
$$

Hence $x$ lies in the span of the images, so
$$
 R(T) = \operatorname{span}\{T(v_1),\dots,T(v_n)\}. 
$$

---

### Result 4 (A basis adapted to $T$)

Suppose $V$ possesses a basis

$$
 \bigl(u_1,\dots,u_m,\;v_1,\dots,v_n\bigr)
$$
with
$$
 T(u_i)=0 \quad (1\le i\le m),\qquad T(v_1),\dots,T(v_n) \text{ linearly independent.}
$$

Then
$$
 (u_1,\dots,u_m) \text{ is a basis of } N(T),\qquad
 \bigl(T(v_1),\dots,T(v_n)\bigr) \text{ is a basis of } R(T).
$$

**Proof.**
First, every $u_i$ lies in $N(T)$, hence
$$
 \operatorname{span}\{u_i\}\subseteq N(T). \tag{2}
$$
Conversely, take $u\in N(T)$ and expand it in the given basis:
$$
\begin{align*}
\sum_{i=1}^m \alpha_i u_i + \sum_{j=1}^n \beta_j v_j &= u \\ 
\sum_{j=1}^n \beta_j T(v_j)&= T(u) \\ 
\implies \sum_{j=1}^n \beta_j T(v_j) &= 0
\end{align*}
$$

Because the $T(v_j)$ are independent, every $\beta_j=0$.  Thus $u\in\operatorname{span}\{u_i\}$ and together with (2) we have

$$
 N(T)=\operatorname{span}\{u_i\}.
$$

To identify $R(T)$ 
$$
\begin{align*}
x\in R(T) &\implies x = T(w) \text{ for some } w\in V \\
\implies w &= \sum_{i=1}^m \gamma_i u_i + \sum_{j=1}^n \delta_j v_j \\
x &= T(w)= \sum_{i=1}^m \gamma_i T(u_i) + \sum_{j=1}^n \delta_j T(v_j) \\
x &= 0 + \sum_{j=1}^n \delta_j T(v_j) \in \operatorname{span}\{T(v_j)\} \\
\therefore R(T)&= \operatorname{span}\{T(v_1),\dots,T(v_n)\}.
\end{align*}
$$

Since the $T(v_j)$ are already known to be linearly independent, they form a basis of $R(T)$.


# Problem 1  
Give an example of a linear map $T$ with $d\,N(T)=3$ and $d\,R(T)=2$.

$$
\begin{align*}
d\,N(T)=3,\;d\,R(T)=2 &\stackrel{\text{FTLM}}{\implies} d(V)=5 \\
\text{Take } V=W=\mathbb F^{5},\;\bigl(e_1,\dots,e_5\bigr) \text{ standard basis} \\[4pt]
\text{Define }T(e_i)=0\,(i\le3),\;T(e_i)=e_i\,(i=4,5) &&\{\text{linear by basis assignment}\} \\
N(T)=\operatorname{span}\{e_1,e_2,e_3\}\;\implies d\,N(T)=3 \\
R(T)=\operatorname{span}\{e_4,e_5\}\;\Rightarrow\;d\,R(T)=2 \\
\text{Hence }T\text{ fulfils the requirement (cf. Result 4).}
\end{align*}
$$

---

# Problem 2  
Suppose $S,T\in\mathcal L(V)$ with $R(S)\subseteq N(T)$.  Prove $(ST)^2=0$.

$$
\begin{align*}
R(S)\subseteq N(T) &\implies T S=0 \tag{*} \\
(ST)^2 = STST &= S(\underbrace{TS}_{0})T = S0T = 0.
\end{align*}
$$
Thus $(ST)^2=0$.

---

# Problem 3  
Let $v_1,\dots,v_m\in V$ and define $T\in\mathcal L(\mathbb F^{m},V)$ by
$$T(z_1,\dots,z_m)=z_1v_1+\cdots+z_m v_m.$$
(a) Which property of $T$ corresponds to $v_1,\dots,v_m$ spanning $V$?  
(b) Which property corresponds to $v_1,\dots,v_m$ being linearly independent?

$$
\begin{align*}
& (a)\; v_1,\dots,v_m \text{ span }V &&\Longleftrightarrow&& R(T)=V\;\text{(surjective)}\\[4pt]
& (b)\; v_1,\dots,v_m \text{ independent} &&\Longleftrightarrow&& N(T)=\{0\}\;\text{(injective)}
\end{align*}
$$

---

# Problem 4  
Show that $$\bigl\{T\in\mathcal L(\mathbb R^{5},\mathbb R^{4}) : d\,N(T)>2\bigr\}$$ is **not** a subspace of $\mathcal L(\mathbb R^{5},\mathbb R^{4})$.

The solution is to construct $T_1,T_2$ with 
$$
d\,N(T_k)\ge3 \text{ yet } d\,N(T_1+T_2)\le2.
$$

Choose bases $(e_1,\dots,e_5)$ for $\mathbb R^5,\; (f_1,\dots,f_4)$ for $\mathbb R^4.$


Let $T_1$ such that $$T_1(e_1)=T_1(e_2)=T_1(e_5)=0,\; T_1(e_3)=f_3,\; T_1(e_4)=f_4$$
since $e1, e2, e3, .. e5$ is a basis and $f3$ and $f4$ are independent then Result 4 applies
so $$d\,N(T_1)=3$$
Let $T_2$ such that $$T_2(e_3)=T_2(e_4)=T_2(e_5)=0,\; T_2(e_1)=f_1,\; T_2(e_2)=f_2$$
since $e1, e2, e3, e4, e5$ is a basis and $f1$ and $f2$ are independent then Result 4 applies
so $$d\,N(T_2)=3$$
Let $T_1+T_2$ such that $$(T_1+T_2)(e_5)=0,\; (T_1+T_2)(e_k)=f_k\;(k=1,2,3,4)$$
since $e1, e2, e3, e4, e5$ is a basis and $f_1, f_2, f_3, f_4$ are independent then Result 4 applies
so $$d\,N(T_1+T_2)=1$$

Thus $T_1,T_2$ are in the set, while $T_1+T_2$ is not.

---

# Problem 5
Give an example of $T \in \mathcal{L}(\mathbb{R}^4)$ such that range $T$ = null $T$.

Let $(e_1, e_2, e_3, e_4)$ be the standard basis of $\mathbb R^4$. We need $d(N(T))=d(R(T))$, so by the FTLM, $d(\mathbb R^4)=4=2d(N(T)) \implies d(N(T))=2$.

$$
\begin{align*}
\text{Define } T(e_1)&=0,\; T(e_2)=0,\; T(e_3)=e_1,\; T(e_4)=e_2. \\[6pt]
\text{Let } u_1=e_1, u_2=e_2, &\text{ and } v_1=e_3, v_2=e_4. \\
(u_1, u_2, v_1, v_2) &\text{ is a basis for } \mathbb R^4. \\
T(u_1)=T(u_2)&=0. \\
T(v_1)=e_1, T(v_2)=e_2 &\text{ are linearly independent.} \\[6pt]
\text{Result 4 applies}&\text{, giving:} \\
N(T) &= \operatorname{span}\{u_1, u_2\} = \operatorname{span}\{e_1, e_2\}. \\
R(T) &= \operatorname{span}\{T(v_1), T(v_2)\} = \operatorname{span}\{e_1, e_2\}. \\[6pt]
\therefore N(T) &= R(T).
\end{align*}
$$

---

# Problem 6
Prove that there does not exist $T \in \mathcal{L}(\mathbb{R}^5)$ such that range $T$ = null $T$.

$$
\begin{align*}
\text{Assume for contradiction } &\text{that such a } T \text{ exists.} \\
R(T) = N(T) &\implies d(R(T)) = d(N(T)). \\
\text{Let } k &= d(R(T)) = d(N(T)). \\[6pt]
\text{By the FTLM, } d(\mathbb R^5) &= d(R(T)) + d(N(T)). \\
5 &= k + k = 2k. \\[6pt]
\text{This implies } k=2.5&, \text{ which is impossible as dimension must be an integer.} \\
\therefore \text{No such } T &\text{ can exist.}
\end{align*}
$$

---

# Problem 7
Show that $\{T \in \mathcal{L}(V, W) : T \text{ is not injective}\}$ is not a subspace of $\mathcal{L}(V, W)$, where $2 \le \dim V \le \dim W$.

Let $n = \dim V$ and $m = \dim W$. The set in question is $$\{T \in \mathcal{L}(V, W) : d(N(T)) > 0 \}.$$ We find $T_1, T_2$ in this set whose sum is not.


Choose bases $(v_1, \dots, v_n)$ for $V, (w_1, \dots, w_m)$ for $W.$

Define 
$$T_1(v_1) = 0, T_1(v_i)=w_i \text{ for } i=2, \dots, n.$$

$$N(T_1)=\operatorname{span}\{v_1\}, \text{ so by result 4 ~~} d(N(T_1))=1 $$
$$T_1 \text{ is not injective.} $$

Define 

$$T_2(v_2) = 0, T_2(v_i)=w_i \text{ for } i \in \{1, 3, \dots, n\}.$$

$$N(T_2)=\operatorname{span}\{v_2\}, \text{ so by result 4 ~~} d(N(T_2))=1 $$
$$T_2 \text{ is not injective.} $$

Consider $T = T_1 + T_2:$


$$
\begin{align*}
T(v_1) &= T_1(v_1) + T_2(v_1) = 0 + w_1 = w_1. \\
T(v_2) &= T_1(v_2) + T_2(v_2) = w_2 + 0 = w_2. \\
T(v_i) &= T_1(v_i) + T_2(v_i) = w_i + w_i = 2w_i \text{ for } i \ge 3. \\[6pt]
\text{Let } v &= \sum \alpha_i v_i \in N(T). \\
T(v) &= \alpha_1 w_1 + \alpha_2 w_2 + \sum_{i=3}^n 2\alpha_i w_i = 0. \\
\text{Since } (w_i) \text{ are independent, } &\alpha_1=\alpha_2=0 \text{ and } 2\alpha_i=0, \text{ so all } \alpha_i=0. \\
\text{Therefore, } v&=0 \text{ and } N(T)=\{0\}. \\
\text{Therefore, } T \text{ is injective.}
\end{align*}
$$

Thus $T_1,T_2$ are in the set, while $T_1+T_2$ is not.

---

# Problem 8
Show that $\{T \in \mathcal{L}(V, W) : T \text{ is not surjective}\}$ is not a subspace, where $\dim V \ge \dim W \ge 2$.

Let $n = \dim V$ and $m = \dim W$. The set is $$\{T \in \mathcal{L}(V, W) : d(R(T)) < m \}.$$    


Choose bases $(v_1, \dots, v_n)$ for $V, (w_1, \dots, w_m)$ for $W.$

Define 
$$T_1(v_1)=0, T_1(v_i)=w_i \text{ for } i=2, \dots, m.$$
$$R(T_1) = \operatorname{span}\{w_2, \dots, w_m\}, \\[2pt]
\text{ so by result 4 ~~} d(R(T_1))=m-1 < m.$$

Define 
$$T_2(v_1)=w_1, T_2(v_i)=0 \text{ for } i=2, \dots, m.$$

$$R(T_2) = \operatorname{span}\{w_1\}, \\[2pt]
\text{ so by result 4 ~~} d(R(T_2))=1 < m \text{ (as } m \ge 2). 
$$

Consider $T = T_1 + T_2:$


$$
\begin{align*}
\text{Consider } T = T_1 + T_2: & \\
T(v_i) = T_1(v_i) + T_2(v_i) &= w_i \text{ for } i=1, \dots, m. \\
\text{Images include all basis vectors of } W, &\text{ so } R(T) = W. \\
\therefore T \text{ is surjective, and not in the set.}
\end{align*}
$$

Thus $T_1,T_2$ are in the set, while $T_1+T_2$ is not.

---



