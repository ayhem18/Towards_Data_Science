# Chapter 4: Probability Inequalities

The most interesting part in this chapter is proving the material. 

## Mill's Inequality 

Given $Z \sim N(0, 1)$, we have 

$$
P(|Z| \geq t) \leq \sqrt{\frac{2}{\pi}} \cdot \frac{e^{-\frac{t^2}{2}}}{t} 
$$
 
for any $t > 0$. (The right hand side is negative for $t < 0$ for which the inequality does not hold.) 

My proof: 

$$ 
\begin{align*}
P(|Z| \geq t) &= P(Z > t) + P(Z < -t) = 2 P(Z > t) \\
\implies P(Z > t) \cdot t &= 2 \cdot t \int_{t}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}} dx \\
&= \sqrt{\frac{2}{\pi}}  \int_{t}^{\infty} t \cdot e^{-\frac{x^2}{2}} dx \\
&\leq \sqrt{\frac{2}{\pi}}  \int_{t}^{\infty} x \cdot e^{-\frac{x^2}{2}} dx && \text{since x $\geq$ t}\\
&= \sqrt{\frac{2}{\pi}}  \cdot e^{-\frac{t^2}{2}}
\end{align*}
$$

Hence the result. 

## Pearson's Coefficient 

Given $X$ and $Y$ are random variables, we have 

$$
\rho_{X,Y} = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}
$$

We have for any 2 random variables $X$ and $Y$,   

$$
-1 \leq \rho_{X,Y} \leq 1
$$

My proof: 

Let's start with a simple yet important result; $$var(X) \geq 0$$ for any random variable 

as well as some important properties of Covariance

$$ 
Cov(aX, bY) = ab Cov(X,Y)
$$


$$
\begin{align*}
Var(X + Y) &= Var(X) + Var(Y) + 2Cov(X,Y) \\
Var(\frac{X}{\sqrt{Var(X)}} + \frac{Y}{\sqrt{Var(Y)}}) &= 
Var(\frac{X}{\sqrt{Var(X)}}) + Var(\frac{Y}{\sqrt{Var(Y)}}) + 2Cov(\frac{X}{\sqrt{Var(X)}}, \frac{Y}{\sqrt{Var(Y)}}) \\
&= 2 + 2 \cdot \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}} \\
&= 2 + 2 \rho_{X,Y}
\end{align*}
$$

With $Var(Z) \geq 0$, we have 

$$
2 + 2 \rho_{X,Y} \geq 0 \implies \rho_{X,Y} \geq -1
$$ 

Since $Var(aX) = a^2 Var(X)$, then we use $-Y$ instead of $Y$ in the same argument above and: 

$$
\begin{align*}
Var(\frac{X}{\sqrt{Var(X)}} - \frac{Y}{\sqrt{Var(Y)}}) &= 
Var(\frac{X}{\sqrt{Var(X)}}) + Var(\frac{Y}{\sqrt{Var(Y)}}) - 2Cov(\frac{X}{\sqrt{Var(X)}}, \frac{Y}{\sqrt{Var(Y)}}) \\
&= 2 - 2 \cdot \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}} \\
&= 2 - 2 \rho_{X,Y}
\end{align*}
$$

With $Var(Z) \geq 0$, we have 

$$
2 - 2 \rho_{X,Y} \geq 0 \implies \rho_{X,Y} \leq 1
$$

and Hence: 

$$
-1 \leq \rho_{X,Y} \leq 1
$$

for any random variables $X$ and $Y$. 



## Chauchy-Schwarz Inequality  

Given $X$ and $Y$ are random variables, we have 

$$
E[XY] \leq \sqrt{E[X^2]E[Y^2]}
$$

My proof:

Using the fact that $\rho_{X,Y} \leq 1$, we have 

$$
\begin{align*}
Cov(X,Y) &\leq \sqrt{Var(X)Var(Y)} \\
E[XY] &\leq \sqrt{Var(X)Var(Y)} + E[X]E[Y] \\
\end{align*}
$$

My next step is to show that 

$$
\begin{align*}
| \sqrt{Var(X)Var(Y)} + E[X]E[Y] | &\leq \sqrt{E[X^2]E[Y^2]} \\
\iff
Var(X)Var(Y) + 2E[X]E[Y] \sqrt{Var(X)Var(Y)} + E[X]^2E[Y]^2 &\leq E[X^2]E[Y^2] \\
\text{rewriting $E[X^2] = Var(X) + E[X]^2$ and $E[Y^2] = Var(Y) + E[Y]^2$} \\ 
\iff
Var(X)Var(Y) + 2E[X]E[Y] \sqrt{Var(X)Var(Y)} + E[X]^2E[Y]^2 &\leq Var(X)Var(Y) + E[X]E[Y] + Var(Y)E[X]^2 + Var(X)E[Y]^2 \\
\text{eliminating common terms} \\
\iff
2E[X]E[Y] \sqrt{Var(X)Var(Y)} &\leq  Var(Y)E[X]^2 + Var(X)E[Y]^2
\end{align*}
$$

Using the famous inequality $a + b \geq 2\sqrt{ab}$ for any $a,b \geq 0$ with $a = Var(X)E[Y]^2$ and $b = Var(Y)E[X]^2$, we have 

$$
Var(Y)E[X]^2 + Var(X)E[Y]^2 \geq 2 \sqrt{Var(X)E[Y]^2 \cdot Var(Y)E[X]^2} = 2E[X]E[Y] \sqrt{Var(X)Var(Y)}
$$

Hence, 

$$
\begin{align*}
| \sqrt{Var(X)Var(Y)} + E[X]E[Y] | &\leq \sqrt{E[X^2]E[Y^2]} \\
\implies
E[XY] &\leq \sqrt{E[X^2]E[Y^2]}
\end{align*}
$$


