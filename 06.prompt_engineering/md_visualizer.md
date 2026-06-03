**Understanding the arithmetic‑progression (AP) sum formula**

An arithmetic progression is a list of numbers in which each term after the first is obtained by adding a constant difference \(d\) to the preceding term.  
If we denote  

* the first term by \(a\),  
* the common difference by \(d\),  
* the number of terms by \(n\),

then the \(k\)-th term is  

\[
a_k = a + (k-1)d \qquad (k = 1,2,\dots ,n).
\]

The sum of the first \(n\) terms, \(S_n\), can be found without adding each term individually. The classic derivation (attributed to Gauss) pairs the series with its reverse:

\[
\begin{aligned}
S_n &= a + (a+d) + (a+2d) + \dots + \bigl[a+(n-2)d\bigr] + \bigl[a+(n-1)d\bigr] \\
S_n &= \bigl[a+(n-1)d\bigr] + \bigl[a+(n-2)d\bigr] + \dots + (a+d) + a .
\end{aligned}
\]

Adding these two expressions term‑by‑term gives

\[
2S_n = n\bigl[\,a + (a+(n-1)d)\,\bigr] = n\bigl(2a + (n-1)d\bigr).
\]

Hence the sum formula is  

\[
\boxed{S_n = \frac{n}{2}\,\bigl(2a + (n-1)d\bigr)}.
\]

A convenient equivalent, especially when the first and last terms are known, is  

\[
\boxed{S_n = \frac{n}{2}\,(a_{\text{first}} + a_{\text{last}})}.
\]

---

### Applying the formula to the problem

**Problem:** Find the sum of all even numbers between 1 and 50.

1. **Identify the AP parameters**  
   * First even number \(a = 2\).  
   * Common difference \(d = 2\) (each even number is 2 more than the previous).  
   * Last even number \(L = 50\).

2. **Determine the number of terms \(n\).**  
   The \(n\)-th term of this AP is \(a + (n-1)d = 50\).

   \[
   2 + (n-1) \cdot 2 = 50 \;\Longrightarrow\; (n-1) \cdot 2 = 48 \;\Longrightarrow\; n-1 = 24 \;\Longrightarrow\; n = 25.
   \]

   So there are 25 even numbers from 2 up to 50.

3. **Use the sum‑of‑AP formula**  

   \[
   S = \frac{n}{2}\,(a + L) = \frac{25}{2}\,(2 + 50) = \frac{25}{2}\times 52.
   \]

   Compute:

   \[
   \frac{25}{2}\times 52 = 25 \times 26 = 650.
   \]

---

\[
\boxed{650}
\]

Thus, the sum of all even numbers between 1 and 50 is **650**.
