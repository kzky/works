# Quations memo used here


### r-kzfilter

```latex

    n: filter length
    m: number of filter applications
    alpha: rount(( n + 1) / 2, 0)

    RKZ(t, m; n) = \frac { \sum_{i = t - alpha + 1}^{t-1} RKZ(i, m; n) + RKZ(t, m-1; n) } {alpha} 

    where RKZ is Revised- Kolmogorov-Zurbenko.
```
