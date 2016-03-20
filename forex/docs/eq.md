# Quations memo used here


### r-kzfilter

```latex

    n: filter length
    m: number of filter applied
    alpha: rount(( n + 1) / 2, 0)

    RKZ(t, m; n) = \frac { \sum_{i = t - alpha + 1}^{t} RKZ(i, m - 1; n)  } {alpha} 

    where RKZ is Revised- Kolmogorov-Zurbenko.
```
