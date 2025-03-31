# Statistical Analysis of Contemplative Alignment Benchmark

## Summary Statistics

| Technique | Mean Score | Std Dev | Sample Size |
| --- | --- | --- | --- |
| boundless_care | 71.63 | 14.64 | 100 |
| contemplative_alignment | 74.71 | 14.86 | 100 |
| emptiness | 64.65 | 14.46 | 100 |
| mindfulness | 69.38 | 14.83 | 100 |
| non_duality | 71.26 | 15.67 | 100 |
| prior_relaxation | 68.86 | 15.27 | 100 |
| standard | 59.42 | 17.04 | 100 |

## One-way ANOVA Results

F-statistic: 11.0120

p-value: 0.0000

The difference between techniques is statistically significant (p < 0.05).

## Tukey's HSD Test for Pairwise Comparisons

```
                  Multiple Comparison of Means - Tukey HSD, FWER=0.05                  
=======================================================================================
         group1                  group2         meandiff p-adj   lower    upper  reject
---------------------------------------------------------------------------------------
         boundless_care contemplative_alignment   3.0817 0.7873  -3.3058  9.4693  False
         boundless_care               emptiness  -6.9785 0.0219 -13.3661 -0.5909   True
         boundless_care             mindfulness   -2.246 0.9446  -8.6336  4.1416  False
         boundless_care             non_duality   -0.369    1.0  -6.7566  6.0186  False
         boundless_care        prior_relaxation  -2.7668 0.8607  -9.1543  3.6208  False
         boundless_care                standard   -12.21    0.0 -18.5976 -5.8224   True
contemplative_alignment               emptiness -10.0602 0.0001 -16.4478 -3.6727   True
contemplative_alignment             mindfulness  -5.3277 0.1733 -11.7153  1.0598  False
contemplative_alignment             non_duality  -3.4507 0.6841  -9.8383  2.9368  False
contemplative_alignment        prior_relaxation  -5.8485 0.0979 -12.2361  0.5391  False
contemplative_alignment                standard -15.2917    0.0 -21.6793 -8.9042   True
              emptiness             mindfulness   4.7325 0.3017  -1.6551 11.1201  False
              emptiness             non_duality   6.6095 0.0371   0.2219 12.9971   True
              emptiness        prior_relaxation   4.2117  0.448  -2.1758 10.5993  False
              emptiness                standard  -5.2315 0.1909 -11.6191  1.1561  False
            mindfulness             non_duality    1.877  0.977  -4.5106  8.2646  False
            mindfulness        prior_relaxation  -0.5208    1.0  -6.9083  5.8668  False
            mindfulness                standard   -9.964 0.0001 -16.3516 -3.5764   True
            non_duality        prior_relaxation  -2.3978 0.9251  -8.7853  3.9898  False
            non_duality                standard  -11.841    0.0 -18.2286 -5.4534   True
       prior_relaxation                standard  -9.4432 0.0003 -15.8308 -3.0557   True
---------------------------------------------------------------------------------------
```

## Pairwise Comparisons with Standard Prompting

| Technique | Mean Difference | p-value | Significant? |
| --- | --- | --- | --- |

* p < 0.05, ** p < 0.01, *** p < 0.001

## Effect Size

Cohen's d (standard vs. contemplative_alignment): 0.9615

Interpretation: large effect

## Conclusion

The statistical analysis confirms that there are significant differences in safety scores between the prompting techniques. The post-hoc analysis identifies which specific techniques differ significantly from each other.