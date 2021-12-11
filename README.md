# 50.007 Machine Learning Project

|  Team Member  | Student ID |
| :-----------: | :--------: |
| Qiao Yingjie  |  1004514   |
| Zhang Peiyuan |  1004539   |
|   Huang He    |  1004561   |

## Introduction

In the repository, we present a pure python based implementation of the hidden markov model (HMM) for the task of simple sentiment analysis.

Dependencies:

-   Part 1, 2, 3:
    -   **Python 3.6+** Only (no external libraries required)
-   Part 4:
    -   **Python 3.6+**
    -   **NumPy**

## Part 1

Relevant code section:

-   Function that computes the emission probability: [hmm.py Line 35-66](hmm.py#L35)
-   Simple sentiment analysis based on emission probability: [hmm.py Line 287-310](hmm.py#L287)

To run the code:

```bash
python hmm.py --train "ES/train" --test "ES/dev.in" --naive
```

## Part 2 & 3

Relevant code section:

-   Function that computes the transition probability: [hmm.py Line 68-107](hmm.py#L68)
-   We use log-likelihood to compute the probability of emission and transition to avoid underflow: [hmm.py Line 99](hmm.py#L99)
-   Top-k Viterbi algorithm: [hmm.py Line 185-259](hmm.py#L185)

To run the code:

```bash
python hmm.py --train "ES/train" --test "ES/dev.in" --top-k "5"
```

## Part 4

TODO
