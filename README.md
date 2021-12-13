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

-   Function that computes the emission probability: [hmm.py Line 36-67](hmm.py#L36)
-   Simple sentiment analysis based on emission probability: [hmm.py Line 298-321](hmm.py#L298)

To run the code:

```bash
python hmm.py --train "ES/train" --test "ES/dev.in" --naive
```

## Part 2 & 3

Relevant code section:

-   Function that computes the transition probability: [hmm.py Line 69-108](hmm.py#L69)
-   We use log-likelihood to compute the probability of emission and transition to avoid underflow: [hmm.py Line 100](hmm.py#L100)
-   Top-k Viterbi algorithm: [hmm.py Line 186-260](hmm.py#L186)

To run the code:

```bash
python3 hmm.py --train "ES/train" --test "ES/dev.in" --top-k "5"

python3 hmm.py --train "RU/train" --test "RU/dev.in" --top-k "5"
```

## Part 4

TODO
python3 hmm.py --train "RU/train" --test "RU/dev.in" --top-k "5"

python3 EvalScript/evalResult.py ES/dev.out ES/dev.p4.out



python3 second_order_hmm.py --train "RU/train" --test "RU/dev.in"
python3 EvalScript/evalResult.py RU/dev.out Ru/dev.p4.out
## Evaluation