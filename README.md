# 50.007 Machine Learning Project

|  Team Member  | Student ID |
| :-----------: | :--------: |
| Qiao Yingjie  |  1004514   |
| Zhang Peiyuan |  1004539   |
|   Huang He    |  1004561   |

## Introduction

In the repository, we present a pure python based implementation of the hidden markov model (HMM) and second-order hidden markov model for the task of simple sentiment analysis.

**Please refer to our [Report](Report.pdf) for implementation details and results.**

### Dependencies:

-   **Python 3.6+** Only (no external libraries required)

### Demo

Run all trainings and evaluations for Part 1,2,3,4 at one go:

```bash
./run_all
```

## Part 1

Relevant code section:

-   Function that computes the emission probability: [hmm.py Line 36-67](hmm.py#L36)
-   Introducing $k=1$ for calculating the emission probability for \<UNK>: [hmm.py Line 60-65](hmm.py#L60)
-   Simple sentiment analysis based on emission probability: [hmm.py Line 298-321](hmm.py#L298)

### To run the code

```bash
python3 hmm.py --train "ES/train" --test "ES/dev.in" --naive
python3 hmm.py --train "RU/train" --test "RU/dev.in" --naive
```

## Part 2 & 3

Relevant code section:

-   Function that computes the transition probability: [hmm.py Line 69-108](hmm.py#L69)
-   We use log-likelihood to compute the probability of emission and transition to avoid underflow: [hmm.py Line 100](hmm.py#L100)
-   We also introduce a smoothing factor which is a small constant that will be used to estimate close-to-zero probability to avoid undefined log-likelihood probabilities. [hmm.py Line 100](hmm.py#L100)
-   Top-k Viterbi algorithm: [hmm.py Line 186-260](hmm.py#L186)

### To run the code

```bash
python3 hmm.py --train "ES/train" --test "ES/dev.in" --top-k "5"
python3 hmm.py --train "RU/train" --test "RU/dev.in" --top-k "5"
```

## Part 4: RNN & 2nd Order HMM

We implement the second order HMM in [second_order_hmm.py](second_order_hmm.py)

### To run the code

```bash
python3 second_order_hmm.py --train "ES/train" --test "ES/dev.in"
python3 second_order_hmm.py --train "RU/train" --test "RU/dev.in"
```

## Acknowledgement

We refer to the following resources when implementing this project:

-   Sung-Hyun, Yang, et al. “Log-Viterbi Algorithm Applied on Second-Order Hidden Markov Model for Human Activity Recognition.” International Journal of Distributed Sensor Networks, Apr. 2018, doi:[10.1177/1550147718772541.](https://journals.sagepub.com/doi/10.1177/1550147718772541)
-   RNN: [pangolulu / rnn-from-scratch](https://github.com/pangolulu/rnn-from-scratch)
