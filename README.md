# 50.007 Machine Learning Project

|  Team Member  | Student ID |
| :-----------: | :--------: |
| Qiao Yingjie  |  1004514   |
| Zhang Peiyuan |  1004539   |
|   Huang He    |  1004561   |

## Introduction

In the repository, we present a pure python based implementation of the hidden markov model (HMM) for the task of simple sentiment analysis.

Dependencies:

-   **Python 3.6+** Only (no external libraries required)


## Part 1

Relevant code section:

-   Function that computes the emission probability: [hmm.py Line 36-67](hmm.py#L36)
-   Simple sentiment analysis based on emission probability: [hmm.py Line 298-321](hmm.py#L298)

### To run the code

```bash
python hmm.py --train "ES/train" --test "ES/dev.in" --naive
python hmm.py --train "RU/train" --test "RU/dev.in" --naive
```

### To evaluate accuracy

For ES:

```bash
python3 EvalScript/evalResult.py ./ES/dev.out ES/dev.p1.out
```

result:

```
#Entity in gold data: 255
#Entity in prediction: 1288

#Correct Entity : 184
Entity  precision: 0.1429
Entity  recall: 0.7216
Entity  F: 0.2385

#Correct Sentiment : 109
Sentiment  precision: 0.0846
Sentiment  recall: 0.4275
Sentiment  F: 0.1413
```

For RU:

```bash
python3 EvalScript/evalResult.py ./RU/dev.out RU/dev.p1.out
```

result:

```
#Entity in gold data: 461
#Entity in prediction: 1377

#Correct Entity : 301
Entity  precision: 0.2186
Entity  recall: 0.6529
Entity  F: 0.3275

#Correct Sentiment : 131
Sentiment  precision: 0.0951
Sentiment  recall: 0.2842
Sentiment  F: 0.1425
```

## Part 2 & 3

Relevant code section:

-   Function that computes the transition probability: [hmm.py Line 69-108](hmm.py#L69)
-   We use log-likelihood to compute the probability of emission and transition to avoid underflow: [hmm.py Line 100](hmm.py#L100)
-   Top-k Viterbi algorithm: [hmm.py Line 186-260](hmm.py#L186)

### To run the code

```bash
python hmm.py --train "ES/train" --test "ES/dev.in" --top-k "5"
python hmm.py --train "RU/train" --test "RU/dev.in" --top-k "5"
```

### To evaluate accuracy

### Part 2

For ES:

```bash
python3 EvalScript/evalResult.py ./ES/dev.out ES/dev.p2.out
```

result:

```
#Entity in gold data: 255
#Entity in prediction: 201

#Correct Entity : 130
Entity  precision: 0.6468
Entity  recall: 0.5098
Entity  F: 0.5702

#Correct Sentiment : 105
Sentiment  precision: 0.5224
Sentiment  recall: 0.4118
Sentiment  F: 0.4605
```

For RU:

```bash
python3 EvalScript/evalResult.py ./RU/dev.out RU/dev.p2.out
```

result:

```
#Entity in gold data: 461
#Entity in prediction: 337

#Correct Entity : 226
Entity  precision: 0.6706
Entity  recall: 0.4902
Entity  F: 0.5664

#Correct Sentiment : 156
Sentiment  precision: 0.4629
Sentiment  recall: 0.3384
Sentiment  F: 0.3910
```

### Part 3

For ES:

```bash
python3 EvalScript/evalResult.py ./ES/dev.out ES/dev.p3.out
```

result:

```
#Entity in gold data: 255
#Entity in prediction: 476

#Correct Entity : 116
Entity  precision: 0.2437
Entity  recall: 0.4549
Entity  F: 0.3174

#Correct Sentiment : 93
Sentiment  precision: 0.1954
Sentiment  recall: 0.3647
Sentiment  F: 0.2544
```

For RU:

```bash
python3 EvalScript/evalResult.py ./RU/dev.out RU/dev.p3.out
```

result:

```
#Entity in gold data: 461
#Entity in prediction: 782

#Correct Entity : 198
Entity  precision: 0.2532
Entity  recall: 0.4295
Entity  F: 0.3186

#Correct Sentiment : 132
Sentiment  precision: 0.1688
Sentiment  recall: 0.2863
Sentiment  F: 0.2124
```

## Part 4-RNN

We first tryout Recurrent Neural Network to perform this sequence labeling task. However, the result was poor.

The code we use can be found in RNN/ and specifically RNN/part4.py,RNN/script.sh. We will not use that for our finnal submission.


## Part 4-HMM

We then opt-for second order HMM model.

### Transition matrix

P(y<sub>i</sub>|y<sub>i-2</sub>,y<sub>i-1</sub> ) = COUNT(y<sub>i-2</sub>,y<sub>i-1</sub>,y<sub>i</sub>) / COUNT(y<sub>i-2</sub>,y<sub>i-1</sub>)

### Viterbi Algirithm

We also need to rewrite viterbi algorithm. 

We use S<sub>t</sub>(j, k) to denote the set of possible decoding sequences ending with y<sub>t-1</sub>=j and y<sub>t</sub>=k.

We use M<sub>t</sub>(j, k) to denote max{P(element | X) for element in S<sub>t</sub>(j, k) }

Then M can be calculated using Dynamic Programmign:

M<sub>t</sub>(j, k) = max{M<sub>t-1</sub>(i, j)* P(y<sub>k</sub>|y<sub>i</sub>,y<sub>j</sub>) * P(x=X<sub>t</sub>| y=k)}

We then perform backtracking to find out the decoding with maximum likelyhood.

### Label Smoothing

What's more, we add label smoothing in the model when calculating model parameters, shown in line 197 and 64.








To Evaluate on development set:

For ES:

```bash
python3 second_order_hmm.py --train "ES/train" --test "ES/dev.in" 

python3 EvalScript/evalResult.py ES/dev.out ES/dev.p4.out
```

For RU:

```bash
python3 second_order_hmm.py --train "RU/train" --test "RU/dev.in"
python3 EvalScript/evalResult.py RU/dev.out Ru/dev.p4.out
```

## Inference on testset
```bash
python3 second_order_hmm.py --train "ES/train" --test "ES-test/test.in" 

python3 second_order_hmm.py --train "RU/train" --test "RU-test/test.in" 
```

You can find the outout file at ES-test/test.p4.out and RU-test/test.p4.out.

## Reference
For second order HMM
```
@article{article,
author = {Yang, Sung-Hyun and Thappa, Kshav and Kabir, M Humayun and Hee-Chan, Lee},
year = {2018},
month = {04},
pages = {155014771877254},
title = {Log-Viterbi algorithm applied on second-order hidden Markov model for human activity recognition},
volume = {14},
journal = {International Journal of Distributed Sensor Networks},
doi = {10.1177/1550147718772541}
}
```
For RNN, code adapted from

```
https://github.com/pangolulu/rnn-from-scratch
```

