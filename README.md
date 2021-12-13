# 50.007 Machine Learning Project

|  Team Member  | Student ID |
| :-----------: | :--------: |
| Qiao Yingjie  |  1004514   |
| Zhang Peiyuan |  1004539   |
|   Huang He    |  1004561   |

## Introduction

In the repository, we present a pure python based implementation of the hidden markov model (HMM) and second-order hidden markov model for the task of simple sentiment analysis.

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

## Part 4: RNN & 2nd Order HMM

### RNN

We first try Recurrent Neural Network (RNN) to perform this sequence labeling task. However, the result was poor.

The code we use can be found in RNN/ and specifically RNN/part4.py,RNN/script.sh. We will not use that for our final submission.

### Second Order Hidden Markov Model

We then opt-for second order HMM model.

### Transition matrix

P(y<sub>i</sub>|y<sub>i-2</sub>,y<sub>i-1</sub> ) = COUNT(y<sub>i-2</sub>,y<sub>i-1</sub>,y<sub>i</sub>) / COUNT(y<sub>i-2</sub>,y<sub>i-1</sub>)

### Viterbi Algorithm

We also need to rewrite viterbi algorithm.

We use S<sub>t</sub>(j, k) to denote the set of possible decoding sequences ending with y<sub>t-1</sub>=j and y<sub>t</sub>=k.

We use M<sub>t</sub>(j, k) to denote max{P(element | X) for element in S<sub>t</sub>(j, k) }

Then M can be calculated using Dynamic Programming:

M<sub>t</sub>(j, k) = max{M<sub>t-1</sub>(i, j)_ P(y<sub>k</sub>|y<sub>i</sub>,y<sub>j</sub>) _ P(x=X<sub>t</sub>| y=k)}

We then perform backtracking to find out the decoding with maximum likelihood.

### Label Smoothing

What's more, we add label smoothing in the model when calculating model parameters, shown in line 197 and 64 in second_order_hmm.py .

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

## Evaluation Results for Part 1,2,3,4 on ES and RU development set

You can run all trainings and evaluations for Part 1,2,3,4 at one go:

```bash
./run_all
```

Result summary as follows:

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th colspan="2">Part 1</th>
    <th colspan="2">Part 2</th>
    <th colspan="2">Part 3</th>
    <th colspan="2">Part 4</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td></td>
    <td>Filename</td>
    <td colspan="2">dev.p1.out</td>
    <td colspan="2">dev.p2.out</td>
    <td colspan="2">dev.p3.out</td>
    <td colspan="2">dev.p4.out</td>
  </tr>
  <tr>
    <td></td>
    <td>Dataset</td>
    <td>ES</td>
    <td>RU</td>
    <td>ES</td>
    <td>RU</td>
    <td>ES</td>
    <td>RU</td>
    <td>ES</td>
    <td>RU</td>
  </tr>
  <tr>
    <td rowspan="4">Entity</td>
    <td>#Correct</td>
    <td>205</td>
    <td>335</td>
    <td>130</td>
    <td>226</td>
    <td>116</td>
    <td>198</td>
    <td>110</td>
    <td>175</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>0.1183</td>
    <td>0.1604</td>
    <td>0.6468</td>
    <td>0.6706</td>
    <td>0.2437</td>
    <td>0.2532</td>
    <td>0.6077</td>
    <td>0.4605</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.8039</td>
    <td>0.7267</td>
    <td>0.5098</td>
    <td>0.4902</td>
    <td>0.4549</td>
    <td>0.4295</td>
    <td>0.4314</td>
    <td>0.3796</td>
  </tr>
  <tr>
    <td>F Score</td>
    <td>0.2062</td>
    <td>0.2627</td>
    <td>0.5702</td>
    <td>0.5664</td>
    <td>0.3174</td>
    <td>0.3186</td>
    <td>0.5046</td>
    <td>0.4162</td>
  </tr>
  <tr>
    <td rowspan="4">Sentiment</td>
    <td>#Correct</td>
    <td>113</td>
    <td>136</td>
    <td>105</td>
    <td>156</td>
    <td>93</td>
    <td>132</td>
    <td>89</td>
    <td>130</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td>0.0652</td>
    <td>0.0651</td>
    <td>0.5224</td>
    <td>0.4629</td>
    <td>0.1954</td>
    <td>0.1688</td>
    <td>0.4917</td>
    <td>0.3421</td>
  </tr>
  <tr>
    <td>Recall</td>
    <td>0.4431</td>
    <td>0.2950</td>
    <td>0.4118</td>
    <td>0.3384</td>
    <td>0.3647</td>
    <td>0.2863</td>
    <td>0.3490</td>
    <td>0.2820</td>
  </tr>
  <tr>
    <td>F Score</td>
    <td>0.1137</td>
    <td>0.1067</td>
    <td>0.4605</td>
    <td>0.3910</td>
    <td>0.2544</td>
    <td>0.2124</td>
    <td>0.4083</td>
    <td>0.3092</td>
  </tr>
</tbody>
</table>

## Acknowledgement

We refer to the following resources when implementing this project:

-   Sung-Hyun, Yang, et al. “Log-Viterbi Algorithm Applied on Second-Order Hidden Markov Model for Human Activity Recognition.” International Journal of Distributed Sensor Networks, Apr. 2018, doi:[10.1177/1550147718772541.](https://journals.sagepub.com/doi/10.1177/1550147718772541)
-   RNN: [pangolulu / rnn-from-scratch](https://github.com/pangolulu/rnn-from-scratch)
