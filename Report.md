# 50.007 Machine Learning Project Report

|  Team Member  | Student ID |
| :-----------: | :--------: |
| Qiao Yingjie  |  1004514   |
| Zhang Peiyuan |  1004539   |
|   Huang He    |  1004561   |

## Introduction

In this project, we present a pure python based implementation of the hidden markov model (HMM) and the second-order hidden markov model for the task of simple sentiment analysis.

Dependencies:

-   **Python 3.6+ Only** (no external libraries required)

Please refer to the [README.md](README.md) for instructions on how to run the project. All outputs are saved in the `outputs` folder. Implementation details and results are reported below.

## Part 1

Relevant code section:

-   Function that computes the emission probability: [hmm.py Line 36-67](hmm.py#L36)
-   Introducing $k=1$ for calculating the emission probability for \<UNK>: [hmm.py Line 60-65](hmm.py#L60)
-   Simple sentiment analysis based on emission probability: [hmm.py Line 298-321](hmm.py#L298)

## Part 2 & 3

Relevant code section:

-   Function that computes the transition probability: [hmm.py Line 69-108](hmm.py#L69)
-   We use log-likelihood to compute the probability of emission and transition to avoid underflow: [hmm.py Line 100](hmm.py#L100)
-   We also introduce a smoothing factor which is a small constant that will be used to estimate close-to-zero probability to avoid undefined log-likelihood probabilities. [hmm.py Line 100](hmm.py#L100)
    ```python
    prob_func = lambda x: math.log(x) if x > 0 else math.log(smoothing_factor)
    ```
-   Top-k Viterbi algorithm: [hmm.py Line 186-260](hmm.py#L186)

## Part 4: RNN & 2nd Order HMM

### RNN

We first attempted using a pure python implementation of Recurrent Neural Network (RNN) to perform this sequence labeling task. However, the result was poor. Relevant code can be found in [RNN](RNN) folder, but we will not use that for our final submission.

### Second Order Hidden Markov Model

We then opt-for the second order HMM model. Relevant code: [second_order_hmm.py](second_order_hmm.py)

### Transition matrix

$$
P(y_i \mid y_{i-2}, y_{i-1}) = \frac{ \texttt{count}(y_i, y_{i-2}, y_{i-1}) }{ \texttt{count}(y_{i-2}, y_{i-1}) }
$$

### Viterbi Algorithm

We also need to rewrite Viterbi algorithm.

We use $S_t(j, k)$ to denote the set of possible decoding sequences ending with $y_{t-1}=j$ and $y_t=k$.

We use $M_t(j, k)$ to denote $\underset{i \in S_t(j, k)}{\operatorname{max}} \{ P(i | X) \} $

Then $M$ can be calculated using Dynamic Programming:

$$
M_t(j, k) = \max_i \{ M_{t-1} (i, j) \times P(y_k \mid y_i, y_j) \times P(x=X_t \mid y=k)    \}
$$

We then perform backtracking to find out the decoding with maximum likelihood.

### Label Smoothing

Additionally, we add Laplacian smoothing as label smoothing in the model when calculating model parameters, we set $\lambda = 1$ for calculating transition probability and $\lambda = 0.3$ for calculating emission probability.

$$
P_{\lambda} (X^{(j)} = a_{jl} \mid Y = c_k) = \frac { \sum_{i=1}^N I (x_i^{(j)} = a_{jl}, y_i = c_k ) + \lambda   } { \sum_{i=1}^N I ( y_i = c_k ) + S_j\lambda }
$$

## Evaluation Results on ES & RU Dev Set

Please refer to our [README](README.md) for instructions on how to run the code.

Please refer to the [outputs](outputs) folder for the output files. We summarize the results as follows:

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th colspan="4" style="text-align:center;vertical-align:middle;">Entity</th>
    <th colspan="4" style="text-align:center;vertical-align:middle;">Sentiment</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align:center;vertical-align:middle;"></td>
    <td style="text-align:center;vertical-align:middle;"></td>
    <td style="text-align:center;vertical-align:middle;">#Correct</td>
    <td style="text-align:center;vertical-align:middle;">Precision</td>
    <td style="text-align:center;vertical-align:middle;">Recall</td>
    <td style="text-align:center;vertical-align:middle;">F Score</td>
    <td style="text-align:center;vertical-align:middle;">#Correct</td>
    <td style="text-align:center;vertical-align:middle;">Precision</td>
    <td style="text-align:center;vertical-align:middle;">Recall</td>
    <td style="text-align:center;vertical-align:middle;">F Score</td>
  </tr>
  <tr>
    <td rowspan="4">ES</td>
    <td style="text-align:center;vertical-align:middle;">dev.p1.out</td>
    <td style="text-align:center;vertical-align:middle;">205</td>
    <td style="text-align:center;vertical-align:middle;">0.1183</td>
    <td style="text-align:center;vertical-align:middle;">0.8039</td>
    <td style="text-align:center;vertical-align:middle;">0.2062</td>
    <td style="text-align:center;vertical-align:middle;">113</td>
    <td style="text-align:center;vertical-align:middle;">0.0652</td>
    <td style="text-align:center;vertical-align:middle;">0.4431</td>
    <td style="text-align:center;vertical-align:middle;">0.1137</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p2.out</td>
    <td style="text-align:center;vertical-align:middle;">130</td>
    <td style="text-align:center;vertical-align:middle;">0.6468</td>
    <td style="text-align:center;vertical-align:middle;">0.5098</td>
    <td style="text-align:center;vertical-align:middle;">0.5702</td>
    <td style="text-align:center;vertical-align:middle;">105</td>
    <td style="text-align:center;vertical-align:middle;">0.5224</td>
    <td style="text-align:center;vertical-align:middle;">0.4118</td>
    <td style="text-align:center;vertical-align:middle;">0.4605</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p3.out</td>
    <td style="text-align:center;vertical-align:middle;">116</td>
    <td style="text-align:center;vertical-align:middle;">0.2437</td>
    <td style="text-align:center;vertical-align:middle;">0.4549</td>
    <td style="text-align:center;vertical-align:middle;">0.3174</td>
    <td style="text-align:center;vertical-align:middle;">93</td>
    <td style="text-align:center;vertical-align:middle;">0.1954</td>
    <td style="text-align:center;vertical-align:middle;">0.3647</td>
    <td style="text-align:center;vertical-align:middle;">0.2544</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p4.out</td>
    <td style="text-align:center;vertical-align:middle;">110</td>
    <td style="text-align:center;vertical-align:middle;">0.6077</td>
    <td style="text-align:center;vertical-align:middle;">0.4314</td>
    <td style="text-align:center;vertical-align:middle;">0.5046</td>
    <td style="text-align:center;vertical-align:middle;">89</td>
    <td style="text-align:center;vertical-align:middle;">0.4917</td>
    <td style="text-align:center;vertical-align:middle;">0.3490</td>
    <td style="text-align:center;vertical-align:middle;">0.4083</td>
  </tr>
  <tr>
    <td rowspan="4">RU</td>
    <td style="text-align:center;vertical-align:middle;">dev.p1.out</td>
    <td style="text-align:center;vertical-align:middle;">335</td>
    <td style="text-align:center;vertical-align:middle;">0.1604</td>
    <td style="text-align:center;vertical-align:middle;">0.7267</td>
    <td style="text-align:center;vertical-align:middle;">0.2627</td>
    <td style="text-align:center;vertical-align:middle;">136</td>
    <td style="text-align:center;vertical-align:middle;">0.0651</td>
    <td style="text-align:center;vertical-align:middle;">0.2950</td>
    <td style="text-align:center;vertical-align:middle;">0.1067</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p2.out</td>
    <td style="text-align:center;vertical-align:middle;">226</td>
    <td style="text-align:center;vertical-align:middle;">0.6706</td>
    <td style="text-align:center;vertical-align:middle;">0.4902</td>
    <td style="text-align:center;vertical-align:middle;">0.5664</td>
    <td style="text-align:center;vertical-align:middle;">156</td>
    <td style="text-align:center;vertical-align:middle;">0.4629</td>
    <td style="text-align:center;vertical-align:middle;">0.3384</td>
    <td style="text-align:center;vertical-align:middle;">0.3910</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p3.out</td>
    <td style="text-align:center;vertical-align:middle;">198</td>
    <td style="text-align:center;vertical-align:middle;">0.2532</td>
    <td style="text-align:center;vertical-align:middle;">0.4295</td>
    <td style="text-align:center;vertical-align:middle;">0.3186</td>
    <td style="text-align:center;vertical-align:middle;">132</td>
    <td style="text-align:center;vertical-align:middle;">0.1688</td>
    <td style="text-align:center;vertical-align:middle;">0.2863</td>
    <td style="text-align:center;vertical-align:middle;">0.2124</td>
  </tr>
  <tr>
    <td style="text-align:center;vertical-align:middle;">dev.p4.out</td>
    <td style="text-align:center;vertical-align:middle;">175</td>
    <td style="text-align:center;vertical-align:middle;">0.4605</td>
    <td style="text-align:center;vertical-align:middle;">0.3796</td>
    <td style="text-align:center;vertical-align:middle;">0.4162</td>
    <td style="text-align:center;vertical-align:middle;">130</td>
    <td style="text-align:center;vertical-align:middle;">0.3421</td>
    <td style="text-align:center;vertical-align:middle;">0.2820</td>
    <td style="text-align:center;vertical-align:middle;">0.3092</td>
  </tr>
</tbody>
</table>

## Acknowledgement

We refer to the following resources when implementing this project:

-   Sung-Hyun, Yang, et al. “Log-Viterbi Algorithm Applied on Second-Order Hidden Markov Model for Human Activity Recognition.” International Journal of Distributed Sensor Networks, Apr. 2018, doi:[10.1177/1550147718772541.](https://journals.sagepub.com/doi/10.1177/1550147718772541)
-   RNN: [pangolulu / rnn-from-scratch](https://github.com/pangolulu/rnn-from-scratch)
