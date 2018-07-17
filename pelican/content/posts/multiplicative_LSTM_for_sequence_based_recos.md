---
title: Multiplicative LSTM for sequence-based Recommendations
date: 2017-04-15 18:00
modified: 2017-06-17 18:00
category: post
tags: python, data science, deep learning
authors: Florian Wilhelm
status: draft
---

## Motivation

Recommender Systems support the decision making processes of customers with personalized suggestions. 
They are widely used and influence the daily life of almost everyone in different domains like e-commerce, 
social media, or entertainment. Quite often the dimension of times plays a dominant role in the generation
of a relevant recommendation. Traditional user-item recommenders neglect the sequential nature of this dimension. 
This means that many of these traditional recommenders find for each user a latent representation based on the user's
historical item interactions without any notion of recency and sequence of interactions. To also incorporate 
this kind of contextual information about interactions, sequence-based recommenders were developed and quite a few of them are based on Recurrent Neural Networks (RNNs).
 
Whenever I want to dig deeper into a topic like sequence-based recommenders I follow a few simple steps:
First of all, to learn something I directly need to apply it otherwise learning things doesnt work for me. In order to apply something I need a challenge and a small goal that keeps me motivated on the journey. Following the [SMART citeria] a goal needs to be measurable and thus a typical outcome for me is a blog post like the one you are just reading. Another good thing about a blog post is that the fact that no one wants to publish something completely crappy, so there is an intrinsic quality assurance attached to the outcome which I hope works ;-) 

Actually, this blog post is the outcome of several things I wanted to familiarize myself more and try out:
1) [PyTorch], since this framework is used in a large fraction of publications about deep learning,
2) [Spotlight], since this library gives you a sophisticated structure to play around with new ideas for recommender systems and already has a lot of functionality implemented,
3) applying a paper about [Multiplicative LSTM for sequence modelling] to recommender systems and see how that performs compared to traditional LSTMs.
 
Since Spotlight is based on PyTorch and multiplicative LSTMs (mLSTMs) are not yet implemented in PyTorch the task of evaluating mLSTMs vs. LSTMs inherently addresses all those points outlined above. So let's get going!

## Theory

Long short-term memory architecture (LSTM)s are maybe the most common incarnations of RNNs since they don't adhere 
to the [vanishing gradient problem] and thus are able to capture long-term relationships in a sequence. Find a great
explanations of LSTMs in Colah's post [Understanding LSTM Networks] and more general about the power of RNNs in the 
article [The Unreasonable Effectiveness of Recurrent Neural Networks]. 
More recently, also Gated Recurrent Units (GRUs) which have a simplified structure compared to LSTMs are also used 
in sequential prediction tasks with similar results. Spotlight provides a sequential recommender based on LSTMs and 
the quite renowned [GRU4Rec] model uses GRUs but in general it's not possible to state that one always outperforms the other.

So given these ingredients, how do we now construct a sequential recommender? Let's assume on every timestep 
$t\in\{1,\ldots,T\}$ a user has interacted with an item $i_t$. The basic idea is now to feed these interactions into
 an LSTM up to the time $t$ in order to get a representation of the user's preferences $h_t$ and use these to state
 if the user might like or dislike the next item $i_{t+1}$. The other elements of the model are what one would also do
 in a non-sequential recommender like one-hot encoding of the items and embedding them in a dense vector representation $e_{i_t}$
 which is then feed into the LSTM. We can then just use the output $h_t$ of the LSTM and calculate the inner product ($\bigotimes$) 
 with the embedding $e_{i_{t+1}}$ plus an item bias for varying item popularity to retrieve an output $p_{t+1}$. 
 This output along with others is then used to calculate the actual loss depending on our sample strategy and loss function. 
 Figure 1 illustrates our sequential recommender model and this is what's actually happening inside Spotlight's 
 sequential recommender with an LSTM representation. If you raise your eyebrow due to the usage of a inner product
 then be aware that [low-rank approximations] have been and still are one of the most successful building blocks
 of a recommender system. An alternative would be to replace the inner product with a deep feed forward network but
 in most likely this would also only just learn to perform an approximation to a inner product. A recent paper
 [Latent Cross: Making Use of Context in Recurrent Recommender Systems] by Google also emphasizes the power of learning
 low-rank relations with the help of inner products.
 
<figure>
<p align="center">
<img class="noZoom" src="/images/mLSTM.png" alt="mLSTM">
<figcaption><strong>Figure 1:</strong> At timestep $t$ the item $i_t$ is embedded and fed into an LSTM together with
 cell state $C_{t-1}$ and $h_{t-1}$ of the last timestep which yields a new presentation $h_t$. The inner product of 
 $h_t$ with the embedding of the potential next item $e_{i_{t+1}}$ yields a value corresponding to how likely the user
 would interact with $i_{t+1}$.</figcaption>
</p>
</figure>
        
* Structure of sequential recommender as in Spotlight
* picture? 

\begin{split}\begin{array}{ll}
i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
c_t = f_t * c_{(t-1)} + i_t * g_t \\
h_t = o_t * \tanh(c_t)
\end{array}\end{split}

* g flexible input-dependent
transitions
* easier to recover from mistakes
* The relative magnitude of Whhht−1 to Whxxt will need to be large for the RNN to be
able to use long range dependencies, and the resulting possible hidden state vectors will therefore
be highly correlated across the possible inputs, limiting the width of the tree and making it harder
for the RNN to form distinct hidden representations for different sequences of inputs. However, if
the RNN has flexible input-dependent transition functions, the tree will be able to grow wider more
quickly, giving the RNN the flexibility to represent more probability distributions.


\begin{split}\begin{array}{ll}
m_t = (W_{im} x_t + b_{im}) \hadamard{} ( W_{hm} h_{(t-1)} + b_{hm}) \\
i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{mi} m_t + b_{mi}) \\
f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{mf} m_t + b_{mf}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{mc} m_t + b_{mg}) \\
o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{mo} m_t + b_{mo}) \\
c_t = f_t * c_{(t-1)} + i_t * g_t \\
h_t = o_t * \tanh(c_t)
\end{array}\end{split}

## Implementation

* inspired by Mixture-of-tastes Models for Representing Users with Diverse Interests by Maciej Kula
* Die Variablennamen noch anpassen wie oben

```python
import math

import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import functional as F


class mLSTM(RNNBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=input_size, hidden_size=input_size,
                 num_layers=1, bias=bias, batch_first=True,
                 dropout=0, bidirectional=False)

        w_mx = torch.Tensor(hidden_size, input_size)
        w_mh = torch.Tensor(hidden_size, hidden_size)
        b_mx = torch.Tensor(hidden_size)
        b_mh = torch.Tensor(hidden_size)
        self.w_mx = Parameter(w_mx)
        self.b_mx = Parameter(b_mx)
        self.w_mh = Parameter(w_mh)
        self.b_mh = Parameter(b_mh)

        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        n_batch, n_seq, n_feat = input.size()

        assert hx is not None

        hx, cx = hx
        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq):
            mx = F.linear(input[:, seq, :], self.w_mx, self.b_mx) * F.linear(hx, self.w_mh, self.b_mh)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        return torch.cat(steps, dim=1)
```


## Evaluation

* Hyperparameter Search
"""Umschreiben
﻿Our results are robust to hyperparameter optimization. Figure 1 plots the maximum test MRR achieved by each algorithm as a func- tion of the number of elapsed hyperparameter search iterations. Both baseline and mixture models benefit from hyperparameter tuning. All algorithms converge to their optimum performance rela- tively quickly, suggesting a degree of robustness to hyperparameter choices. Mixture-LSTM and Embedding Mixture models quickly outperform their baseline counterparts, and maintain a stable per- formance lead thereafter (with the exception of the factorization Movielens experiments). This lends support to our belief that the mixture models’ superior accuracy reflects their greater capacity to model the recommendation problem well, rather than being an artifact of the experimental procedure or researcher bias.
5.3
"""

* Wieviele Runs wurden gemacht, muessten 200 gewesen sein ./run.py -m mlstm -n 200 10m


|dataset        | type  | validation    | test      | learn_rate    | batch_size    |embedding_dim  | l2      | n_iter   |  
| --:           | --:   | --:           | --:       | --:           | --:           | --:           | --:     | --:      |
| Movielens 1m  | LSTM  | 0.1199        | 0.1317    | 1.93e-2       | 208           | 112           | 6.01e-06| 50       |
| Movielens 1m  | mLSTM | 0.1275        | 0.1386    | 1.25e-2       | 240           | 120           | 5.90e-06| 40       |
| Movielens 10m | LSTM  | 0.1090        | 0.1033    | 4.19e-3       | 224           | 120           | 2.43e-07| 50       |
| Movielens 10m | mLSTM | 0.1142        | 0.1115    | 4.50e-3       | 224           | 128           | 1.12e-06| 45       |
| Amazon        | LSTM  | 0.2629        | 0.2642    | 2.85e-3       | 224           | 128           | 2.42e-11| 50       |
| Amazon        | mLSTM | 0.3061        | 0.3123    | 2.48e-3       | 144           | 120           | 4.53e-11| 50       |

Comparing the test performance For Movielens 10m it's 7.96% more, for Movielens 1m it's 5.30% more and for Amazon it's 18.19% more.

## Conclusion

[low-rank approximations]: https://en.wikipedia.org/wiki/Low-rank_approximation
[GRU4Rec]: https://github.com/hidasib/GRU4Rec
[vanishing gradient problem]: https://en.wikipedia.org/wiki/Vanishing_gradient_problem
[HyperOpt]: http://hyperopt.github.io/hyperopt/
[LSTM implementation]: http://pytorch.org/docs/0.3.1/nn.html?highlight=lstm#torch.nn.LSTM
[SMART criteria]: https://en.wikipedia.org/wiki/SMART_criteria
[PyTorch]: https://pytorch.org/
[Spotlight]: https://github.com/maciejkula/spotlight
[Github repo]: https://github.com/FlorianWilhelm/mlstm4reco
[Understanding LSTM Networks]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[The Unreasonable Effectiveness of Recurrent Neural Networks]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[Multiplicative LSTM for sequence modelling]: https://arxiv.org/abs/1609.07959
[Mixture-of-tastes Models for Representing Users with Diverse Interests]: https://arxiv.org/abs/1711.08379
[mixture repo]: https://github.com/maciejkula/mixture
