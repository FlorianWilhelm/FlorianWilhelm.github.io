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



## Implementation



## Evaluation


In this blog post
* get more into PyTorch
* Implement a paper
* inspired by Mixture-of-tastes Models for Representing Users with Diverse Interests by Maciej Kula



* g flexible input-dependent
transitions
* easier to recover from mistakes
* The relative magnitude of Whhht−1 to Whxxt will need to be large for the RNN to be
able to use long range dependencies, and the resulting possible hidden state vectors will therefore
be highly correlated across the possible inputs, limiting the width of the tree and making it harder
for the RNN to form distinct hidden representations for different sequences of inputs. However, if
the RNN has flexible input-dependent transition functions, the tree will be able to grow wider more
quickly, giving the RNN the flexibility to represent more probability distributions.



what pytorch does http://pytorch.org/docs/0.3.1/nn.html?highlight=lstm#torch.nn.LSTM


\begin{split}\begin{array}{ll}
i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
c_t = f_t * c_{(t-1)} + i_t * g_t \\
h_t = o_t * \tanh(c_t)
\end{array}\end{split}

\begin{split}\begin{array}{ll}
i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
c_t = f_t * c_{(t-1)} + i_t * g_t \\
h_t = o_t * \tanh(c_t)
\end{array}\end{split}


|dataset        | type  | validation    | test      | learn_rate    | batch_size    |embedding_dim  | l2      | n_iter   |  
| --:           | --:   | --:           | --:       | --:           | --:           | --:           | --:     | --:      |
| Movielens 1m  | LSTM  | 0.1199        | 0.1317    | 1.93e-2       | 208           | 112           | 6.01e-06| 50       |
| Movielens 1m  | mLSTM | 0.1275        | 0.1386    | 1.25e-2       | 240           | 120           | 5.90e-06| 40       |
| Movielens 10m | LSTM  | 0.1090        | 0.1033    | 4.19e-3       | 224           | 120           | 2.43e-07| 50       |
| Movielens 10m | mLSTM | 0.1142        | 0.1115    | 4.50e-3       | 224           | 128           | 1.12e-06| 45       |
| Amazon        | LSTM  | 0.2629        | 0.2642    | 2.85e-3       | 224           | 128           | 2.42e-11| 50       |
| Amazon        | mLSTM | 0.3061        | 0.3123    | 2.48e-3       | 144           | 120           | 4.53e-11| 50       |

For Movielens 10m it's 7.96% more, for Movielens 1m it's 5.30% more and for Amazon it's 18.19% more.


Movielens 10m: LSTM
Best lstm: {'loss': -0.10897761687940868, 'status': 'ok', 'validation_mrr': 0.10897761687940868, 'test_mrr': 0.10331566577703966, 'elapsed': 513.5248389999906, 'hyper': {'batch_size': 224.0, 'embedding_dim': 120.0, 'l2': 2.431720723375239e-07, 'learn_rate': 0.004191673293596244, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'representation': 'lstm', 'type': 'lstm'}}
Best test lstm: {'loss': -0.10737651447383792, 'status': 'ok', 'validation_mrr': 0.10737651447383792, 'test_mrr': 0.10819644872417063, 'elapsed': 512.039592999994, 'hyper': {'batch_size': 224.0, 'embedding_dim': 128.0, 'l2': 8.308174943883086e-07, 'learn_rate': 0.00448018629203927, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'representation': 'lstm', 'type': 'lstm'}}

Movielens 10m: MLSTM with bias
Best mlstm: {'loss': -0.114222527671016, 'status': 'ok', 'validation_mrr': 0.114222527671016, 'test_mrr': 0.1115424354735183, 'elapsed': 2922.5535500000115, 'hyper': {'batch_size': 224.0, 'embedding_dim': 128.0, 'l2': 1.122383064664196e-06, 'learn_rate': 0.004502583196502701, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.11314558305018928, 'status': 'ok', 'validation_mrr': 0.11314558305018928, 'test_mrr': 0.11416640215314312, 'elapsed': 2842.687773999991, 'hyper': {'batch_size': 256.0, 'embedding_dim': 128.0, 'l2': 5.612150600227044e-07, 'learn_rate': 0.00282413979022822, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}

Movielens 10m: MLSTM no bias
Best mlstm: {'loss': -0.11284279641455065, 'status': 'ok', 'validation_mrr': 0.11284279641455065, 'test_mrr': 0.11130102431345008, 'elapsed': 5864.302134000027, 'hyper': {'batch_size': 96.0, 'embedding_dim': 128.0, 'l2': 3.780019195029335e-07, 'learn_rate': 0.004284719311967256, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.11007872916092554, 'status': 'ok', 'validation_mrr': 0.11007872916092554, 'test_mrr': 0.11326304067312118, 'elapsed': 2606.692317000008, 'hyper': {'batch_size': 176.0, 'embedding_dim': 120.0, 'l2': 5.475872081408865e-07, 'learn_rate': 0.0024802226415303096, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}




Movielens 1m: LSTM

Best lstm: {'loss': -0.1199341944558793, 'status': 'ok', 'validation_mrr': 0.1199341944558793, 'test_mrr': 0.13166397144959427, 'elapsed': 37.21245499999986, 'hyper': {'batch_size': 208.0, 'embedding_dim': 112.0, 'l2': 6.008895453807441e-06, 'learn_rate': 0.0192665791725462, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'representation': 'lstm', 'type': 'lstm'}}
Best test lstm: {'loss': -0.11574967165161487, 'status': 'ok', 'validation_mrr': 0.11574967165161487, 'test_mrr': 0.14332325978424815, 'elapsed': 37.19731500000012, 'hyper': {'batch_size': 176.0, 'embedding_dim': 120.0, 'l2': 5.6579678144200235e-06, 'learn_rate': 0.016373741157563775, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'representation': 'lstm', 'type': 'lstm'}}

Movielens 1m: MLSTM with bias
Best mlstm: {'loss': -0.12752956303130814, 'status': 'ok', 'validation_mrr': 0.12752956303130814, 'test_mrr': 0.13863908625141577, 'elapsed': 241.61905800000386, 'hyper': {'batch_size': 240.0, 'embedding_dim': 120.0, 'l2': 5.897712437596623e-06, 'learn_rate': 0.012513516182107765, 'loss': 'adaptive_hinge', 'n_iter': 40.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.10890931693511384, 'status': 'ok', 'validation_mrr': 0.10890931693511384, 'test_mrr': 0.1420496905472083, 'elapsed': 282.38592600000266, 'hyper': {'batch_size': 176.0, 'embedding_dim': 80.0, 'l2': 2.4066253483306436e-06, 'learn_rate': 0.013443370545975792, 'loss': 'adaptive_hinge', 'n_iter': 35.0, 'type': 'mlstm'}}


Amazon dataset: LSTM 

Best lstm: {'loss': -0.2629374199498105, 'status': 'ok', 'validation_mrr': 0.2629374199498105, 'test_mrr': 0.26423511413756995, 'elapsed': 591.0852689999956, 'hyper': {'batch_size': 224.0, 'embedding_dim': 128.0, 'l2': 2.4246884767658603e-11, 'learn_rate': 0.002853417868412002, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'representation': 'lstm', 'type': 'lstm'}}
Best test lstm: {'loss': -0.2588859626563219, 'status': 'ok', 'validation_mrr': 0.2588859626563219, 'test_mrr': 0.2659896721664309, 'elapsed': 444.27522399999725, 'hyper': {'batch_size': 240.0, 'embedding_dim': 120.0, 'l2': 1.555992651335121e-11, 'learn_rate': 0.0025356298009402024, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'representation': 'lstm', 'type': 'lstm'}}

Amazon dataset: MLSTM with bias
Best mlstm: {'loss': -0.3060741355825258, 'status': 'ok', 'validation_mrr': 0.3060741355825258, 'test_mrr': 0.3123092950844201, 'elapsed': 3047.43397199997, 'hyper': {'batch_size': 144.0, 'embedding_dim': 120.0, 'l2': 4.531181846116166e-11, 'learn_rate': 0.002482051873029504, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.3060741355825258, 'status': 'ok', 'validation_mrr': 0.3060741355825258, 'test_mrr': 0.3123092950844201, 'elapsed': 3047.43397199997, 'hyper': {'batch_size': 144.0, 'embedding_dim': 120.0, 'l2': 4.531181846116166e-11, 'learn_rate': 0.002482051873029504, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}

Amazon dataset: MLSTM no bias
Best mlstm: {'loss': -0.29474998682827735, 'status': 'ok', 'validation_mrr': 0.29474998682827735, 'test_mrr': 0.2990891410380816, 'elapsed': 1504.2551740000054, 'hyper': {'batch_size': 176.0, 'embedding_dim': 128.0, 'l2': 5.950730944395493e-10, 'learn_rate': 0.0024877129712099428, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.29474998682827735, 'status': 'ok', 'validation_mrr': 0.29474998682827735, 'test_mrr': 0.2990891410380816, 'elapsed': 1504.2551740000054, 'hyper': {'batch_size': 176.0, 'embedding_dim': 128.0, 'l2': 5.950730944395493e-10, 'learn_rate': 0.0024877129712099428, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}


Movielens 10m: LSTM
Best lstm: {'loss': -0.10897761687940868, 'status': 'ok', 'validation_mrr': 0.10897761687940868, 'test_mrr': 0.10331566577703966, 'elapsed': 513.5248389999906, 'hyper': {'batch_size': 224.0, 'embedding_dim': 120.0, 'l2': 2.431720723375239e-07, 'learn_rate': 0.004191673293596244, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'representation': 'lstm', 'type': 'lstm'}}
Best test lstm: {'loss': -0.10737651447383792, 'status': 'ok', 'validation_mrr': 0.10737651447383792, 'test_mrr': 0.10819644872417063, 'elapsed': 512.039592999994, 'hyper': {'batch_size': 224.0, 'embedding_dim': 128.0, 'l2': 8.308174943883086e-07, 'learn_rate': 0.00448018629203927, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'representation': 'lstm', 'type': 'lstm'}}

Movielens 10m: MLSTM with bias
Best mlstm: {'loss': -0.114222527671016, 'status': 'ok', 'validation_mrr': 0.114222527671016, 'test_mrr': 0.1115424354735183, 'elapsed': 2922.5535500000115, 'hyper': {'batch_size': 224.0, 'embedding_dim': 128.0, 'l2': 1.122383064664196e-06, 'learn_rate': 0.004502583196502701, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.11314558305018928, 'status': 'ok', 'validation_mrr': 0.11314558305018928, 'test_mrr': 0.11416640215314312, 'elapsed': 2842.687773999991, 'hyper': {'batch_size': 256.0, 'embedding_dim': 128.0, 'l2': 5.612150600227044e-07, 'learn_rate': 0.00282413979022822, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}

Movielens 10m: MLSTM no bias
Best mlstm: {'loss': -0.11284279641455065, 'status': 'ok', 'validation_mrr': 0.11284279641455065, 'test_mrr': 0.11130102431345008, 'elapsed': 5864.302134000027, 'hyper': {'batch_size': 96.0, 'embedding_dim': 128.0, 'l2': 3.780019195029335e-07, 'learn_rate': 0.004284719311967256, 'loss': 'adaptive_hinge', 'n_iter': 45.0, 'type': 'mlstm'}}
Best test mlstm: {'loss': -0.11007872916092554, 'status': 'ok', 'validation_mrr': 0.11007872916092554, 'test_mrr': 0.11326304067312118, 'elapsed': 2606.692317000008, 'hyper': {'batch_size': 176.0, 'embedding_dim': 120.0, 'l2': 5.475872081408865e-07, 'learn_rate': 0.0024802226415303096, 'loss': 'adaptive_hinge', 'n_iter': 50.0, 'type': 'mlstm'}}




[SMART criteria]: https://en.wikipedia.org/wiki/SMART_criteria
[PyTorch]: https://pytorch.org/
[Spotlight]: https://github.com/maciejkula/spotlight
[Github repo]: https://github.com/FlorianWilhelm/mlstm4reco
[Understanding LSTM Networks]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[The Unreasonable Effectiveness of Recurrent Neural Networks]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[Multiplicative LSTM for sequence modelling]: https://arxiv.org/abs/1609.07959
[Mixture-of-tastes Models for Representing Users with Diverse Interests]: https://arxiv.org/abs/1711.08379
[mixture repo]: https://github.com/maciejkula/mixture
