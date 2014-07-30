Multi-Armed Bandits
===================

**Contents:**

    #. `What is the multi-armed bandit problem?`_
    #. `Applications`_
    #. `Policies`_
    #. `Pointers`_

What is the multi-armed bandit problem?
------------------------------

Imagine you are in front of *K* slot machines.
Each machine has a possibly different, unknown payout rate. You are allowed to pull the arm *T* times.
The problem is how to allocate the number of pulls to each machine to maximize your payout.

**The mathematical setup is as follows:**

* *K* random variables :math:`\{A_1, ..., A_K\}` with unknown payout distributions
* At each time interval :math:`t=1, ..., T` we observe a single random variable :math:`a_t`
* The payout is :math:`r_{t,a_t}` (note: :math:`r_{t,a_t}` for :math:`a_q \neq a_t` is *unobserved*)
* Want to maximize

.. math::

        \mathbb{E} [ \sum\limits_{i=1}^T r_{t,a_t} ] = expected\;total\;reward

**Tradeoffs**

The tradeoff of each pull is gaining knowledge about the payout rates vs. getting the biggest payout with current knowledge
(Exploration vs. Exploration). We can define regret as follows:

.. math::

        Regret = \mathbb{E} [ \sum\limits_{i=1}^T r_{t,a_t}^{\star} ] - \mathbb{E} [ \sum\limits_{i=1}^T r_{t,a_t} ] = best\;possible\;reward - actually\;obtained\;reward

.. note::

        Any policy that explores at all will have non-zero regret.

Applications
-----------------------------------

There are many applications that map well onto this problem.
For example, we can model a Click Through Rate (CTR) problem as
a multi-armed bandit instance.
In this case, each arm is an ad or search result, each click is a success,
and our objective is to maximize clicks.

Another application is experiments (A/B testing)
where we would like to find the best solutions as fast as possible
and limit how often bad solutions are chosen.

Policies
-----------------------------------

There are many different policies for this problem:

We have implemented the following policies in our package:

* :mod:`~moe.bandit.epsilon_greedy.EpsilonGreedy`

Other policies include:

* Weighted random choice
* `Epsilon-first`_
* `Epsilon-decreasing`_ \*
* `UCB-exp (Upper Confidence Bound)`_ \*
* `UCB-tuned`_ \*
* `BLA (Bayesian Learning Automaton)`_ \*
* `SoftMax`_ \*

\* Regret bounded as :math:`t \rightarrow \infty`

.. _Epsilon-first: http://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
.. _Epsilon-decreasing: http://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
.. _UCB-exp (Upper Confidence Bound): http://moodle.technion.ac.il/pluginfile.php/192340/mod_resource/content/0/UCB.pdf
.. _UCB-tuned: http://moodle.technion.ac.il/pluginfile.php/192340/mod_resource/content/0/UCB.pdf
.. _BLA (Bayesian Learning Automaton): http://dl.acm.org/citation.cfm?id=1491370
.. _SoftMax: http://arxiv.org/pdf/1402.6028v1.pdf

Pointers
-----------------------------------

You can learn more about multi-armed bandits at: http://www.youtube.com/watch?v=qAN6iyYPbEE

The slides for the talk is available at: http://slidesha.re/1zOrOJy
