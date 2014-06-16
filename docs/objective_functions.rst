Objective Functions
=======

**Contents:**

    #. `What is an objective function?`_
    #. `Properties of an objective function`_
    #. `Parameters`_
    #. `Phi Objective Functions`_

What is an objective function?
-----

The objective function is the function *f* that we are trying to minimize/maximize over some set of `Parameters`_.

.. math::
    \begin{eqnarray*}
        \underset{\vec{x} \in \mathbb{R}^{d}}{\mathrm{argmax}} \ f (\vec{x}) \\
        f : \mathbb{R}^{d} \rightarrow \mathbb{R}
    \end{eqnarray*}

The objective function is considered a **black box** (http://en.wikipedia.org/wiki/Black_box) function. We require no internal knowledge or access to the function. In fact it can potentially be non-convex, non-differentiable or non-continuous. Furthermore, we assume that evaluating the function, *f*, is **expensive** and we need to find the best set of parameters/inputs :math:`\vec{x}` with as few function evaluations as possible.

The input to MOE is some set of historical parameters sampled :math:`\{\vec{x}_{1}, \ldots, \vec{x}_{n}\}` and their associated function valuations :math:`\{f(\vec{x}_{1}), \ldots, f(\vec{x}_{n})\}`. The evaluations may require A/B testing, an map reduce job or some other expensive or time consuming process. MOE will never need to evaluate *f*, it only takes the outputs you provide and suggests new optimal inputs to test/evaluate.

Using this information MOE builds a model the function space *f* is drawn from (using a Gaussian Process (GP) :doc:`gpp_covariance`) over values of :math:`\vec{x} \in \mathbb{R}^{d}` and maximizes the Expected Improvement (EI, :doc:`gpp_expected_improvement_demo`) of sampling different potential values :math:`\vec{x}` in this space, without actually evaluating *f*. MOE then outputs the *q* value(s) in :math:`\mathbb{R}^{d \times q}` that have the highest EI to be sampled next by the user.

Properties of an objective function
----

* An objective function is any real valued function *f* defined over the input parameters.
* MOE works best if the objective function has 0 mean, although this in not required depending on the covariance kernel (:doc:`gpp_covariance`)
* MOE minimizes an objective function by default. To maximize just multiply all function evaluations by -1.
* Objective functions can be composed of many different metrics see `The Metrics`_
* See below for examples of different objective functions and a class of objective functions that work well with MOE

See :doc:`examples`

Parameters
----

Parameters are any real valued constants, config values, thresholds, hyperparameters or *magic numbers* in your codebase or inputs to your system/model/experiment.

**Examples:**

* Weights of arbitrary scoring functions
* Thresholds for when advertisments have certain attributes
* Hyperparameters for features of scoring functions
* Hyperparameters of ML algorithms
* Thresholds for when/how to show ads or other content
* Constants large/expensive training algorithms

See :doc:`examples`

.. _Phi Objective Functions:
:math:`\Phi` Objective Functions
-----

In this writeup we define a class of objective functions, :math:`\Phi` objective functions, that MOE can use, their components and some examples.

Properties of the proposed objective function :math:`\Phi`:
....

#. The objective function, :math:`\Phi`, is the quantity that MOE seeks to maximize by manipulating the values of parameters in some space.
#. The objective function :math:`\Phi` is well defined for every set of parameters :math:`C_{i} \in \vec{C}` for which the experiment is run.
#. The objective function is defined relative to the status quo set of parameters; :math:`C_{S} \in \vec{C}`. By construction the objective function has value 0 for the status quo. Parameters that outperform the status quo will have positive values, those that perform worse will have negative values.
#. The objective function is designed by construction to be scaleless, unitless and have meaningful and intuitive representation in linear and log space. For examples see below.
#. The objective function is defined over parameters :math:`\vec{C}` and objective function parameters :math:`\mathbb{P}` which encompasses all relevant metrics, weights and thresholds.
#. :math:`\Phi : C_{i} \in \vec{C}, \mathbb{P} \rightarrow [-1, \infty)`. Lower values imply the parameters are worse than the status quo, higher values imply they are better. A value of 0 implies they are indistinguishable from the status quo.

It is worth noting that MOE will attempt to increase the objective function blindly. MOE is a **black box**, global optimization experimental design framework. If there are easy ways for it to exploit the code by manipulating parameters it will probably find it. Defining the objective function well is the most important part of running a MOE experiment.

Classes of :math:`\Phi` objective functions
....

The general class of objective functions :math:`\Phi` are products of weighted, thresholded relative compositions of metrics :math:`M` defined for each set of parameters :math:`C_{i}` of an experiment as follows:

.. math::
    \begin{equation}
        \Phi\left(C_{i}, \mathbb{P} = \left(\vec{M}, \vec{\omega_{M}}, \vec{\tau_{M}}\right)\right) = \prod_{\{M, \omega, \tau \} \in \mathbb{P}} \left( \left(\frac{M(C_{i})}{M(C_{S})}\right)^{\omega_{M}} \mathcal{H} \left(\text{sgn}\left(\omega_{M}\right)\left(\frac{M(C_{i})}{M(C_{S})} - \tau_{M}\right)\right) \right) - 1.
    \end{equation}

where `M(C_{i})` is the value of the metric for a set of parameters :math:`C_{i}`, `M(C_{S})` is the value of the metric for the status quo parameters :math:`C_{S} \in \vec{C}` and :math:`\mathcal{H}(x)` is the Heaviside function defined as

.. math::
    \begin{equation}
        \mathcal{H}(x) = \left\{ 1 \text{ if } x \geq 0; \ 0 \text{ otherwise.} \right.
    \end{equation}

The parameters :math:`\omega_{M}, \tau_{M}` represent the weight and threshold of the metric respectively. Note that we subtract 1 from the end of the objective function so that the status quo parameters :math:`C_{S}` will result in a value of 0,

.. math::
    \begin{equation}
        \Phi\left(C_{S}, \mathbb{P}\right) = 0.
    \end{equation}

One can break the objective function for each metric into two distinct parts. First, the relitave gain over the status quo,

.. math::
    \begin{equation}
        \left(\frac{M(C_{i})}{M(C_{S})}\right)^{\omega_{M}}
    \end{equation}

The fraction will be larger than 1 if the parameters have a larger metric :math:`M` than the status quo, otherwise it will be less than one. Note that for the status quo itself this fraction will be exactly 1. The magnitude and sign of the weight determine how important this metric :math:`M` is in the overall objective function. The weight will be discussed in a later section.

The second component of the objective function is the threshold,

.. math::
    \begin{equation}
        \mathcal{H} \left(\text{sgn}\left(\omega_{M}\right)\left(\frac{M(C_{i})}{M(C_{S})} - \tau_{M}\right)\right)
    \end{equation}

If the relative gain (or loss if :math:`\omega_{M} < 0`) of the metric :math:`M` for the set of parameters :math:`C_{i}` is below the threshold :math:`\tau_{M}` this component will have value 0. Note that this will cancel all gains in all other metrics and give the objective function its lowest possible value. One can also replace the Heavyside function with a logistic function, or a probability of violating the constraints.

The Metrics
....

The metric is any quantity defined over sets of parameters.

Possible examples include:

#. Click Through Rate (CTR)
#. Sell Through Rate (STR, the number of ads shown per page)
#. Revenue Per Opportunity (RPO)
#. Average/median/95th delivery timings
#. Any happiness metric defined on the reals
#. Number of reviews written
#. Photo contributions/views in a session
#. User engagement
#. Conversions
#. Any metric about the user, session or page that can be defined on the reals
#. :math:`M : C_{i} \rightarrow \Re \ \ \ \forall C_{i} \in \vec{C}`

The Weight
....

The weight :math:`\omega_{M}` of a function represents how much we want the ratio of that metric to effect the overall objective function.

.. math::
    \begin{equation}
        \omega_{M} \in [0, \infty)
    \end{equation}

A weight :math:`\omega_{M} = 0` corresponds to no effect. The objective function will become just the Heaviside function.

Small weights :math:`0 < \omega_{M} < 1` will pull ratios lower and higher than 1 closer to 1. Large weights :math:`1 < \omega_{M}` will have the opposite effect.

The Threshold
....

The threshold represents how far we are willing to allow the specific metric to drop before we consider there to be no utility. For example if we wish to keep Sell Through Rate (STR) at at least 85\% of its current value we would set :math:`\tau_{M} = 0.85` for that metric. Parameters with a ratio of STR lower than this threshold will have an objective function equal to 0.

.. math::
    \begin{equation}
        \tau_{M} \in [0, 1]
    \end{equation}

.. Note:

    It is also possible to use other thresholding functions like the logistic function (smoother) or some probability of violating the constraints.

Log Space
....

We note that the objective functions decompose into log space readily, which is helpful because maximizing the original objective function is equivalent to maximizing it in log space (because it is a monotonic transform),

.. math::
    \begin{equation}
        \log \Phi(C_{i}, M, \omega_{M}, \tau_{M}) = \omega_{M} \log \left(\frac{M(C_{i})}{M(C_{S})}\right) + \log \mathcal{H} \left(\text{sgn}\left(\omega_{M}\right)\left(\frac{M(C_{i})}{M(C_{S})} - \tau_{M}\right)\right).
    \end{equation}

The log of the Heaviside function now returns a value of 0 or :math:`-\infty` and can be calculated separately for numerical reasons.

.. Note::

    The range now becomes,

    .. math::

        \begin{equation}
            \log \Phi : C_{i} \in \vec{C} \rightarrow (-\infty, \infty)
        \end{equation}

Example of Objective Functions
----

Below are examples of different intuitive ideals and the resulting objective functions.

Click Through Rate (CTR) Only
....

Let's say we only care about CTR, and we want to make sure no parameters allow it to fall more than 95\%. We define :math:`\mathbb{P}` as;

.. math::
    \begin{eqnarray*}
        \vec{M} & = & \left\{\text{CTR}\right\} \\
        \vec{\omega} & = & \left\{1.0\right\} \\
        \vec{\tau} & = & \left\{0.95\right\}
    \end{eqnarray*}

which results in

.. math::
    \begin{equation}
        \Phi(C_{i}, \mathbb{P}) = \left(\frac{CTR(C_{i})}{CTR(C_{S})}\right)\mathcal{H}\left(\frac{CTR(C_{i})}{CTR(C_{S})} - 0.95\right) - 1
    \end{equation}

Clicks Per Opportunity (CPO)
....

Let's say we only care about Clicks Per Opportunity (CPO) which is the product of CTR and STR. We define :math:`\mathbb{P}` as;

.. math::
    \begin{eqnarray*}
        \vec{M} & = & \left\{\text{CTR}, \text{STR}\right\} \\
        \vec{\omega} & = & \left\{1, 1\right\} \\
        \vec{\tau} & = & \left\{0, 0\right\}
    \end{eqnarray*}

which results in

.. math::
    \begin{equation}
        \Phi(C_{i}, \mathbb{P}) = \left(\frac{CTR(C_{i})}{CTR(C_{S})}\right)\left(\frac{STR(C_{i})}{STR(C_{S})}\right) - 1
    \end{equation}

Note that :math:`\tau = 0` for a metric effectively removes the Heavyside function from the objective function for that metric.

Mixture Example
....

Let's say we mostly care about CTR, but wouldn't mind if STR also went up. We don't want to make the site any more than 10\% slower though so we introduce a metric MDT, which will be the Mean Delivery Time (MDT) in milliseconds for the given set of parameters. We define :math:`\mathbb{P}` as;

.. math::
    \begin{eqnarray*}
        \vec{M} & = & \left\{\text{CTR}, \text{STR}, \text{MDT}\right\} \\
        \vec{\omega} & = & \left\{1, \frac{1}{5}, 0\right\} \\
        \vec{\tau} & = & \left\{0, 0, \frac{9}{10}\right\}
    \end{eqnarray*}

which results in

.. math::
    \begin{equation}
        \Phi(C_{i}, \mathbb{P}) = \left(\frac{CTR(C_{i})}{CTR(C_{S})}\right)\left(\frac{STR(C_{i})}{STR(C_{S})}\right)^{\frac{1}{5}}\mathcal{H}\left(\frac{MDT(C_{i})}{MDT(C_{S})} - \frac{9}{10}\right) - 1
    \end{equation}
