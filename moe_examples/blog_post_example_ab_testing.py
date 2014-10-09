# -*- coding: utf-8 -*-
"""An example of using MOE to optimize a simulated A/B testing framework.

Blog post: http://engineeringblog.yelp.com/2014/10/using-moe-the-metric-optimization-engine-to-optimize-an-ab-testing-experiment-framework.html

"""
import copy

import numpy

import matplotlib
import matplotlib.pylab as plt

from moe.bandit.constant import EPSILON_SUBTYPE_GREEDY
from moe.bandit.data_containers import HistoricalData as BanditHistoricalData
from moe.bandit.data_containers import SampleArm
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.bandit_simple_endpoint import bandit as bandit_endpoint
from moe.easy_interface.simple_endpoint import gp_next_points, gp_mean_var
from moe.optimal_learning.python.constant import GRADIENT_DESCENT_OPTIMIZER
from moe.views.constant import GP_NEXT_POINTS_KRIGING_ROUTE_NAME, BANDIT_EPSILON_ROUTE_NAME

matplotlib.use('Agg')  # Repress screen output from matplotlib

#: Initial parameter value
STATUS_QUO_PARAMETER = numpy.array([0.2])
#: Local optima of CTR function
LOCAL_OPTIMAL_PARAMETER = numpy.array([0.14289063])
#: Global optima of CTR function
GLOBAL_OPTIMAL_PARAMETER = numpy.array([0.71428711])

#: domain over which our parameter ``x`` can vary in optimizing ``CTR(x)``
EXPERIMENT_DOMAIN = [[0.0, 1.0]]

#: Simulated traffic volume
TRAFFIC_PER_DAY = 1000000  # 1 Million

#: Number of parameters to simulate each day
NUMBER_OF_ACTIVE_COHORTS = 3

#: Number of times to have MOE update cohorts
EXPERIMENT_ITERATIONS = 5

#: Set as a constant for clarity of example, see mod:`moe_examples.combined_example` for hyperparameter updating
COVARIANCE_INFO = {
        'hyperparameters': [0.3, 0.1],
        }

#: Our bandit policy :mod:`moe.bandit.epsilon.epsilon_greedy` requires an epsilon parameter
#: This is the fraction of allocations that will only explore, the rest will only exploit
EPSILON_BANDIT_PARAMETER = 0.15


def true_click_through_rate(x):
    r"""Return the *true* underlying Click Through Rate (CTR) with respect to the parameter ``x``.

    Higher values are better. There is an local optima at ``LOCAL_OPTIMAL_PARAMETER`` (0.14289063) and a global optima at ``GLOBAL_OPTIMAL_PARAMETER`` (0.71428711).

    .. Note:

        The underlying function does not need to be convex, differentiable, or even continuous for MOE to be able to optimize it.

    .. math::

        \texttt{CTR}(x) = \begin{cases}
                \frac{1}{100}\left( \sin\left( \frac{7\pi}{2} x\right) + 2.5\right) & x \leq 0.6 \\
                \frac{1}{100}\left( \sin\left( \frac{7\pi}{2} x\right) + 1.1\right) & x > 0.6
            \end{cases}

    :param x: The parameter that we are optimizing
    :type x: array of float64 with shape (1, )
    :return: CTR evaluated at ``x``
    :rtype: array of float64 with shape (1, )

    """
    if x[0] > 0.6:
        return (numpy.sin(x[0] * 3.5 * numpy.pi) + 2.5) / 100.0
    else:
        return (numpy.sin(x[0] * 3.5 * numpy.pi) + 1.1) / 100.0


def bernoulli_mean_and_var(observed_clicks, samples):
    """Return the mean and variance of a Bernoulli Distribution with a given number of successes (clicks) and trials (impressions).

    :param observed_clicks: number of successes aka clicks
    :type observed_clicks: int >= 0
    :param samples: number of trials aka impressions
    :type sampels: int >= 0
    :return: (mean, variance) of the Bernoulli Distribution producing the observed clicks/samples
    :rtype: tuple

    """
    if samples > 0:
        observed_success_rate = observed_clicks / float(samples)
    else:
        observed_success_rate = 0.0

    # We use the fact that the Beta distribution is the cojugate prior to the binomial distribution
    # http://en.wikipedia.org/wiki/Beta_distribution
    alpha = observed_clicks + 1
    beta = samples - observed_clicks + 1
    observed_variance = (alpha * beta) / float((alpha + beta) * (alpha + beta) * (alpha + beta + 1))

    return observed_success_rate, observed_variance


def objective_function(observed_sample_arm, observed_status_quo_sample_arm):
    r"""The observed objective function MOE is attempting to maximize; returns the mean (relative to status quo) and variance.

    Lower values are better (minimization). Our objective is relative to the CTR of the initial
    parameter value, ``STATUS_QUO_PARAMETER``.

    .. math::

        \Phi(\vec{x}) = \frac{\mathtt{CTR}(\vec{x})}{\mathtt{CTR}(\vec{SQ})}

    :param observed_sample_arm: a bandit arm corresponding to the cohort
      currently behing evaluated; i.e., the cohort whose objective we want
    :type observed_sample_arm: :class:`moe.bandit.data_containers.SampleArm`
    :param observed_status_quo_sample_arm: the bandit arm containing the data for the
      status quo cohort. The objective is defined relative to status quo.
    :type observed_status_quo_sample_arm: :class:`moe.bandit.data_containers.SampleArm`
    :return: (relative CTR, variance) *observed* mean and variance of the objective function
    :rtype: tuple

    """
    sample_ctr, sample_ctr_var = bernoulli_mean_and_var(
            observed_sample_arm.win,
            observed_sample_arm.total,
            )
    status_quo_ctr, status_quo_ctr_var = bernoulli_mean_and_var(
            observed_status_quo_sample_arm.win,
            observed_status_quo_sample_arm.total,
            )

    # Now compute the mean and variance of a quotient of random variables.
    # Formulas come from 1st order taylor expansion.
    # See: http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
    # Subtract 1.0 from the mean so that it is centered at 0.0; no effect on variance
    mean_ctr_ratio = sample_ctr / status_quo_ctr - 1.0
    # Note: We take an upper bound of the variance (by ignoring the correlation between the two random variables)
    variance_ctr_ratio = sample_ctr_var / status_quo_ctr ** 2 + status_quo_ctr_var * sample_ctr ** 2 / status_quo_ctr ** 4
    return mean_ctr_ratio, variance_ctr_ratio


def find_new_points_to_sample(experiment, num_points=1, verbose=False):
    """Find the optimal next point(s) to sample using expected improvement (via MOE).

    :param experiment: an Experiment object containing the historical data and metadata MOE
      needs to optimize
    :type experiment: :class:`moe.easy_interface.experiment.Experiment`
    :param num_points: number of new points (experiments) that we want MOE to suggest
    :type num_points: int >= 1
    :param verbose: whether to print status messages to stdout
    :type verbose: bool
    :return: the next point(s) to sample
    :rtype: list of length ``num_points`` of coordinates (list of length ``dim``)

    """
    if verbose:
        print "Getting {0} new suggested point(s) to sample from MOE...".format(num_points)

    # Query MOE for the next points to sample
    next_points_to_sample = gp_next_points(
            experiment,
            method_route_name=GP_NEXT_POINTS_KRIGING_ROUTE_NAME,
            covariance_info=COVARIANCE_INFO,
            num_to_sample=num_points,
            optimizer_info={
                'optimizer_type': GRADIENT_DESCENT_OPTIMIZER,
                },
            )

    if verbose:
        print "Optimal points to sample next: {0}".format(next_points_to_sample)

    return next_points_to_sample


def get_allocations(active_arms, sample_arms, verbose=False):
    """Return the allocation for each of ``active_arms`` using the epsilon greedy multi-armed bandit strategy.

    :param active_arms: list of coordinate-tuples corresponding to arms/cohorts currently being sampled
    :type active_arms: list of tuple
    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :param verbose: whether to print status messages to stdout
    :type verbose: bool
    :return: traffic allocations for the ``active_arms``, one for each tuple in ``active_arms``
    :rtype: dict

    """
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        # json wants all arm names to be strings
        # so convert our 1D point coordinate tuple to str
        active_sample_arms[str(active_arm[0])] = sample_arms[active_arm]

    # Bundle up the arm information
    bandit_data = BanditHistoricalData(
            sample_arms=active_sample_arms,
            )

    # Query the MOE bandit endpoint for optimal traffic allocation
    bandit_allocation = bandit_endpoint(
            bandit_data,
            type=BANDIT_EPSILON_ROUTE_NAME,
            subtype=EPSILON_SUBTYPE_GREEDY,
            hyperparameter_info={
                'epsilon': EPSILON_BANDIT_PARAMETER,
                },
            )

    arm_allocations = {}
    for arm_name_as_string, allocation in bandit_allocation.iteritems():
        # json produced keys as str
        # convert str back to 1D coordinate-tuple
        arm_allocations[tuple([float(arm_name_as_string)])] = allocation

    if verbose:
        print "Optimal arm allocations: {0}".format(arm_allocations)

    return arm_allocations


def prune_arms(active_arms, sample_arms, verbose=False):
    """Remove all arms from ``active_arms`` that have an allocation less than two standard deviations below the current best arm.

    :param active_arms: list of coordinate-tuples corresponding to arms/cohorts currently being sampled
    :type active_arms: list of tuple
    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :param verbose: whether to print status messages to stdout
    :type verbose: bool
    :return: list of coordinate-tuples that are the *well-performing* members of ``active_arms``
      length is at least 1 and at most ``len(active_arms)``
    :rtype: list of tuple

    """
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        active_sample_arms[active_arm] = sample_arms[active_arm]

    # Find the best arm
    # Our objective is a relative CTR, so status_quo is 0.0; we
    # know that the best arm cannot be worse than status_quo
    best_arm_val = 0.0
    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if arm_value > best_arm_val:
            best_arm_val = arm_value

    # Remove all arms that are more than two standard deviations worse than the best arm
    pruned_arms = copy.copy(active_arms)
    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if sample_arm.total > 0 and arm_value + 2.0 * numpy.sqrt(arm_variance) < best_arm_val:
            if verbose:
                print "Removing underperforming arm: {0}".format(sample_arm_point)
            pruned_arms.remove(sample_arm_point)

    return pruned_arms


def moe_experiment_from_sample_arms(sample_arms):
    """Make a MOE experiment with all historical data (i.e., parameter value, CTR, noise variance triples).

    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :return: MOE Experiment object usable with GP endpoints like ``gp_mean_var`` and ``gp_next_points``
    :rtype: :class:`moe.easy_interface.experiment.Experiment`

    """
    experiment = Experiment(EXPERIMENT_DOMAIN)
    for sample_arm_point, sample_arm in sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        # MOE *minimizes* and we want to *maximize* CTR so
        # we multiply the objetive (``arm_value``) by -1.0
        experiment.historical_data.append_sample_points([
            [
                sample_arm_point,
                -arm_value,
                arm_variance,
            ]])

    return experiment


def plot_system_ctr(system_ctr):
    """Plot the system ctr time series and write it to ``ctr_plot.pdf``.

    Time is "fake" so we index it 0, 1, .. len(system_ctr) - 1; i.e.,
    time is just iteration/round number in our experiment.

    :param system_ctr: ctr values to plot, ordered by "time"
    :type system_ctr: list of float

    """
    figure = plt.figure()
    ax = figure.add_subplot(111)

    # The total CTR of the system at each experiment interval
    ax.plot(
            system_ctr,
            'bo',
            )

    # status quo ctr as a dashed line, this is no optimization
    ax.plot(
            [0, len(system_ctr)],
            [true_click_through_rate(STATUS_QUO_PARAMETER), true_click_through_rate(STATUS_QUO_PARAMETER)],
            'k--',
            alpha=0.7,
            )

    # local max ctr as a dashed line, this is local optimization
    ax.plot(
            [0, len(system_ctr)],
            [true_click_through_rate(LOCAL_OPTIMAL_PARAMETER), true_click_through_rate(LOCAL_OPTIMAL_PARAMETER)],
            'k--',
            alpha=0.7,
            )

    ax.set_xlim(0, EXPERIMENT_ITERATIONS)
    plt.xlabel("Experiment Iteration")
    plt.ylabel("System CTR")
    plt.title("System CTR vs Experiment Iteration")

    plt.savefig("ctr_plot.pdf", bbox_inches=0)


def plot_sample_arms(active_arms, sample_arms, iteration_number):
    """Plot the underlying Gaussian Process, showing samples, GP mean/variance, and the true CTR.

    Plot is written to ``ab_plot_%.2d.pdf`` where ``%.2d`` is the two-digit ``iteration_number``.
    This makes it convenient to sequence the plots later for animation.

    The shows the current state of the Gaussian Process. Arms currently being sampled are highlighted
    with red Xs and previous arms are marked with blue Xs. The GP mean line is drawn and the variance
    region is shaded. The true CTR curve (that the GP approximates) is also shown.

    :param active_arms: list of coordinate-tuples corresponding to arms/cohorts currently being sampled
    :type active_arms: list of tuple
    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :param iteration_number: the index of the current iteration/round being tested in our experiment
    :type iteration_number: int >= 0

    """
    figure = plt.figure()
    ax = figure.add_subplot(111)

    experiment = moe_experiment_from_sample_arms(sample_arms)

    resolution = 0.01
    x_vals = [[x] for x in numpy.arange(0, 1, resolution)]  # uniform grid of points

    # Calculate the mean and variance of the underlying Gaussian Proccess
    mean, var = gp_mean_var(
            experiment.historical_data.to_list_of_sample_points(),
            x_vals,
            covariance_info=COVARIANCE_INFO,
            )
    mean = -numpy.array(mean)

    # Plot the mean line
    ax.plot(
            numpy.arange(0, 1, resolution),
            mean,
            'b--',
            )
    # Plot the variance
    var_diag = numpy.fabs(numpy.diag(var))
    ax.fill_between(
            numpy.arange(0, 1, resolution),
            mean - var_diag,
            mean + var_diag,
            facecolor='green',
            alpha=0.2,
            )

    # Plot the true, underlying CTR of the system wrt the parameter
    true_vals = numpy.array([true_click_through_rate(x) for x in x_vals]) / true_click_through_rate(STATUS_QUO_PARAMETER) - 1.0
    ax.plot(
            numpy.arange(0, 1, resolution),
            true_vals,
            'k--',
            alpha=0.5,
            )

    # Plot the observed CTR of the simulated system at sampled parameter values
    sample_points = experiment.historical_data.to_list_of_sample_points(),
    for sample_point in sample_points[0]:
        if tuple(sample_point.point) in active_arms:
            # New points are plotted as red x's
            fmt = 'rx'
        else:
            # Previously sampled points are plotted as blue x's
            fmt = 'bx'
        # These are simulated samples, include error bars
        ax.errorbar(
                sample_point.point,
                [-sample_point.value],
                yerr=sample_point.noise_variance,
                fmt=fmt,
                )

    plt.xlabel("Underlying Parameter")
    plt.ylabel("Relative CTR Gain vs Status Quo")
    plt.title("Relative CTR Gain and Gaussian Process")

    plt.savefig("ab_plot_%.2d.pdf" % iteration_number, bbox_inches=0)


def generate_initial_traffic():
    """Generate initial traffic allocations.

    ``active_arms``: list of coordinate-tuples corresponding to arms/cohorts currently being sampled

    ``sample_arms``: all arms from prev and current cohorts, keyed by coordinate-tuples.

    Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`

    :return: (active_arms, sample_arms) tuple as described above
    :rtype: tuple

    """
    ctr_at_status_quo = true_click_through_rate(STATUS_QUO_PARAMETER)

    # We draw the clicks from a binomial distribution
    clicks = numpy.random.binomial(TRAFFIC_PER_DAY, ctr_at_status_quo)
    status_quo_sample_arm = SampleArm(
            win=clicks,
            total=TRAFFIC_PER_DAY,
            )

    # We start with the only active arm being the status quo
    active_arms = [tuple(STATUS_QUO_PARAMETER)]
    sample_arms = {
            tuple(STATUS_QUO_PARAMETER): status_quo_sample_arm,
            }
    return active_arms, sample_arms


def generate_new_arms(active_arms, sample_arms, verbose=False):
    """Find optimal new parameters to sample to get ``active_arms`` up to ``NUMBER_OF_ACTIVE_COHORTS``.

    This is done in the following way:
        1) Find the initial allocations of all active parameters
        2) MOE suggests new, optimal arms/parameters to sample, given all previous samples, using Bayesian Global Optimization
        3) The new parameters are allocated traffic according to a Multi-Armed Bandit policy for all ``active_arms``
        4) If the objective function for a given parameter is too low (with high statistical significance), it is turned off
        5) Steps 2-4 are repeated until there are ``NUMBER_OF_ACTIVE_COHORTS`` parameters being sampled with non-zero traffic

    :param active_arms: list of coordinate-tuples corresponding to arms/cohorts currently being sampled
    :type active_arms: list of tuple
    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :param verbose: whether to print status messages to stdout
    :type verbose: bool
    :return: (allocations, active_arms, sample_arms) describing the next round of experiments to run

      ``allocations``: dict of traffic allocations, indexed by the ``active_arms`` return value

      ``active_arms``: new active arms to run in the next round of our experiment; same format as the input

      ``sample_arms``: the sample arms input updated with the newly generated active arms; same format as the input
    :rtype: tuple

    """
    # 1) Find initial allocations of all active parameters
    experiment = moe_experiment_from_sample_arms(sample_arms)
    allocations = get_allocations(active_arms, sample_arms)

    # Loop while we have too few arms
    while len(active_arms) < NUMBER_OF_ACTIVE_COHORTS:
        # 2) MOE suggests new, optimal arms/parameters to sample, given all previous samples, using Bayesian Global Optimization
        new_points_to_sample = find_new_points_to_sample(
                experiment,
                num_points=NUMBER_OF_ACTIVE_COHORTS - len(active_arms),
                verbose=verbose,
                )

        # Add the new points to the list of active_arms
        for new_point_to_sample in new_points_to_sample:
            sample_arms[tuple(new_point_to_sample)] = SampleArm()
            active_arms.append(tuple(new_point_to_sample))

        # 3) The new parameters are allocated traffic according to a Multi-Armed Bandit policy for all ``active_arms``
        allocations = get_allocations(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

        # 4) If the objective function for a given parameter is too low (with high statistical significance), it is turned off
        # Remove arms that have no traffic from active_arms
        active_arms = prune_arms(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

    return allocations, active_arms, sample_arms


def run_time_consuming_experiment(allocations, sample_arms, verbose=False):
    """Run the time consuming or expensive experiment.

    .. Note::

        Obtaining the value of the objective function is assmumed to be either time consuming or expensive,
        where every evaluation is precious and we want to find the optimal set of parameters with as few calls as possible.

    This experiment runs a simulation of user traffic over various parameters and users.
    It simulates an A/B testing experiment framework.

    For each arm/cohort, we compute the true CTR. From bandits, we know the allocation of
    traffic for that arm. So the simulation involves running (TRAFFIC_PER_DAY * allocation)
    Bernoulli trials each with probability = true CTR. Then since we cannot know the true CTR,
    we only work with the observed CTR and variance (as computed in
    :func:~`moe_examples.blog_post_example_ab_testing.objective_function`).

    :param allocations: traffic allocations for the ``active_arms``, one for each tuple in ``active_arms``.
      These are the cohorts for the experiment round--the parameters being tested and on what portion of traffic.
    :type allocations: dict
    :param sample_arms: all arms from prev and current cohorts, keyed by coordinate-tuples
      Arm refers specifically to a :class:`moe.bandit.data_containers.SampleArm`
    :type sample_arms: dict
    :param verbose: whether to print status messages to stdout
    :type verbose: bool

    """
    arm_updates = {}
    for arm_point, arm_allocation in allocations.iteritems():
        # Find the true, underlying CTR at the point
        ctr_at_point = true_click_through_rate(arm_point)
        # Calculate how much user traffic is allocated to this point
        traffic_for_point = int(TRAFFIC_PER_DAY * arm_allocation)
        # Simulate the number of clicks this point will garner
        clicks = numpy.random.binomial(
                traffic_for_point,
                ctr_at_point,
                )
        # Create a SampleArm with the assoicated simulated data
        sample_arm_for_day = SampleArm(
                win=clicks,
                total=traffic_for_point,
                )
        # Store the information about the simulated experiment
        arm_updates[arm_point] = sample_arm_for_day
        sample_arms[arm_point] = sample_arms[arm_point] + sample_arm_for_day

    if verbose:
        print "Updated the samples with:"
        for arm_name, sample_arm in arm_updates.iteritems():
            print "\t{0}: {1}".format(arm_name, sample_arm)

    return sample_arms, arm_updates


def calculate_system_ctr(arms):
    """Calculate the average CTR for the entire system, given a set of arms.

    This is different than :func:~`moe_examples.blog_post_example_ab_testing.objective_function`
    which computes observed mean and variance for a single arm.
    This function is a summary statistic for the entire experiment.

    :param arms: arms currently being evaluated in the system
    :type arms: list of :class:`moe.bandit.data_containers.SampleArm`
    :return: calculated system-wide CTR
    :type: float

    """
    clicks = 0
    impressions = 0
    for sample_arm in arms.itervalues():
        clicks += sample_arm.win
        impressions += sample_arm.total
    return float(clicks) / impressions


def run_example(verbose=False):
    """Run the example experiment framework.

    This is done in the following way:
        1) Generate initial data
        2) Run the experiment ``EXPERIMENT_ITERATIONS`` times
            a) Removing underperforming arms/parameters
            b) Generate new parameters to sample (aka new arms to replace those removed)
            c) Sample the new arms/parameters
            d) Plot the updated GP from the new parameters
            e) Repeat
        3) Plot out summary metrics

    .. Note::

        Below, ``active_arms`` is a list of tuples. The tuples are the coordinates of the point
        represented by ``active_arms``. This is so that members of ``active_arms`` can be used
        as keys. In turn, ``sample_arms`` is a dict keyed on those same coordinate tuples.
        The tuples make it convenient to switch between serializable types and lists/arrays.

        In practice, users would likely give their cohorts str names. These names could key
        both ``active_arms`` (which becomes a dict mapping name to coordinates) and
        ``sample_arms``.

    :param verbose: whether to print status messages to stdout
    :type verbose: bool

    """
    # 1) Generate initial data
    # Initialize the experiment with the status quo values
    active_arms, sample_arms = generate_initial_traffic()
    system_ctr = [
            calculate_system_ctr(sample_arms),
            ]

    # status_quo is iteration 0
    plot_sample_arms(active_arms, sample_arms, 0)

    # start indexing from 1, but we want EXPERIMENT_ITERATIONS total runs
    for iteration_number in range(1, EXPERIMENT_ITERATIONS + 1):
        # 2a) Remove underperforming arms/parameters
        active_arms = prune_arms(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

        # 2b) Generate new parameters to sample (aka new arms to replace those removed)
        allocations, active_arms, sample_arms = generate_new_arms(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

        # 2c) Sample the new parameters/arms
        sample_arms, arm_updates = run_time_consuming_experiment(
                allocations,
                sample_arms,
                verbose=verbose,
                )

        # save the current system CTR for the final summary plot
        system_ctr.append(
                calculate_system_ctr(arm_updates)
                )

        # 2d) Plot the updated GP from the new parameters
        plot_sample_arms(active_arms, sample_arms, iteration_number)

    # 3) Plot out summary metrics
    # plot the CTR time series (i.e., vs iteration number)
    plot_system_ctr(system_ctr)


if __name__ == '__main__':
    # Run the example from the blog post
    run_example(verbose=True)
