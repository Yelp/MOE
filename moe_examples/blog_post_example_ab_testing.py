# -*- coding: utf-8 -*-
"""An example of using MOE to optimize a simulated A/B testing framework.

Blog post: TODO(sclark): Link to blog post
---------
"""
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
STATUS_QUO_PARAMETER = numpy.array([0.2])
LOCAL_OPTIMAL_PARAMETER = numpy.array([0.14289063])
GLOBAL_OPTIMAL_PARAMETER = numpy.array([0.71428711])
TRAFFIC_PER_DAY = 1000000  # 1 Million
NUMBER_OF_ACTIVE_COHORTS = 3
EXPERIMENT_ITERATIONS = 5
COVARIANCE_INFO = {
        'hyperparameters': [0.3, 0.1],
        }


def click_through_rate(x):
    r"""Return the underlying Click Through Rate (CTR) with respect to the parameter ``x``.

    Higher values are better. There is an local optima at ``LOCAL_OPTIMAL_PARAMETER`` (0.14289063) and a global optima at ``GLOBAL_OPTIMAL_PARAMETER`` (0.71428711).

    ..Note:

        The underlying function does not need to be differentiable or even continuous for MOE to be able to optimize it.

    math::

        \texttt{CTR}(x) = \begin{cases}
                \frac{1}{100}\left( \sin\left( \frac{7\pi}{2} x\right) + 2.5\right) & x \leq 0.6 \\
                \frac{1}{100}\left( \sin\left( \frac{7\pi}{2} x\right) + 1.1\right) & x > 0.6
            \end{cases}

    :param x: The parameter that we are optimizing
    :type x: 1-D numpy array with coordinate in [0,1]
    """
    if x[0] > 0.6:
        return (numpy.sin(x[0] * 3.5 * numpy.pi) + 2.5) / 100.0
    else:
        return (numpy.sin(x[0] * 3.5 * numpy.pi) + 1.1) / 100.0


def bernoulli_mean(observed_clicks, samples):
    """Return the mean and variance of the mean of a bernoulli distribution with a given number of successes (clicks) and trials (impressions)."""
    if samples > 0:
        observed_success_rate = observed_clicks / float(samples)
    else:
        observed_success_rate = 0.0

    # We use the fact that the Beta distribution is the cojugate prior to the binomial distribution
    alpha = observed_clicks + 1
    beta = samples - observed_clicks + 1
    observed_variance = (alpha * beta) / float((alpha + beta) * (alpha + beta) * (alpha + beta + 1))

    return observed_success_rate, observed_variance


def objective_function(observed_sample_arm, observed_status_quo_sample_arm):
    r"""The objective function MOE is attempting to maximize.

    Lower values are better (minimization). Our objective is relative to the CTR of the initial parameter value, ``STATUS_QUO_PARAMETER``.

    math::

        \Phi(\vec{x}) = \frac{\mathtt{CTR}(\vec{x})}{\mathtt{CTR}(\vec{SQ})}
    """
    sample_ctr, sample_ctr_var = bernoulli_mean(
            observed_sample_arm.win,
            observed_sample_arm.total,
            )
    status_quo_ctr, status_quo_ctr_var = bernoulli_mean(
            observed_status_quo_sample_arm.win,
            observed_status_quo_sample_arm.total,
            )

    # Note: We take an upper bound of the variance (by ignoring the correlation between the two random variables)
    return sample_ctr / status_quo_ctr - 1, sample_ctr_var + status_quo_ctr_var


def find_new_points_to_sample(exp, num_points=1, verbose=False):
    """Find the optimal next point to sample using expected improvement."""
    if verbose:
        print "Getting {0} new suggested point(s) to sample from MOE...".format(num_points)

    # Query MOE for the next points to sample
    next_points_to_sample = gp_next_points(
            exp,
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
    """Return the allocation for each active_arm using the epsilon greedy multi-armed bandit strategy."""
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        # json wants all arm names to be strings
        active_sample_arms[str(active_arm[0])] = sample_arms[active_arm]

    # Bundle up the arm information
    bandit_data = BanditHistoricalData(
            sample_arms=active_sample_arms,
            )

    # Query the MOE endpoint for optimal traffic allocation
    bandit_allocation = bandit_endpoint(
            bandit_data,
            type=BANDIT_EPSILON_ROUTE_NAME,
            subtype=EPSILON_SUBTYPE_GREEDY,
            hyperparameter_info={
                'epsilon': 0.10,
                },
            )

    arm_allocations = {}
    for arm_name_as_string, allocation in bandit_allocation.iteritems():
        arm_allocations[tuple([float(arm_name_as_string)])] = allocation

    if verbose:
        print "Optimal arm allocations: {0}".format(arm_allocations)

    return arm_allocations


def prune_arms(active_arms, sample_arms, verbose=False):
    """Remove all arms from ``active_arms`` that have an allocation less than two standard deviations below the current best arm."""
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        active_sample_arms[active_arm] = sample_arms[active_arm]

    # Find the best arm
    best_arm_val = 0.0
    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if arm_value > best_arm_val:
            best_arm_val = arm_value

    # Remove all arms that are more than two standard deviations worse than the best arm
    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if sample_arm.total > 0 and arm_value + 2.0 * numpy.sqrt(arm_variance) < best_arm_val:
            if verbose:
                print "Removing underperforming arm: {0}".format(sample_arm_point)
            active_arms.remove(sample_arm_point)

    return active_arms


def moe_exp_from_sample_arms(sample_arms):
    """Make MOE experiment with all historical data."""
    exp = Experiment([[0, 1]])
    for sample_arm_point, sample_arm in sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        exp.historical_data.append_sample_points([
            [
                sample_arm_point,
                -arm_value,
                arm_variance,
            ]])
    return exp


def plot_system_ctr(system_ctr):
    """Plot the system ctr time series."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # The total CTR of the system at each experiment interval
    ax.plot(
            system_ctr,
            'bo',
            )

    # status quo ctr as a dashed line, this is no optimization
    ax.plot(
            [0, len(system_ctr)],
            [click_through_rate(STATUS_QUO_PARAMETER), click_through_rate(STATUS_QUO_PARAMETER)],
            'k--',
            alpha=0.7,
            )

    # local max ctr as a dashed line, this is local optimization
    ax.plot(
            [0, len(system_ctr)],
            [click_through_rate(LOCAL_OPTIMAL_PARAMETER), click_through_rate(LOCAL_OPTIMAL_PARAMETER)],
            'k--',
            alpha=0.7,
            )

    ax.set_xlim(0, EXPERIMENT_ITERATIONS)
    plt.xlabel("Experiment Iteration")
    plt.ylabel("System CTR")
    plt.title("System CTR vs Experiment Iteration")

    plt.savefig("ctr_plot.pdf", bbox_inches=0)


def plot_sample_arms(active_arms, sample_arms, iteration_number):
    """Plot the sample arms."""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    exp = moe_exp_from_sample_arms(sample_arms)

    res = 0.01
    x_vals = [[x] for x in numpy.arange(0, 1, res)]  # uniform grid of points

    # Calculate the mean and variance of the underlying Gaussian Proccess
    mean, var = gp_mean_var(
            exp.historical_data.to_list_of_sample_points(),
            x_vals,
            covariance_info=COVARIANCE_INFO,
            )
    mean = -numpy.array(mean)

    # Plot the mean line
    ax.plot(
            numpy.arange(0, 1, res),
            mean,
            'b--',
            )
    # Plot the variance
    var_diag = abs(numpy.diag(var))
    ax.fill_between(
            numpy.arange(0, 1, res),
            mean - var_diag,
            mean + var_diag,
            facecolor='green',
            alpha=0.2,
            )

    # Plot the true, underlying CTR of the system wrt the parameter
    true_vals = numpy.array([click_through_rate(x) for x in x_vals]) / click_through_rate(STATUS_QUO_PARAMETER) - 1.0
    ax.plot(
            numpy.arange(0, 1, res),
            true_vals,
            'k--',
            alpha=0.5,
            )

    # Plot the observed CTR of the simulated system at sampled parameter values
    sample_points = exp.historical_data.to_list_of_sample_points(),
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
    """Generate initial traffic."""
    ctr_at_status_quo = click_through_rate(STATUS_QUO_PARAMETER)

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
            a) If the objective function for a given parameter is too low, to high statistical significance, it is turned off
        2) MOE suggests new parameters to sample, given all previous samples, using Bayesian Global Optimization
        3) The new parameters are allocated traffic according to a Multi-Armed Bandit policy
        4) Steps 2-3 are repeated until there are ``NUMBER_OF_ACTIVE_COHORTS`` parameters being sampled with non-zero traffic

    """
    # Find initial allocations
    exp = moe_exp_from_sample_arms(sample_arms)
    allocations = get_allocations(active_arms, sample_arms)

    # Loop while we have too few arms
    while len(active_arms) < NUMBER_OF_ACTIVE_COHORTS:
        # Find optimal new arms to sample
        new_points_to_sample = find_new_points_to_sample(
                exp,
                num_points=NUMBER_OF_ACTIVE_COHORTS - len(active_arms),
                verbose=verbose,
                )

        # Add the new points to the list of active_arms
        for new_point_to_sample in new_points_to_sample:
            sample_arms[tuple(new_point_to_sample)] = SampleArm()
            active_arms.append(tuple(new_point_to_sample))

        # Get traffic allocations for all active_arms
        allocations = get_allocations(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

        # Remove arms that have no traffic from active_arms
        active_arms = prune_arms(
                active_arms,
                sample_arms,
                verbose=verbose,
                )

    return allocations, active_arms, sample_arms


def run_time_consuming_experiment(allocations, sample_arms, verbose=False):
    """Run the time consuming or expensive experiment.

    ..note::

        Obtaining the value of the objective function is assmumed to be either time consuming or expensive, where every evaluation is precious and we want to find the optimal set of parameters with as few calls as possible.

    This experiment runs a simulation of user traffic over various parameters and users. It simulates an A/B testing experiment framework.
    """
    arm_updates = {}
    for arm_point, arm_allocation in allocations.iteritems():
        # Find the true, underlying CTR at the point
        ctr_at_point = click_through_rate(arm_point)
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
    """Calculate the CTR for the entire system, given a set of arms."""
    clicks = 0
    impressions = 0
    for sample_arm in arms.itervalues():
        clicks += sample_arm.win
        impressions += sample_arm.total
    return float(clicks) / impressions


def run_example():
    """Run the example experiment framework.

    This is done in the following way:
        1) Generate initial data
        2) Run the experiment ``EXPERIMENT_ITERATIONS`` times
            a) Removing poorly performing parameters
            b) Get new parameters to sample
            c) Sample the new parameters
            d) Repeat
        3) Plot out metrics
    """
    # Initialize the experiment with the status quo values
    active_arms, sample_arms = generate_initial_traffic()
    system_ctr = [
            calculate_system_ctr(sample_arms),
            ]

    for iteration_number in range(EXPERIMENT_ITERATIONS):

        plot_sample_arms(active_arms, sample_arms, iteration_number)

        # Remove underperforming arms
        active_arms = prune_arms(
                active_arms,
                sample_arms,
                verbose=True,
                )

        # Generate new arms to replace those removed
        allocations, active_arms, sample_arms = generate_new_arms(
                active_arms,
                sample_arms,
                verbose=True,
                )

        # Sample the arms
        sample_arms, arm_updates = run_time_consuming_experiment(
                allocations,
                sample_arms,
                verbose=True,
                )

        system_ctr.append(
                calculate_system_ctr(arm_updates)
                )

    plot_system_ctr(system_ctr)

if __name__ == '__main__':
    # Run the example from the blog post
    run_example()
