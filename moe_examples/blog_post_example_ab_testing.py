# -*- coding: utf-8 -*-
"""An example of using MOE to optimize an A/B testing framework.

Blog post:

More info
---------
"""
import numpy

from moe.bandit.data_containers import HistoricalData as BanditHistoricalData
from moe.bandit.data_containers import SampleArm
from moe.bandit.epsilon_greedy import EpsilonGreedy
from moe.bandit.ucb1_tuned import UCB1Tuned
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points, gp_hyper_opt, gp_mean_var
from moe.optimal_learning.python.constant import GRADIENT_DESCENT_OPTIMIZER
from moe.views.constant import GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_ROUTE_NAME

STATUS_QUO_PARAMETER = numpy.array([0.2])
TRAFFIC_PER_DAY = 1000000  # 1 Million
NUMBER_OF_ACTIVE_COHORTS = 3
EXPERIMENT_ITERATIONS = 10

def click_through_rate(x):
    """This is the underlying Click Through Rate (CTR) with respect to the parameter ``x``

    Higher values are better. There is an optima at VALUE and a global optima at VALUE.

    http://www.wolframalpha.com/input/?i=%28sin%28x+*+3.5*+pi%29+%2B+1%29+*+%281+%2B+1%2F%281+%2B+exp%28-10000*%28x-0.6%29%29%29%29+%2F+200.0+%2B+0.002%2C+x+in+%5B0%2C+1%5D

    :param x: The parameter that we are optimizing
    :type x: 1-D numpy array with coordinate in [0,1]
    """
    return (numpy.sin( x[0] * 3.5 * numpy.pi ) + 1) * (1.0 + 1.0 / (1.0 + numpy.exp(-10000.0 * (x[0] - 0.6)))) / 200.0 + 0.002

def bernoulli_mean(observed_clicks, samples):
    """Return the mean (and variance of the mean) of a bernoulli distribution with ..."""
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
    """The objective function MOE is attempting to maximize.

    Lower values are better (minimization). Our objective is relative to the CTR of the ``STATUS_QUO_PARAMETER``, SQ.

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

    # TODO fix variance
    return sample_ctr / status_quo_ctr - 1, numpy.sqrt(sample_ctr_var*sample_ctr_var + status_quo_ctr_var*status_quo_ctr_var)

def assign_traffic(moe_experiment):
    """Determines the optimal assignment of traffic for each parameter using the BLA bandit, given historical information.

    :param moe_experiment: A MOE experiment
    :type moe_experiment: A MOE experiment (link)
    """
    pass

def find_new_points_to_sample(exp, num_points=1):
    """Find the optimal next point to sample using expected improvement."""
    # First we optimize the hyperparameters of the GP
    print "Updating hyperparameters..."
    covariance_info = gp_hyper_opt(
            exp.historical_data.to_list_of_sample_points(),
            )

    # Next we query MOE for the next points to sample
    print "Getting {0} point(s) to sample from MOE...".format(num_points)
    next_points_to_sample = gp_next_points(
            exp,
            method_route_name=GP_NEXT_POINTS_KRIGING_ROUTE_NAME,
            covariance_info=covariance_info,
            num_to_sample=num_points,
            optimizer_info={
                'optimizer_type': GRADIENT_DESCENT_OPTIMIZER,
                },
            )
    print "sample next", next_points_to_sample
    return next_points_to_sample

def get_allocations(active_arms, sample_arms):
    """Return the allocation for each active_arm."""
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        active_sample_arms[active_arm] = sample_arms[active_arm]

    bandit_data = BanditHistoricalData(
            sample_arms=active_sample_arms,
            )
    # find initial allocation

    bandit = EpsilonGreedy(
            historical_info=bandit_data,
            epsilon=0.10,
            )

    return bandit.allocate_arms()

def prune_arms(active_arms, sample_arms):
    """Remove all arms from ``active_arms`` that have an allocation less than the ``threshold``."""
    # Find all active sample arms
    active_sample_arms = {}
    for active_arm in active_arms:
        active_sample_arms[active_arm] = sample_arms[active_arm]

    best_arm_val = 0.0
    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if arm_value > best_arm_val:
            best_arm_val = arm_value

    for sample_arm_point, sample_arm in active_sample_arms.iteritems():
        arm_value, arm_variance = objective_function(
                sample_arm,
                sample_arms[tuple(STATUS_QUO_PARAMETER)],
                )
        if sample_arm.total > 0 and arm_value + 2.0 * numpy.sqrt(arm_variance) < best_arm_val:
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

def plot_sample_arms(sample_arms):
    """Plot the sample arms."""
    for sample_arm_name, sample_arm in sample_arms.iteritems():
        print sample_arm_name, sample_arm.json_payload()

def plot_allocations(allocations):
    """Plot the arm allocations."""
    print allocations

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

def generate_new_arms(active_arms, sample_arms):
    """Find optimal new arms to get ``active_arms`` up to ``NUMBER_OF_ACTIVE_COHORTS``."""
    exp = moe_exp_from_sample_arms(sample_arms)
    # while num arms < 3 draw new arms
    while len(active_arms) < NUMBER_OF_ACTIVE_COHORTS:
        new_points_to_sample = find_new_points_to_sample(
                exp,
                num_points=NUMBER_OF_ACTIVE_COHORTS - len(active_arms),
                )

        for new_point_to_sample in new_points_to_sample:
            sample_arms[tuple(new_point_to_sample)] = SampleArm()
            active_arms.append(tuple(new_point_to_sample))

        allocations = get_allocations(active_arms, sample_arms)
        active_arms = prune_arms(active_arms, sample_arms)
    return allocations, active_arms, sample_arms

def run_time_consuming_experiment(allocations, sample_arms):
    """This is the experiment."""
    for arm_point, arm_allocation in allocations.iteritems():
        ctr_at_point = click_through_rate(arm_point)
        traffic_for_point = int(TRAFFIC_PER_DAY * arm_allocation)
        clicks = numpy.random.binomial(
                traffic_for_point,
                ctr_at_point,
                )
        sample_arm_for_day = SampleArm(
                win=clicks,
                total=traffic_for_point,
                )
        sample_arms[arm_point] = sample_arms[arm_point] + sample_arm_for_day
    return sample_arms

def run_example():
    """Initialize the experiment with the status quo values."""

    active_arms, sample_arms = generate_initial_traffic()

    for i in range(EXPERIMENT_ITERATIONS):
        plot_sample_arms(sample_arms)

        # Remove underperforming arms
        active_arms = prune_arms(active_arms, sample_arms)

        # Generate new arms to replace those removed
        allocations, active_arms, sample_arms = generate_new_arms(active_arms, sample_arms)

        plot_allocations(allocations)

        # Sample the arms
        sample_arms = run_time_consuming_experiment(allocations, sample_arms)

if __name__ == '__main__':
    run_example()
