import math
import random
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, List, Union
from bisect import bisect_left


def symmetric_bernoulli(p: float = 0.5) -> int:
    return 1 if random.random() < p else -1


def normalize_sum(n: int, random_var_realization: List[Callable[[], int]], rv_mean: float, rv_std: float) -> float:
    """_summary_

    Args:
        n (int): the number of random variables
        random_var_realization (List[Callable[[], int]]): a callable that returns a realization of the random variable

    Returns:
        float: the realization of the normalized sum of the random variables
    """
    
    return (np.sum([random_var_realization() for _ in range(n)]) - n * rv_mean) / (rv_std * np.sqrt(n)).item()


def simulate_random_variable(num_trials: int, random_var_realization: Callable[[], int]) -> List[float]:
    """This functino simulates the random variable by producing "num_trials" realization and returns them.

    The output can be used for a variety of estimations such as: 
        1. emperical estimation of the random variable CDF

    Args:
        num_trials (int): the number of realizations to simulate    
        random_var_realization (Callable[[], int]): a callable that returns a realization of the random variable

    Returns:
        List[float]: the realizations of the random variable
    """
    return [random_var_realization() for _ in range(num_trials)]


def estimate_tail_distribution(num_trials: int, random_var_realization: Callable[[], int], tail_values: List[float]) -> List[float]:
    """This function estimates the tail distribution of the random variable by producing "num_trials" realizations 
    and then counting the number of realizations that fall into each tail.

    Args:
        num_trials (int): the number of realizations to simulate
        random_var_realization (Callable[[], int]): a callable that returns a realization of the random variable
        tail_values (List[float]): the values of the tail to estimate
        
    Returns:
        List[float]: The probability estimates P(X > t) for each t in tail_values
    """ 
    # the tail distribution P(X > t) can be estimated as follows:
    # 1. simulate the random variable num_trials times
    # 2. sort the realizations
    # 3. for each tail value t, count the number of realizations that are greater than t
    # 4. divide the count by the total number of realizations: the ratio is an estimation of P(X > t)

    # sort and then apply the bisect_left method to compute 
    estimations = sorted(simulate_random_variable(num_trials, random_var_realization))
    
    # For each tail value t, calculate P(X > t) = (number of values > t) / total_values
    tail_probabilities = [
        (num_trials - bisect_left(estimations, t)) / num_trials 
        for t in tail_values
    ]

    return tail_probabilities


def guassian_tail_distribution(t: float) -> float:
    """This function computes the tail distribution of the normalized sum of n i.i.d. Symmetric Bernoulli random variables.

    Args:
        t (float): the tail value

    """
    return np.exp(-t**2 / 2).item()



def task1(num_trials: int):
    # create the plot and stuff
    n_values = [10, 50, 100, 200, 500, 1000]

    # Create figure and subplots with more height to accommodate spacing
    fig, axes = plt.subplots(nrows=len(n_values), ncols=1, figsize=(18, 20))
    
    # Add more vertical space between subplots
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots

    # different values of n

    # the values that a normalized sum of n i.i.d. Symmetric Bernoulli can take
    # the values are symmetric around 0, and the range is [-n, n]
    # we can generate them with a list comprehension
    
    for index, n in enumerate(n_values):
        # define the random variable realization
        rv = lambda : normalize_sum(n, symmetric_bernoulli, 0, 1) # 0 mean, 1 std
        
        low = 0
        high = math.ceil(np.sqrt(n))

        tail_values = list(range(low, high+1))

        # estimate tail distributions
        tail_distributions = estimate_tail_distribution(num_trials, rv, tail_values)

        # theoretical tail distribution
        theoretical_tail_distributions = [guassian_tail_distribution(t) for t in tail_values]

        # plot the tail distributions
        axes[index].plot(tail_values, tail_distributions, label=f'empirical tail distribution')
        axes[index].plot(tail_values, theoretical_tail_distributions, label=f'theoretical tail distribution')
        axes[index].set_title(f'tail distributions of normalized sum of {n} i.i.d. Symmetric Bernoulli random variables')
        axes[index].legend()
        # set the x to contain at least 10 values
        axes[index].set_xticks([int(v) for v in np.linspace(low, high, 11)]) 

        # add descriptions to the 'y' and 'x' axes
        axes[index].set_ylabel('P(X > t)')
        axes[index].set_xlabel('t')

    plt.show()


def meanSymmetricBernoulli(p: float):
    return 2 * p - 1

def stdSymmetricBernoulli(p: float):
    return 2 - 2 * p

def task2(num_trials: int, n: int = 100):

    p_vals = [0.25, 0.4, 0.6]

    # Create figure and subplots with more height to accommodate spacing
    fig, axes = plt.subplots(nrows=len(p_vals), ncols=1, figsize=(18, 20))
    
    for index, p in enumerate(p_vals):
        mean = meanSymmetricBernoulli(p)
        var = np.sqrt(stdSymmetricBernoulli(p))

        # define the random variable realization
        rv = lambda : normalize_sum(n, symmetric_bernoulli, mean, var)
        
        low = 0
        high = math.ceil(np.sqrt(n))

        tail_values = list(range(low, high+1))

        # estimate tail distributions
        tail_distributions = estimate_tail_distribution(num_trials, rv, tail_values)

        # theoretical tail distribution
        theoretical_tail_distributions = [guassian_tail_distribution(t) for t in tail_values]

        # plot the tail distributions
        axes[index].plot(tail_values, tail_distributions, label=f'empirical tail distribution')
        axes[index].plot(tail_values, theoretical_tail_distributions, label=f'theoretical tail distribution')
        axes[index].set_title(f'tail distributions of normalized sum of {n} i.i.d. Symmetric Bernoulli random variables with p = {p}')
        axes[index].legend()
        # set the x to contain at least 10 values
        axes[index].set_xticks([int(v) for v in np.linspace(low, high, 11)]) 

        # add descriptions to the 'y' and 'x' axes
        axes[index].set_ylabel('P(X > t)')
        axes[index].set_xlabel('t')

    plt.show()  



def mean_estimator(num_trials: int, rv_realization: Callable[[], Union[int, float]]) -> float:
    return np.mean([rv_realization() for _ in range(num_trials)]).item()


def mean_of_median_estimator(num_trials: int, k: int, rv_realization: Callable[[], Union[int, float]]) -> float:
    if num_trials % k != 0:
        raise ValueError("num_trials must be divisible by k")
    
    # split the trials into k groups
    groups = [simulate_random_variable(k, rv_realization) for _ in range(num_trials // k)]
    
    # compute the mean of each group
    means = [np.mean(group).item() for group in groups]

    # compute the median of the means
    return np.median(means).item()


def run_estimators(num_estimator_trials: int, num_trials: int, k: int, n: int):
    """n : represents the number of i.i.d Systematic Bernoulli random variables producing the random variable X

    num_trials: the number of i.i.d X stimulate, each of these trials will give a realization of the estimator

    num_estimator_trials: the number of realizations for the estimator value
    """

    X = lambda : normalize_sum(n, symmetric_bernoulli, 0, 1) 

    classical_mean = [mean_estimator(num_trials, X) for _ in range(num_estimator_trials)]
    median_of_means = [mean_of_median_estimator(num_trials, k, X) for _ in range(num_estimator_trials)]

    return classical_mean, median_of_means



def task3(num_estimator_trials: int, num_trials: int, k):

    plt.figure(figsize=(18, 8))

    n_values = [10, 100, 200, 500, 1000, 2000, 5000, 10000]

    # define the random variable realization    

    mcs, mcms = [], []

    for _, n in enumerate(n_values):
        classical_mean_estimations, mean_of_median_estimations = run_estimators(num_estimator_trials, num_trials, k, n)

        mc = np.var(classical_mean_estimations).item()
        mcm = np.var(mean_of_median_estimations).item()

        mcs.append(mc)
        mcms.append(mcm)

    # plot the tail distributions
    plt.plot(mcs, label=f'classical mean')
    plt.plot(mcms, label=f'mean of median')

    # horizontal line at the true mean
    plt.title(f'classical mean and mean of median estimators of a sum of {n} i.i.d. Symmetric Bernoulli random variables')

    # add descriptions to the 'y' and 'x' axes
    plt.ylabel('mean of the estimator')
    plt.xlabel('number of (estimator) trials')


    plt.xticks(range(len(n_values)), [str(n) for n in n_values])

    plt.axhline(y=0, color='black', linestyle='--', label='true mean = 0')

    plt.legend()
    plt.show()      
    

def task4(num_estimator_trials: int, num_trials: int, k):

    plt.figure(figsize=(18, 8))

    n_values = [10, 100, 200, 500, 1000, 2000, 5000, 10000]

    mcs, mcms, true_vars = [], [], []

    for index, n in enumerate(n_values):
        classical_mean_estimations, mean_of_median_estimations = run_estimators(num_estimator_trials, num_trials, k, n)

        mc = np.var(classical_mean_estimations).item()
        mcm = np.var(mean_of_median_estimations).item()

        mcs.append(mc)
        mcms.append(mcm)

        true_vars.append(1 / n)

    # plot the tail distributions
    plt.plot(mcs, label=f'classical mean')
    plt.plot(mcms, label=f'mean of median')
    plt.plot(true_vars, label=f'theoretical variance of the estimators')

    # horizontal line at the true mean
    plt.title(f'classical mean and mean of median estimators of a sum of {n} i.i.d. Symmetric Bernoulli random variables')

    # add descriptions to the 'y' and 'x' axes
    plt.ylabel('variance of the estimator')
    plt.xlabel('number of (estimator) trials') 

    plt.xticks(range(len(n_values)), [str(n) for n in n_values])

    plt.legend()
    plt.show()      
    


def task5(num_estimator_trials: int, num_trials: int, k):

    n_values = [100, 500, 1000]


    # Create figure and subplots with more height to accommodate spacing
    fig, axes = plt.subplots(nrows=len(n_values), ncols=1, figsize=(18, 20))
    

    for index, n in enumerate(n_values):
        classical_mean_estimations, mean_of_median_estimations = run_estimators(num_estimator_trials, num_trials, k, n)
        # plot the tail distributions

        # plot the histograms
        # Set alpha (transparency) for both histograms to make overlapping regions visible
        axes[index].hist(classical_mean_estimations, label=f'classical mean realizations', density=True, alpha=0.5)
        axes[index].hist(mean_of_median_estimations, label=f'mean of median realizations', density=True, alpha=0.5)
        
        axes[index].set_title(f'the distributions of the classical mean and mean of median estimators of a sum of {n} i.i.d. Symmetric Bernoulli random variables')
        axes[index].legend()

        # add descriptions to the 'y' and 'x' axes
        axes[index].set_xlabel('estimator value probability')
        axes[index].set_ylabel('estimator value')

    plt.show()