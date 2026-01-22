import torch
import imageio
import numpy as np
import random

""" Some dynamic utils useful"""


@torch.no_grad()
def render_ode_traj(dynamical_model):
    """
    Render 2D point cloud ODE trajectory
    """
    raise NotImplementedError


def curriculum_training_strategy(total_iterations, num_timesteps, base=1, min_iterations=500):
    """
    Calculate the number of iterations for each curriculum stage with increasing emphasis on later stages.
    
    Args:
        total_iterations (int): Total number of training iterations
        num_timesteps (int): Total number of timesteps in the sequence
        base (float): Base for exponential weighting (higher values put more emphasis on later stages)
        if base == 1, then uniform
    
    Returns:
        dict: Dictionary mapping sequence length to number of iterations
    NOTE: Set min_iterations <= max_training_steps //  number_of_timesteps
    """
    required_min_iterations = min_iterations * num_timesteps #calculate how much we need to allocate at minimum

    if total_iterations < required_min_iterations:
        raise ValueError(f"Total iterations ({total_iterations}) must be at least {required_min_iterations} "
                        f"to ensure minimum {min_iterations} iterations per stage")
    
    # Calculate remaining iterations after satisfying minimums
    remaining_iterations = total_iterations - required_min_iterations
    
    # Calculate weights for each stage (exponentially increasing)
    weights = np.array([base ** i for i in range(num_timesteps)])
    normalized_weights = weights / weights.sum()
    
    # how much we allocate per each timestep 
    additional_iterations = (normalized_weights * remaining_iterations).astype(int) 
    
    # Ensure we use exactly remaining_iterations by adjusting the last stage
    additional_iterations[-1] += remaining_iterations - additional_iterations.sum()
    
    # Create curriculum schedule with minimum iterations plus weighted additional iterations
    curriculum_schedule = {} #tells you how many training iterations for each timestep
    for i in range(num_timesteps):
        time_index = i + 1
        curriculum_schedule[time_index] = min_iterations + additional_iterations[i]

    running_sum_lst = []
    cumulative = 0
    for v in curriculum_schedule.values():
        cumulative += v
        running_sum_lst.append(cumulative)
    #check all timesteps are being trained on and sum is total number of timesteps.
    sum = 0
    for k, v in curriculum_schedule.items():
        assert v != 0
        sum += v
    assert sum == total_iterations 
    # Changing the end of running_sum_lst to match the last iteration number
    running_sum_lst[-1] = total_iterations - 1
    return curriculum_schedule, running_sum_lst 


def generate_linear_sequence(target_sum, n, start=None):
    """
    Generate a linearly increasing sequence of n numbers that sum to target_sum.
    
    Parameters:
    - target_sum: The desired sum of the sequence
    - n: Number of elements in the sequence
    - start: First element of the sequence (if None, it's calculated)
    
    Returns:
    - List of n linearly increasing numbers that sum to target_sum
    """
    if n <= 0:
        return []
    
    if start is None:
        # Calculate the first term using the formula:
        # target_sum = n*a + d*n*(n-1)/2
        # If we set d = 1, then:
        # a = (target_sum - n*(n-1)/2) / n
        a = (target_sum - (n*(n-1)/2)) / n
    else:
        a = start
        
    while target_sum - n*a < (n*(n-1)/2): #we want the ratio to be at least the denominator, so we have at least increment of 1
        a -= 1 

    print(f"we start with a value of {a}")
    # Calculate the common difference
    d = (target_sum - n*a) / (n*(n-1)/2) if n > 1 else 0
    
    # Generate the sequence
    integer_list = [int(a + i*d) for i in range(n)]
    remainder  = target_sum - sum(integer_list)
    for i in range(remainder):  #to maintain monotonicity, add from the back onward, this will break the start but wtv
        mod_i = i % n
        back_i = n - mod_i - 1
        integer_list[back_i] +=1
    assert sum(integer_list) == target_sum, "something is wrong, the sum of our integer list is not equal to the target"
    return integer_list

    
def generate_quadratic_sequence(target_sum, n, reduction_factor=0.2, start=None):
    """
    Generate a sequence of n numbers that increase according to x² and sum to target_sum.
    
    Parameters:
    - target_sum: The desired sum of the sequence
    - n: Number of elements in the sequence
    - reduction_factor: Factor to reduce the quadratic growth (default: 0.2)
    - start: First element of the sequence (if None, it's calculated)
    
    Returns:
    - List of n numbers with quadratic growth that sum to target_sum
    """
    if n <= 0:
        return []
    
    # Calculate the sum of the quadratic part: reduction_factor * (0², 1², 2², ..., (n-1)²)
    quad_terms = [reduction_factor * (i*i) for i in range(n)]
    quad_sum = sum(quad_terms)
    
    if start is None:
        # Calculate an appropriate starting value
        a = (target_sum - quad_sum) / n
        # Ensure a is an integer and not too large
        a = int(a)
    else:
        a = start
    
    # Check if a is too large by calculating the actual sum with this value of a
    test_sum = sum(int(a + quad_terms[i]) for i in range(n))
    
    # If test_sum is greater than target_sum, reduce a until we get under target
    while test_sum > target_sum:
        a -= 1
        test_sum = sum(int(a + quad_terms[i]) for i in range(n))
    
    # Generate the sequence with scaled quadratic growth
    integer_list = [int(a + quad_terms[i]) for i in range(n)]
    
    # Distribute any remainder to ensure the target sum
    remainder = target_sum - sum(integer_list)
    for i in range(remainder):
        # Add to elements from the end to maintain the growth pattern
        mod_i = i % n
        back_i = n - mod_i - 1
        integer_list[back_i] += 1
    
    assert sum(integer_list) == target_sum, "Sum of the integer list does not equal the target"
    return integer_list
        