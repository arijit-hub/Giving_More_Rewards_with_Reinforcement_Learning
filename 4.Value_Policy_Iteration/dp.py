import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy

def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    # TODO: Write your implementation here

    ## Loop over all the states ##
    for state in range(mdp.num_states):
        state_val = 0

        ## Loop over all the actions ##
        for action in range(mdp.num_actions):

            ## Each action will have different target state and reward ##
            for action_prob , target_state , reward , is_done in mdp.P[state][action]:
                state_val += policy[state][action] * action_prob * (reward + discount * V[target_state])

        V_new[state] = state_val
    
    return V_new

def policy_evaluation(mdp, policy, discount=0.99, theta=0.01):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here

    ## Iterate until convergence ##

    while True:
        V_new = policy_evaluation_one_step(mdp, V, policy, discount)

        max_delta = np.max(np.abs(V - V_new))

        V = V_new.copy()

        if max_delta < theta:
            break

    return V

def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))

    # TODO: Write your implementation here

    ## Loop over all states ##
    for state in range(mdp.num_states):

        Q = np.zeros(mdp.num_actions)
        ## Loop over all actions ##
        for action in range(mdp.num_actions):

            for prob , target_state , reward , is_done in mdp.P[state][action]:
                Q[action] = prob * (reward + discount * V[target_state])

        greedy_action = np.argmax(Q)

        policy[state][greedy_action] = 1.0

    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    # This is only here for the skeleton to run.
    V = init_value(mdp)

    # TODO: Write your implementation here

    while True:
        V = policy_evaluation(mdp, policy, discount, theta)
        updated_policy = policy_improvement(mdp , V , discount)

        if np.array_equal(policy , updated_policy):
            break

        policy = updated_policy.copy()
    return V, policy

def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # TODO: Write your implementation here
    while True:
        updated_V = V.copy()
        for state in range(mdp.num_states):
            Q = np.zeros(mdp.num_actions)
            for action in range(mdp.num_actions):
                for prob , target_state , reward , is_done in mdp.P[state][action]:
                    Q[action] += prob * (reward + discount * V[target_state])

            updated_V[state] = np.max(Q)

        if np.max(np.abs(V - updated_V)) < theta:
            break

        V = updated_V.copy()



    # Get the greedy policy w.r.t the calculated value function
    policy = policy_improvement(mdp, V)
    
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    discount = 0.99
    theta = 0.01

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    # Do one step of policy evaluation and print
    print('----------------------------------------------')
    print('One step of policy evaluation (random policy):')
    print('----------------------------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)