import matplotlib.pyplot as plt
import numpy as np

def value_plot(agent):
    V = agent.V
    policy = agent.policy
    env_map = agent.env.map

    plt.figure(figsize=(7,7))
    plt.imshow(V, cmap='viridis', interpolation='none')
    ax = plt.gca()
    ax.set_xticks(np.arange(V.shape[0])-.5)
    ax.set_yticks(np.arange(V.shape[1])-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            # Plot value
            plt.text(x, y, format(V[y, x], '.2f'),
                    color='white', size=12,  verticalalignment='center',
                    horizontalalignment='center', fontweight='bold')
            # Plot map
            plt.text(x-0.25, y-0.25, str(env_map[y][x]),
                    color='white', size=12,  verticalalignment='center',
                    horizontalalignment='center', fontweight='bold')
            # Plot policy
            for i, prob in enumerate(policy[y, x]):
                if prob == 0.0:
                    continue
                dx = 0.0
                dy = 0.0
                if i == 0: # Up
                    dy = -prob / 2.0
                elif i == 1: # Right
                    dx = prob / 2.0
                elif i == 2: # Down
                    dy = prob / 2.0
                elif i == 3: # Left
                    dx = -prob / 2.0
                plt.arrow(x, y, dx, dy, width=0.01, color='black', length_includes_head=True)

    plt.grid(color='black', lw=1, ls='-')
    plt.colorbar()
    plt.show()

def action_value_plot(agent):
    Q = agent.Q
    env_map = agent.env.map

    plt.figure(figsize=(7,7))
    plt.imshow(np.max(Q, axis=2), cmap='viridis', interpolation='none')
    ax = plt.gca()
    ax.set_xticks(np.arange(Q.shape[0])-.5)
    ax.set_yticks(np.arange(Q.shape[1])-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for y in range(Q.shape[0]):
        for x in range(Q.shape[1]):
            # Plot Q-values
            for i, q in enumerate(Q[y, x]):
                dx = 0.0
                dy = 0.0
                if i == 0: # Up
                    dy = -0.3
                elif i == 1: # Right
                    dx = 0.3 
                elif i == 2: # Down
                    dy = 0.3 
                elif i == 3: # Left
                    dx = -0.3
                plt.text(x+dx, y+dy, format(q, '.2f'),
                        color='white', size=12,  verticalalignment='center',
                        horizontalalignment='center', fontweight='bold')
            # Plot map
            plt.text(x-0.25, y-0.25, str(env_map[y][x]),
                    color='white', size=12,  verticalalignment='center',
                    horizontalalignment='center', fontweight='bold')
            # Plot policy
            max_q_actions = np.argwhere(Q[y, x] == np.amax(Q[y, x])).flatten()
            policy = np.zeros(Q[y,x].shape, dtype=np.float32)
            for a in max_q_actions:
                policy[a] = 1.0/len(max_q_actions)
            for i, prob in enumerate(policy):
                if prob == 0.0:
                    continue
                dx = 0.0
                dy = 0.0
                if i == 0: # Up
                    dy = -prob / 2.0
                elif i == 1: # Right
                    dx = prob / 2.0
                elif i == 2: # Down
                    dy = prob / 2.0
                elif i == 3: # Left
                    dx = -prob / 2.0
                plt.arrow(x, y, dx, dy, width=0.01, color='black', length_includes_head=True)

    plt.grid(color='black', lw=1, ls='-')
    plt.colorbar()
    plt.show()

def test_agent(agent, env, epsilon):
    obs = env.reset()
    env.render()

    while True:
        action = agent.action(obs, epsilon=epsilon)
        obs, reward, done, info = env.step(action)

        env.render()

        if done:
            break