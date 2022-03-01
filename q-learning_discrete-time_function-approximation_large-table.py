import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Q-table variables
# Actions
ACTIONS_NUMBER = 3
# X axes
X_BUCKETS = 10
x_axes = np.linspace(start=0, stop=X_BUCKETS, num=10, endpoint=False)

# Q-learning settings
EPISODES_LIMIT = 200_000 + 1
SHOW_EVERY = 10_000
LEARNING_RATE = 0.5
DISCOUNT = 0.5
Q_VALUE_MIN = -10
Q_VALUE_MAX = -10
epsilon = 0.5
EPS_DECAY = 0.99999 # Every episode will be epsilon*EPS_DECAY


class learning_agent:
    def __init__(self):
        self.y_variation = 0 # unuseful line
    def action(self, choice, previous_y, next_y):
        if choice == 0:
            next_y = previous_y # keep same y
        elif choice == 1:
            next_y = previous_y -1 # decrease y
        elif choice == 2:
            next_y = previous_y + 1 # increase y
        return next_y

# Goal function
y_goal = np.array([0, 1, 3, 3, 3, 5, 7, 10, 10, 11])
# Plot goal function
plt.plot(x_axes, y_goal)
plt.title("goal function")
plt.ylabel("x axes")
plt.xlabel("y axes")
plt.show()

# Q-table: random initialization
q_table = np.random.uniform(low=Q_VALUE_MIN, high=Q_VALUE_MAX, size=(ACTIONS_NUMBER,X_BUCKETS))

# Q-learning process
rewards_history = []
approximator = learning_agent()
y_history = np.zeros(X_BUCKETS)

for episode in range(EPISODES_LIMIT):
    # Reset these temporary values every episode
    x_bucket = 0
    current_x_bucket = x_bucket
    episode_reward = 0

    while x_bucket < (X_BUCKETS-1):
        if np.random.random() > epsilon: # exploiting action
            action = np.argmax(q_table[:, current_x_bucket])
        else: # explorative action
            action = np.random.randint(0, ACTIONS_NUMBER)
    
        # Current state
        current_x_bucket = x_bucket
        y_deviation = y_goal[current_x_bucket] - y_history[current_x_bucket]

        # Take the action!
        new_y = approximator.action(action, y_history[current_x_bucket], y_history[current_x_bucket+1])
        x_bucket += 1 # Increment for the while loop

        # New state
        new_x_bucket = current_x_bucket + 1
        y_history[new_x_bucket] = new_y

        # Evaluate reward for this time step.
        reward = -abs(y_deviation)
        episode_reward += reward
        if reward == 0:
            action = 0
            reward = 100
    
        # Evaluate new Q-value
        max_future_q = np.max(q_table[:, new_x_bucket])
        current_q = q_table[action, current_x_bucket]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[action, current_x_bucket] = new_q
    if episode % SHOW_EVERY == 0:
        print(f"On episode #{episode}, epsilon is {epsilon}")
        plt.plot(x_axes, y_goal)
        plt.plot(x_axes, y_history)
        plt.title("goal function")
        plt.ylabel("y axes")
        plt.xlabel("x axes")
        plt.show()
        plt.pause(1)
        plt.clf()
    epsilon *= EPS_DECAY
    rewards_history.append(episode_reward)
# Moving average of reward history
moving_avg = np.convolve(rewards_history, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
# Plot reward history
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.title("Moving average of reward history")
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
plt.pause(3)
