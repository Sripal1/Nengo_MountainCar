import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode = "human")
# env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

learningRate = 0.1 # how much we want to update our Q-table by every time we take a new action (0 to 1)
discount = 0.95 # how important do we find future actions (0 to 1)
episodes = 25000 # how many times we want to run the environment from the beginning

Discrete_os_size = [20]*len(env.observation_space.high)
Discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/Discrete_os_size
# print(Discrete_os_win_size)

q_table = np.random.uniform(low = -2, high = 0, size = (Discrete_os_size + [env.action_space.n])) # creates a 20x20x3 matrix of random numbers between -2 and 0
# # print(q_table.shape)
# # print(q_table)

def getDiscreteState(state):
    discreteState = (state - env.observation_space.low)/Discrete_os_win_size
    return tuple(discreteState.astype(np.int32))

discreteState = getDiscreteState(env.reset()[0])
print(discreteState)
print(q_table[discreteState]) # prints the q-table values for the current state
print(np.argmax(q_table[discreteState])) # prints the index of the highest value in the q-table for the current state (action to be taken)

done = False # boolean to check if the episode is done

while not done:
    action = np.argmax(q_table[discreteState])
    newState, reward, done, _, _ = env.step(action)
    newDiscreteState = getDiscreteState(newState)
    q_table[newDiscreteState]
    # print(newState)
    env.render()
    
    if not done:
        max_future_q = np.max(q_table[newDiscreteState]) # max Q value for the new state (action to be taken)
        current_q = q_table[discreteState + (action, )] # current Q value for the current state (action taken)-> only one value instead of 3 values
        new_q = (1-learningRate)*current_q+learningRate*(reward+discount*max_future_q) # formula for updating the Q-table
        q_table[discreteState + (action, )] = new_q # updating the Q-table

    elif newState[0]>=env.goal_position:
        q_table[discreteState + (action, )] = 0 # if we reach the goal, we set the Q value to 0

    discreteState = newDiscreteState # updating the current state to the new state after taking the action (moving to the next state)

env.close()