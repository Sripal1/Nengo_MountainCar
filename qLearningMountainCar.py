import gym
import numpy as np
import time

env = gym.make("MountainCar-v0")

learningRate = 0.1 # how much we want to update our Q-table by every time we take a new action (0 to 1)
discount = 0.95 # how important do we find future actions (0 to 1)
episodes = 25000 # how many times we want to run the environment from the beginning

showFreq = 2000
timeout = 0.5

Discrete_os_size = [20]*len(env.observation_space.high)
Discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/Discrete_os_size    
q_table = np.random.uniform(low = -2, high = 0, size = (Discrete_os_size + [env.action_space.n])) # creates a (20x20)x3 matrix of random numbers between -2 and 0

def getDiscreteState(state): 
    discreteState = (state - env.observation_space.low)/Discrete_os_win_size
    return tuple(discreteState.astype(np.int32))

for episode in range(1,episodes): 
    if episode%showFreq == 0:
        print(episode)
    
    discreteState = getDiscreteState(env.reset()[0])
    # print(discreteState)
    # print(q_table[discreteState]) # prints the q-table values for the current state
    # print(np.argmax(q_table[discreteState])) # prints the index of the highest value in the q-table for the current state (action to be taken)

    done = False # boolean to check if the episode is done
    start_time = time.time()

    while not done:
        action = np.argmax(q_table[discreteState]) # action to be taken (0, 1 or 2)
        newState, reward, done, _, _ = env.step(action)
        newDiscreteState = getDiscreteState(newState)

        # if episode % showFreq == 0:
        #     env.render()
        
        if not done:
            max_future_q = np.max(q_table[newDiscreteState]) # max Q value for the new state (action to be taken)
            current_q = q_table[discreteState + (action, )] # current Q value for the current state (action taken)-> only one value instead of 3 values
            new_q = (1-learningRate)*current_q+learningRate*(reward+discount*max_future_q) # formula for updating the Q-table
            q_table[discreteState + (action, )] = new_q # updating the Q-table

        elif newState[0]>=env.goal_position:
            print(f"Successful on EPISODE{episode}")
            q_table[discreteState + (action, )] = 0 # if we reach the goal, we set the Q value to 0
            # env.render()

        discreteState = newDiscreteState # updating the current state to the new state after taking the action (moving to the next state)

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"Episode {episode} took too long. Exiting...")
            done = True  # Exit the episode loop

env.close()