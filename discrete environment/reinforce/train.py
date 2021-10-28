
def train(agent, env, episodes=1000):
    render = False
    
    rewards_per_episode = []

    for episode in range(episodes):
        Return = 0

        state = env.reset()

        done = False

        while not done:

            #env.render()
                 
            action = agent.take_action(state)
                 
            next_state, reward, done, info = env.step(action)

            agent.store_in_memory(state, action, reward)
                 
            state = next_state
                 
            Return += reward
        
        rewards_per_episode.append(Return)
        
        agent.train_network()

        print("Iteration:{}, Return: {:0.2f}".format(episode, Return))

    agent.save_model()

    return rewards_per_episode
