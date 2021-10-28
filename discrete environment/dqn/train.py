
def train(agent, env, episodes=1000):
    rewards_per_episode = []

    render = False

    for episode in range(episodes):
        Return = 0

        state = env.reset()

        done = False

        while not done:

            #env.render()
                 
            action = agent.take_action(state)
                 
            next_state, reward, done, info = env.step(action)

            agent.store_in_memory(state, action, reward, next_state, done)
                 
            state = next_state
                 
            Return += reward

            agent.train_network()

        rewards_per_episode.append(Return)

        print("Iteration:{}, Return: {:0.2f}".format(episode, Return))

    agent.save_model()

    return rewards_per_episode
