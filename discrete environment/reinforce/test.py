
def test(agent, env):
    agent.load_model()

    state = env.reset()

    done = False

    Return = 0

    while not done:

         env.render()

         action = agent.take_action(state)
                 
         next_state, reward, done, info = env.step(action)

         state = next_state

         Return += reward

    print("Test Return: {}".format(Return))
    
    env.close()
