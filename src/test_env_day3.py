from src.env.fulfillment_env import FulfillmentRoutingEnv

if __name__ == "__main__":
    env = FulfillmentRoutingEnv()
    obs, info = env.reset()
    print("reset info:", info)
    print("obs shape:", obs.shape)

    total_reward = 0.0
    for t in range(20):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if t < 3:
            print("t", t, "action", action, "policy", info["policy"], "step_cost", info["step_cost"])
        if done:
            break

    print("total_reward (20 steps):", total_reward)
