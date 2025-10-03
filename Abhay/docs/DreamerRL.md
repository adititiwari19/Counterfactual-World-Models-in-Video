# Dreamer RL

## Two Models
1. **RSSM** – Encodes and understands the environment  
2. **Actor-Critic** – Chooses actions and evaluates them

---

## RSSM

- **Deterministic hidden state**: Keeps track of what the model knows about the environment so far.  
- **GRU/RNN**: Updates the hidden state using the current hidden state, the stochastic state, and the current action, capturing how the environment evolves over time.

### Prior model
- Predicts the possible next state based on the hidden state only.  
- Used during **imagined rollouts** when we don’t have actual observations.  

### Posterior model
- Predicts the next state using the hidden state **and the actual observation**.  
- Used during training to learn a better representation of the environment.  

### Stochastic state
- Represents uncertainty about the environment.  
- Sampled from either the prior (imagining) or posterior (training with observation).  

### Combined state
- Simply the hidden state + stochastic state.  
- Used to predict observations, rewards, and to guide the actor/critic.

### Observation model
- Tries to reconstruct what the agent would observe.  
- Helps the model learn meaningful representations of the environment.  

### Reward model
- Predicts expected reward from the combined state.  
- Used by the actor-critic during imagined rollouts when real rewards aren’t available.  

---

## Actor-Critic

### Actor
- Decides which action to take based on imagined future rewards.  
- Learns to pick actions that are likely to lead to better outcomes.

### Critic
- Estimates how good a state (or action) is in terms of future rewards.  
- Helps the actor learn by telling it whether its actions are better or worse than expected.  