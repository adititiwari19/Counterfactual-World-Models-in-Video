# Countefactual Reasoning

[Paper Link](https://arxiv.org/pdf/2506.05188)

In the case of linear regression through the origin (no intercept), we try to predict

$$Y = \beta * X + U_Y$$

Where **$\beta$** is our estimated slope and **$U_Y$** is our noise (error).

Now to model counterfactual reasoning by changing **$X$** to **$X^{CF}$**, we use:

$$Y^{CF} = \beta * X^{CF} + U_Y$$

In other words, we preserve the **$U_Y$** (in other words, we abduct the noise).

---

### Analogy

The counterfactual query asks:  
*"What if you had studied for 5 hours instead, but kept your specific luck ($U_Y = 1$)?"*

---

### In the context of transformers:

This ability can emerge in transformers under the right training conditions.

- **Attention** = the key. It lets the model “point” at the factual observation referenced by *z*, something RNNs or MLPs struggle with.  
- **Depth** = gives room to compose operations (copy → subtract → multiply → add).  
- **Copying mechanism** = induction heads allow residual noise to be carried forward.  

---

### Key Outcome

It allows the Transformer to answer *“what if”* questions by reusing the hidden randomness from a factual example, instead of just predicting an average outcome. In other words, it reuses the specific hidden randomness from a factual example to imagine what would happen under a different scenario, instead of just predicting an average outcome.