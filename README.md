# ai2thor-experiments

We created an OpenAI gym style wrapper for [ai2thor](http://ai2thor.allenai.org/) and show a random walk example below.
We intend to experiment with different model-free and model-based algorithms within this repo with natural language 
interfaces (e.g. "Pick up the cup", "Open the microwave") and work on representations which can transfer between tasks and environments.


# How to run

`python random_walk.py`

Simpler version below:

```
env = ThorWrapperEnv()
for episode in range(20):
    s = env.reset()
    for t in range(1000):
        a = random.randint(0, len(env.ACTION_SPACE) - 1)
        s, r, done = env.step(a)
        if done:
            break
```
