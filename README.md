# ai2thor-experiments

We created an OpenAI gym style wrapper for [ai2thor](http://ai2thor.allenai.org/) and show a random walk example below.
We intend to experiment with different model-free and model-based algorithms within this repo using a natural language
 interface i.e. the environment returns sentences as part of its state e.g. "Pick up the cup", "Open the microwave". 
 We also will work on representations which can transfer between tasks and environments.

# Requirements

* Python 3.6+
* ai2thor (`pip install ai2thor`)
* skimage

To test to see if all functionality works run:

`cd test`
`python test_ai2thor_wrapper.py`

# How to run

`cd examples`
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
