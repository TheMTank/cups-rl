"""
Adapted from https://github.com/Kaixhin/Rainbow

Environment wrappers for ATARI_games and ai2thor
"""
from collections import deque
import random
import atari_py
import torch
import cv2  # Note that importing cv2 before torch may cause segfaults?
import gym
from gym import spaces


class Env:
    """
    ATARI games environment definition as from original implementation
    """
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = spaces.Discrete(len(self.actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        """
        Note on atari preprocessing from DeepMind DQN:
        Repeat action 4 times and take only the max pooling over the last two frames. This is done
        to solve the following problems:
        1. Since the frames are very similar, they skip several frames so movement can be perceived
        2. Some atari roms render only every second frame, so in order to not get an empty frame and
         use the same code for all atari games they max pool frames 3 & 4.
        A more detailed explanation from D. Takeshi can be found on:
        https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        """
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done, info = 0, False, None
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if 0 < lives < self.lives:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done, info
        return torch.stack(list(self.state_buffer), 0), reward, done, info

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class MultipleStepsEnv(gym.Wrapper):
    """
    Wraps our ai2thor gym environment to execute history_length steps every time its step function
    is called. It is meant to be used only to wrap our ai2thor environment. Use Env class from
    this script to load the atari wrapped environments from the original repository.
    """
    def __init__(self, env, frame_stack, device):
        gym.Wrapper.__init__(self, env)
        self.config = env.config
        self.frame_stack = frame_stack
        self.device = device
        self.state_buffer = deque([], maxlen=self.frame_stack)

    def step(self, action):
        """
        We stack n_step frames together so that the CNN can capture the consistency of movements as
        it is done in DQN nature paper for atari environments.
        The frames are stacked one by one using a deque, which means that every step we move only
        the oldest frame is removed and the newest is appended.
        If n_step == 1, we are simply using one frame as the input to our CNN.
        The done and info belong to the last step only.
        """
        while True:
            state, reward, done, info = self.env.step(action)
            observation = torch.from_numpy(state).float().to(self.device)
            self.state_buffer.append(observation)
            if len(self.state_buffer) == self.frame_stack:
                break
        state = torch.cat(list(self.state_buffer), 0)
        # Return state, reward, done, info
        return state, reward, done, info

    def reset(self):
        _ = self.env.reset()
        self.state_buffer = deque([], maxlen=self.frame_stack)
        state, _, _, _ = self.step(self.env.action_space.sample())

        return state
