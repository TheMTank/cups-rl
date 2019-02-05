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
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
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
        info = None
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
    Wraps ai2thor gym environment to execute history_length steps every time its step function
    is called
    :param environment:
    :return:
    """
    def __init__(self, env, n_steps, device):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.n_steps = n_steps
        self.device = device
        h, w = self.env.config['resolution'][0], self.env.config['resolution'][1]
        self.frame_buffer = torch.zeros((2, h, w), device=self.device, dtype=torch.float32)
        self.state_buffer = deque([], maxlen=n_steps)

    def step(self, action):
        """
        Repeat action n_step times. Regardless of the number of steps, the buffer only stores
        the last 2 frames
        """
        reward, done, info = 0, False, {}
        for t in range(self.n_steps):
            state, reward, done, info = self.env.step(action)
            if t == self.n_steps - 2:
                self.frame_buffer[0] = torch.from_numpy(state)
            elif t == self.n_steps - 1:
                self.frame_buffer[1] = torch.from_numpy(state)
            if done:
                break
        observation = self.frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Return state, reward, done, info
        return torch.stack(list(self.state_buffer), 0), reward, done, info

    def reset(self):
        _ = self.env.reset()
        state, _, _, _ = self.step(self.env.action_space.sample())
        return state
