import os
import gym
import copy
from gym import spaces
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def _merge(row):
    '''merge the row, there may be some improvement'''
    non_zero = row[row != 0]  # remove zeros
    core = [None]
    score  =0 
    for elem in non_zero:
        if core[-1] is None:
            core[-1] = elem
        elif core[-1] == elem:
            core[-1] = 2 * elem
            score += elem
            core.append(None)
        else:
            core.append(elem)
    if core[-1] is None:
        core.pop()
    return core,score

class CustomEnv(gym.Env):
  
    metadata = {'render.modes': ['human']}

    def __init__(self, size=4, score_to_win=None, rate_2=0.5, random=False, enable_rewrite_board=False):

        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, 4, 1), dtype=np.float32)
        
        self.size = size
        if score_to_win is None:
            score_to_win = np.inf
        self.score_to_win = score_to_win
        self.__rate_2 = rate_2
        if random:
            self.__board = \
                2 ** np.random.randint(1, 10, size=(self.size, self.size))
            self.__end = False
        else:
            self.__board = np.zeros((self.size, self.size))
            # initilize the board (with 2 entries)
            self._maybe_new_entry()
            self._maybe_new_entry()
        self.enable_rewrite_board = enable_rewrite_board
        assert not self.end

    def move(self, direction):
        '''
        direction:
            0: left
            1: down
            2: right
            3: up
        '''
        # treat all direction as left (by rotation)
        board_to_left = np.rot90(self.board, -direction)
        for row in range(self.size):
            core,score = _merge(board_to_left[row])
            board_to_left[row, :len(core)] = core
            board_to_left[row, len(core):] = 0

        # rotation to the original
        self.__board = np.rot90(board_to_left, direction)
        self._maybe_new_entry()

        return score
    def __str__(self):
        board = "State:"
        for row in self.board:
            board += ('\t' + '{:8d}' *
                      self.size + '\n').format(*map(int, row))
        board += "Score: {0:d}".format(self.score)
        return board

    @property
    def board(self):
        '''`NOTE`: Setting board by indexing,
        i.e. board[1,3]=2, will not raise error.'''
        return self.__board.copy()

    @board.setter
    def board(self, x):
        if self.enable_rewrite_board:
            assert self.__board.shape == x.shape
            self.__board = x.astype(self.__board.dtype)
        else:
            print("Disable to rewrite `board` manually.")

    @property
    def score(self):
        return int(self.board.max())

    @property
    def end(self):
        '''
        0: continue
        1: lose
        2: win
        '''
        if self.score >= self.score_to_win:
            return 2
        elif self.__end:
            return 1
        else:
            return 0

    def _maybe_new_entry(self):
        '''maybe set a new entry 2 / 4 according to `rate_2`'''
        where_empty = self._where_empty()
        if where_empty:
            selected = where_empty[np.random.randint(0, len(where_empty))]
            self.__board[selected] = \
                2 if np.random.random() < self.__rate_2 else 4
            self.__end = False
        else:
            self.__end = True

    def _where_empty(self):
        '''return where is empty in the board'''
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):


        prv_state = copy.deepcopy(self.board)
        cur_score = self.move(action)
        

        #reward = (np.log2(int(self.board.max()))-np.log(prv_state.max()))/10
        score_update = np.log2(cur_score+1) / 10.0
        empty_update = max(np.sum(np.array(self.board) == 0) - np.sum(np.array(prv_state) == 0),0) 
        higher_update = np.log2(int(self.board.max())) / 10.0 if int(self.board.max()) != int(prv_state.max()) else 0
        next_state = np.array(self.board).reshape(-1)
        norm_state = np.array([np.log2(i) if i!=0 else 0 for i in next_state]).reshape(self.size,self.size,1)
        reward = score_update + empty_update + higher_update
        dummy = {}        
        return norm_state, reward, bool(self.end) , dummy  #next_state, reward, done
    def reset(self):

        self.__board = np.zeros((self.size, self.size))
        # initilize the board (with 2 entries) 
        self._maybe_new_entry()
        self._maybe_new_entry()

        state = np.array(self.board).reshape(-1)
        norm_state = np.array([np.log2(i) if i!=0 else 0 for i in state]).reshape(self.size,self.size,1)
        

        return norm_state


    #def render(self, mode='human'):

    #def close (self):
    




from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


env = CustomEnv(size=4, score_to_win=None, rate_2=0.5, random=False, enable_rewrite_board=False)
env = DummyVecEnv([lambda: env])

#model = PPO2(MlpPolicy, env, verbose=1)
model = PPO2.load("./PPO3")
model.set_env(env)
model.learn(total_timesteps=3000000)
model.save("./PPO4")

#del model # remove to demonstrate saving and loading

#model = DQN.load("./deepq_2048")

