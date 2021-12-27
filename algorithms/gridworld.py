import math
import time
import copy
import random
import operator
import numpy as np
from numpy import linalg as LA


class GridWorld:
    def __init__(self):
      self.grid = self.defineGrid()
      self.gh = 10
      self.gw = 10

    def cell(self, s):
      return self.grid[s[0]][s[1]]
    
    def defineGrid(self):
      grid = [['0' for col in range(10)]
             for row in range(10)]

      for cell in wc:
        grid[cell[0]][cell[1]] = '_'
      
      for cell in nrc:
        grid[cell[0]][cell[1]] = '-1'
      
      for cell in prc:
        grid[cell[0]][cell[1]] = '1'
      
      return grid
    
class Policy:
    def __init__(self, world, start, goal, beta, alpha):
        self.world = world
        self.start = copy.deepcopy(start)
        self.goal = copy.deepcopy(goal)
        self.pos = copy.deepcopy(start)
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.Q = np.zeros((100, 4))
        self.reward_matrix = np.zeros((world.gh, world.gw))
        self.beta = beta
        self.alpha = alpha
        
    def reset(self):
        self.pos = copy.deepcopy(self.start)

    def nextState(self, action):
      # Choose next state with current state and possible actions
        s = copy.deepcopy(self.pos)
        if action == 'UP':
            if s[0] != 0: 
              s[0] -= 1
        if action == 'RIGHT':
            if s[1] != (self.world.gw - 1):
              s[1] += 1
        if action == 'DOWN':
            if s[0] != (self.world.gh - 1): 
              s[0] += 1
        if action == 'LEFT':
            if s[1] != 0: 
              s[1] -= 1

        # check for wall
        if self.world.cell(s) == '_':
            s = self.pos

        return s

    # Choose the action that gives maximum reward
    def maxRewardAction(self, pos):
        pos_index = pos[0] * self.world.gw + pos[1]
        moves = self.Q[pos_index]

        action_dict = {}
        for i, a in enumerate(moves):
            action_dict[i] = a
        actions = list(action_dict.items())
        random.shuffle(actions)

        next_move = -1
        highest_move_index = -1
        for i, a in actions:
            if a > next_move:
                next_move = a
                highest_move_index = i

        return highest_move_index, next_move

    def Qval(self, s):
        return s[0] * self.world.gh + s[1]

    # return reward for being in state s
    def reward(self, s):    
        x = s[0]
        y = s[1]
        self.reward_matrix[x][y] = float(self.world.grid[x][y])
        return self.reward_matrix[x][y]
 
class EpsilonGreedyPolicy(Policy):
    def __init__(self, world, start, goal, beta, alpha, epsilon):
        Policy.__init__(self, world, start, goal, beta, alpha)
        self.epsilon = epsilon

    def next(self):
        curr_state = copy.deepcopy(self.pos)
        s = self.Qval(self.pos)
        moves = self.Q[s]

        # Find best action
        ns, temp = self.maxRewardAction(curr_state)
        if random.uniform(0, 1) < self.epsilon: 
          next_move = random.randint(0, 3)
        else: 
          next_move = ns

        action = self.actions[next_move]    
        self.pos = self.nextState(action)   

        reward = self.reward(self.pos)            
        optimal_i, optimal_q = self.maxRewardAction(self.pos)
        curr_q = self.Q[s][next_move]  
        x = self.alpha * (reward + self.beta * optimal_q - curr_q)
        # update Q
        self.Q[s][next_move] += x
        # Reward 1 reached (goal state)
        if self.pos == self.goal: 
          return False
        else: 
          return True

    def play(self):
      policies = []
      convergence_thresh = 0  
      ts = time.perf_counter()
      num_iter = 0
      conv_count = 1000
      total_time = 0
      while True:
          num_iter += 1
          last_q_matrix = copy.deepcopy(self.Q)

          while self.next() == True: 
            pass
          self.reset()

          if matrix_diff(last_q_matrix, self.Q) <= convergence_thresh:
              conv_count -= 1
              if conv_count == 0: 
                break
          # Reset
          else: 
            conv_count = 1000
      
      te = time.perf_counter()
      total_time += te - ts
      policies.append(policy)
      print(("Time taken {0:.2f} seconds, ran for {1} " +
            "iterations").format((te - ts), (num_iter - 1000)))
      print("Q values for Epsilon Greedy", self.epsilon)
      print_policy_Q(policy, self.epsilon, 'Epsilon')
     
    
class BoltzmannExplorationPolicy(Policy):
    def __init__(self, world, start, goal, beta, alpha, T):
        Policy.__init__(self, world, start, goal, beta, alpha)
        self.T = T
        self.currT = T

    def reset(self):
        if self.T >= 1: 
          self.T -= 1
        Policy.reset(self)

    def next(self):
        curr_state = copy.deepcopy(self.pos)
        s = self.Qval(self.pos)
        moves = self.Q[s]
        
        if self.T > 0:
            action_probs_numes = []
            denom = 0
            for m in moves:
                val = math.exp(m / self.T)
                action_probs_numes.append(val)
                denom += val
            action_probs = [x / denom for x in action_probs_numes]

            rand_val = random.uniform(0, 1)
            prob_sum = 0
            for i, prob in enumerate(action_probs):
                prob_sum += prob
                if rand_val <= prob_sum:
                    picked_move = i
                    break
        else:
            picked_move, picked_move_q = self.maxRewardAction(curr_state)

        action = self.actions[picked_move]   
        self.pos = self.nextState(action)
        reward = self.reward(self.pos)          
        opt_future_i, optimal_q = self.maxRewardAction(self.pos)
        curr_q = self.Q[s][picked_move]
        x = self.alpha * (reward + self.beta * optimal_q - curr_q)
        self.Q[s][picked_move] += x


        if self.pos == self.goal: 
          return False
        else: 
          return True

    def play(self):
      policies = []
      tab = " "
      convergence_thresh = 0  
      converge_count = 1000 
      ts = time.perf_counter()
      num_iter = 0
      this_conv_count = converge_count
      total_time = 0
      while True:
          num_iter += 1
          last_q_matrix = copy.deepcopy(self.Q)

          while self.next() == True: 
            pass
          self.reset()

          if matrix_diff(last_q_matrix, self.Q) <= convergence_thresh:
              this_conv_count -= 1
              if this_conv_count == 0: 
                break
          # Reset
          else: 
            this_conv_count = converge_count
      
      te = time.perf_counter()
      total_time += te - ts
      print(("Time taken {0:.2f} seconds, ran for {1} " +
            "iterations").format((te - ts), (num_iter - converge_count)))
      print("Q values for Boltzman", self.currT)
      print_policy_Q(policy, self.currT, 'Boltzman')
      policies.append(policy)
      
def print_policy_Q(policy, name, type):
    this_str = ""
    for row in policy.Q:
        for i, col in enumerate(row):
            this_str += str(col)
            if i != (len(row) - 1): this_str += ","
        this_str += "\n"
    file_name = 'Q_' + type + str(name) + '.csv'
    with open(file_name, "w") as file:
        file.write(this_str)
    return True

def matrix_diff(matrix_one, matrix_two):
    total_diff = 0
    for i, row in enumerate(matrix_one):
        index_one, value_one = max(enumerate(matrix_one[i]),
                                   key=operator.itemgetter(1))
        index_two, value_two = max(enumerate(matrix_two[i]),
                                   key=operator.itemgetter(1))
        if index_one != index_two: total_diff += 1    
    return total_diff

wc = [(2,1),(2,2),(2,3),(2,4), (2,6), (2,7), (2,8), (3,4), (4,4), (5,4), (6,4), (7,4)]
nrc = [(3,3),(4,5),(4,6),(5,6),(5,8),(6,8),(7,3),(7,5),(7,6)]
prc = [(5,5)]

epsilon =  [0.1, 0.2, 0.3]
world = GridWorld()
for e in epsilon:
  policy = EpsilonGreedyPolicy(world, [0, 0], [5, 5], 0.9, 0.01, e)
  policy.play()

temperature = [2000, 1000, 100, 10, 1]
for t in temperature:
  policy = BoltzmannExplorationPolicy(world, [0, 0], [5, 5], 0.9, 0.01, t)
  policy.play()

