# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:58:49 2023

@author: Graghic
"""

from pyamaze import agent 
from pyamaze import maze 
import pandas as pd
import numpy as np 




class MazeSolver():
    def __init__(self , lr, exp_rat  , gamma , M , start , end , epochs ):
        #here we will intialize all class attributes
        #start and end are tuples
        self.epochs = epochs # number of epochs used in training process 
        self.lr = lr # laerning rate
        self.exp_rat = exp_rat #exploration rate
        self.gamma = gamma #discount rate
        self.maze = M # maze object form pyamaze library
        self.start = start #tuple represent start cell
        self.end = end #tuple represent final cell
        self.actions = ['E', 'W', 'N', 'S'] # all actions can be taken
        self.states = self.maze.grid # list of tuples represent all states (cells) of maze
        #we will make q table in form of pandas datafram to enhace readablity and take benefits of pandas
        #datafram such as writing to csv file
        #q_table initial values are zeros
        self.q_table = pd.DataFrame(data = np.zeros(shape = (len(self.states),len(self.actions))) ,columns = self.actions,
                                    index = [str(state) for state in self.states])
        #this will define the rewards for all actions
        #columns are actions and rows are states
        #initialization of rewards will be done by rewards_finder method
        self.rewards = pd.DataFrame(data = self.rewards_finder() ,columns = self.actions,
                                    index = [str(state) for state in self.states])
        
        
    def rewards_finder(self):
        """
        will used to initalize rewards table

        Returns
        -------
        TYPE 2d array represent rewards 
        
            rows represent states and columns represent actions.

        """
        rewards = []#empty list will store rewards
        #maze_mape is a ditionary of dictionaries in which each state[key of outer] has a dictionary of all actions and
        #its allowed actions. allowed action will have value of 1 and not allowed action will have a value of 0
        for key in self.maze.maze_map.keys():#to loop over all states
            temp_state_reward = [] # will store all rewards of current state
            for sub_key in self.maze.maze_map[key].keys():#to loop over all actions of the state
                temp_state_reward.append(self.maze.maze_map[key][sub_key])
            rewards.append(temp_state_reward) 
        return np.array(rewards) #2d array rows represent states and columns represent actions
        
    def action_getter(self,current_state):
        """
        will used to get action depend on the current_state from the available actions

        Returns
        -------
        a character represent the action should be taken
        """
        #current_state is tuple
        #th
        
        current_state_actions = self.rewards.loc[str(current_state)].values#all actions
        available_action_indexes = np.where((current_state_actions == 1 ) | ( current_state_actions == 20))[0]
        
        available_actions = [self.actions[int(a)] for a in available_action_indexes ]
        
        action = np.random.choice(available_actions)#get action randomly
         
        return action
    
    def state_updater(self,current_state , action):
        """
        this function used to get the next_state from the current_state and the action
        paramters:
            current_state: tuple [row,col]
            action : character one of ['E','N','S','W']
        """
        # the start of our maze is the upper most left cell
        # 'E' and 'W' change the column
        # 'E' increase the col 'W' decrease the col
        
        # 'S' and 'N' change the row
        # 'S' will increas the row and 'N' will decrease the row
        if action == 'E':
            next_state = (current_state[0] , current_state[1]+1)
        elif action == 'W':
            next_state = (current_state[0] , current_state[1]-1)
        elif action == 'N':
            next_state = (current_state[0]-1 , current_state[1])
        elif action == 'S':
            next_state = (current_state[0]+1 , current_state[1])
        return next_state
        
    def train(self):
        """
        this function will be used in training [update q_table values]
        will take no thing 
        return no thing just modify q_table attribute
        """
        for epoch in range(self.epochs):# will loop up to number of epochs
            iter_epoch = 0 #to store number of iteration per epochs
            current_state = self.start
            final_state = self.end
            while (current_state != final_state ):#loop until we get the final_state
                iter_epoch += 1
                action = self.action_getter(current_state)#get action suitable for this state
                reward = self.rewards.loc[str(current_state)][action]#get the reward for the current state and current action
                next_state = self.state_updater(current_state, action)#get the next state from the current state and current action
                if next_state == final_state: #check whether we are in the final state or not
                    reward *=  20 #this line will make the reward of the final state more than usual rewards
                    
                #start of Bellman Equation
                max_q_next_state = self.q_table.loc[str(next_state)].max()
                q_current_state_action = self.q_table.loc[str(current_state)][action]
                
                self.q_table.loc[str(current_state)][action] = \
                q_current_state_action + \
                self.lr*(reward + self.gamma * max_q_next_state - q_current_state_action )
                #end of Bellman Equation
                #check if we reached the final state to decide whether end epoch or not
                if next_state == final_state:
                    break
                else:
                    current_state = next_state
            print(f"{iter_epoch +1 } iteration for epoch {epoch+1}")
            #print(f"epoch number {epoch + 1} iterations {iter_epoch}")
    def path_finder(self):
        path = [self.start]#list will store the states from start_state to the final_state
        temp_state = self.start 
        while (temp_state != self.end) :
            action = self.q_table.loc[str(temp_state)].idxmax()#get the highest q value action of temp_state
            next_state = self.state_updater(temp_state,action)#get the next state
            path.append(next_state)
            temp_state = next_state
        
        path.append(next_state)
        return path
    def start_to_end_actions(self , path):
        #this function will be used to get all actions required to reach from the start point to the final point
        #path is list of tuples
        actions = []#will store all actions required to reach from start to end
        for i in range(len(path)-1):#to loop over all states in path list
            current_state = path[i]
            next_state = path[i+1]
            #our start is the most top left cell
            if current_state[0] > next_state[0]:#row decrease
                actions.append('N')
            if current_state[0] < next_state[0]:#row increase
                actions.append('S')
            if current_state[1] > next_state[1]:#column decrease
                actions.append('W')
            if current_state[1] < next_state[1]:#column increase
                actions.append('E')
        return actions
#******************************************************************************
#******************************************************************************
#       Start of functions Will be Used to convert from 
#            ['E','N','S','W'] to ['F','B','L' ,'R']
# this conversion will depend on the north direction

def north_updater(usual_action , north):
    #this function will return new north depend on the previouse north and taken action
    #parameters :
    # usual_action: taked action , north: north at the usual action
    #return : 
    # updated north after taking usual_action with the north 
    #if action is B or F north does not change
    if north == 'F' and usual_action == 'F' :
        new_north = 'F'
    if north == 'F' and usual_action == 'B' :
        new_north = 'F'
    if north == 'F' and usual_action == 'R' :
        new_north = 'L'
    if north == 'F' and usual_action == 'L' :
        new_north = 'R'
    ###############################################
    if north == 'B' and usual_action == 'F' :
        new_north = 'B'
    if north == 'B' and usual_action == 'B' :
        new_north = 'B'
    if north == 'B' and usual_action == 'R' :
        new_north = 'R'
    if north == 'B' and usual_action == 'L' :
        new_north = 'L'
    ##################################################
    if north == 'R' and usual_action == 'F' :
        new_north = 'R'
    if north == 'R' and usual_action == 'B' :
        new_north = 'R'
    if north == 'R' and usual_action == 'R' :
        new_north = 'F'
    if north == 'R' and usual_action == 'L' :
        new_north = 'B'
    ################################################
    if north == 'L' and usual_action == 'F' :
        new_north = 'L'
    if north == 'L' and usual_action == 'B' :
        new_north = 'L'
    if north == 'L' and usual_action == 'R' :
        new_north = 'B'
    if north == 'L' and usual_action == 'L' :
        new_north = 'F'
        
    return new_north
    
def reverse_direction(direction):
    if direction == 'R':
        return 'L'
    elif direction == 'L':
        return 'R'
    elif direction == 'F':
        return 'B'
    elif direction == 'B':
        return 'F'
def compass_direction_to_usual_direction(comp_action,north = 'F'):
    """
    this functtion will convert from ['E','N','S','W'] to ['F','B','L' ,'R']
    parameters:
        # it will recieve compase direction and
        # where is the north of robot for example may be the north is in front or backward 
        or right or left of the robot
    return:
        #usual direction which is one of ['F','B','L' ,'R']
        #new north direction 
    """
    #IMPORTANT NOTE : CODE DEPEND ON THE FOLLOWING ORDER OF COMP_ACTION AND USUAL_ACTION LISTS
    #DO NOT CHANGE THE FOLLOWING ORDER
    comp_actions = np.array(['E','W','N','S'])
    usual_actions = np.array(['R','L','F','B' ])
    
    if north == 'F':
        ##same index
        usual_action = usual_actions[np.where(comp_actions == comp_action)[0][0]]
        
        
    elif north == 'B':
        usual_action = usual_actions[np.where(comp_actions == comp_action)[0][0]]
        usual_action = reverse_direction(usual_action)
        
    elif north == 'R':
        if comp_action == 'N' or comp_action == 'S':
            usual_action = usual_actions[np.where(comp_actions == comp_action)[0][0]-2]
            
        elif comp_action == 'W':
            usual_action = 'F'
            
        elif comp_action == 'E':
            usual_action = 'B'
    elif north == 'L':
        if comp_action == 'N':
            usual_action = 'L'
        elif comp_action == 'S':
            usual_action = 'R'
        elif comp_action == 'E':
            usual_action = 'F'
        elif comp_action == 'W':
            usual_action = 'B'
    
    new_north = north_updater(usual_action , north)
    print(f"{usual_action} ==> north {new_north}")
    return usual_action , new_north 
    
    
if __name__ == '__main__':
    lr = 0.6 #learning rate 
    gamma = 0.7 #discount rate
    exp_rat = 0.1 #exploration rate
    start_state = (1,1) #start point
    end_state = (5,5) # end point
    #determine maze size
    rows = 5 #number of rows
    columns = 5 #number of columns
    
    epochs = 10
    m=maze(rows,columns)
    m.CreateMaze(x= end_state[0],y=end_state[1] ,loopPercent = 100)
    
    solver = MazeSolver(lr, exp_rat, gamma, m, start_state, end_state, epochs)
    solver.train()  
    
    path = solver.path_finder()
    print(f"path states {path}")
    path_actions = solver.start_to_end_actions(path)
    print(f"path compass actions {path_actions}")
    usual_actions = []
    north = 'F'
    for action in path_actions:
        action , north =  compass_direction_to_usual_direction(action,north )
        usual_actions.append(action)
    print(f"path usual acttions {usual_actions}")
    a=agent(m,x =start_state[0] , y =start_state[1] ,footprints=True)
    m.tracePath({a:path})
    m.run()
