import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import copy
car_stat={}

g_epsilon=0.1
g_alpha=0
g_gamma=0
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

        self.action_set=[None,'forward','left','right']
        self.qDict={}
        global g_alpha
        self.alpha    = g_alpha
        global g_gamma
        self.gamma    = g_gamma
        global g_epsilon
        self.epsilon  =g_epsilon

        self.previous_state = None
        self.previous_action = None
        self.current_state = None
        self.current_action=None

        self.deadline = self.env.get_deadline(self)
        self.total_time=copy.deepcopy(self.deadline)
        self.previous_reward = None

        self.destination=None
        #set debug to 1 to print
        self.debug=0
        global car_stat
        car_stat[(self.gamma,self.epsilon,self.alpha)]={}
        car_stat[(self.gamma,self.epsilon,self.alpha)]["success"]=0
        car_stat[(self.gamma,self.epsilon,self.alpha)]["cum_reward"]=0
        car_stat[(self.gamma,self.epsilon,self.alpha)]["time"]=[]




    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destination=destination
        self.previous_state = None
        self.previous_action = None
        self.current_state = None
        self.total_time=copy.deepcopy(self.env.get_deadline(self))



    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        #Update state
        current_env_state = self.env.sense(self)

        self.current_state= (inputs['light'],
                      inputs['oncoming'],
                      #inputs['right'],
                      inputs['left'],
                      self.next_waypoint)


        if self.current_state not in self.qDict.keys():
            self.qDict[self.current_state]={}
        #Select action according to your policy
        #If the current state and action is not in the Q table, add them as 0.
        for possible_action in self.action_set:
            self.initialize_qDict(self.current_state,possible_action)
        #if the first run do a random choice to start off
        if self.previous_state ==None:
            action=random.choice(self.action_set)
        else:
        #then use policy
            action=self.policy(self.current_state)
            if self.debug==1:
                print "doing action",action
        self.current_action=action


        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self.initialize_qDict(self.current_state,action)
        if self.previous_state !=None:
            self.update_qtable(self.current_state,self.previous_state,self.previous_action,self.previous_reward)



        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #logic to save statistics for car parameter optimization
        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]
        global car_stat
        if location==destination:
            car_stat[(self.gamma,self.epsilon,self.alpha)]["success"]+=1
            car_stat[(self.gamma,self.epsilon,self.alpha)]["time"].append(self.total_time-deadline)
        elif deadline==0:
            car_stat[(self.gamma,self.epsilon,self.alpha)]["time"].append(deadline)
        car_stat[(self.gamma,self.epsilon,self.alpha)]["cum_reward"]+=reward




        # store the previous action and state so that we can update the q table on the next iteration
        self.previous_action = action
        self.previous_state = self.current_state
        self.previous_reward = reward

    #Helper function that sets keys in dict if the state and action is new
    def initialize_qDict(self,state,action):
        if state not in self.qDict.keys():
            self.qDict[state]={}
        if action not in self.qDict[state].keys():

            self.qDict[state][action]=0

    #Function that updates the qtable using approximation of the Qfuction by transition and learning rate alpha
    def update_qtable(self,next_state,state,action,reward):
        self.initialize_qDict(state,action)
        #updates the q table using Q function approximation
        self.qDict[state][action]=(1-self.alpha)*self.get_Q(state,action)+self.alpha*(reward+self.gamma*self.get_max_Q(next_state))
        if self.debug==1:
            print "state",state
            print "qDict",self.qDict[state]

    #Returns the value for the Q function given action a and state s
    def get_Q(self,s,a):

        return self.qDict[s][a]

    #Returns the the maximum value for all action from state s
    def get_max_Q(self,s):
        if s not in self.qDict.keys():
            return 0
        else:
            value = [self.qDict[s][i] for i in self.qDict[s].keys()]
            return max(value)

    #Returns the action with the highest value from state s, or if the value is 0 then returns a random action.
    def policy(self,s):
        if random.random()<self.epsilon:
            return random.choice(self.action_set)
        elif s not in self.qDict.keys():
            return random.choice(self.action_set)
        else:
            # Check for duplicate maxes
            max_value=self.get_max_Q(s)
            duplicates=[i for i in self.qDict[s].keys() if self.qDict[s][i]==max_value]
            #if there exists several actions with the same value select one at random
            if len(duplicates)>1:
                if self.debug==1:
                    print "duplicates"
                return random.choice(duplicates)
            #else select the highest


            else:
                if self.debug==1:
                    print "no duplicates"
                max_action= max(self.qDict[s],key=self.qDict[s].get)
                if self.qDict[s][max_action]==0:
                    return random.choice(self.action_set)
                else:
                    return max_action






def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    #open file to write stats
    f = open('carstats.txt','w')
    #do a grid search for gamma and alpha
    import numpy as np
    for a in np.linspace(0.1,0.5,5):
        for g in np.linspace(0.1,1,5):
                g_gamma=g
                g_alpha=a

                run()


    #print car_stat
    print [(k,car_stat[k]["time"]) for k in car_stat.keys() if car_stat[k]["success"]==max([car_stat[i]["success"] for i in car_stat.keys()])]

    print max([car_stat[i]["success"] for i in car_stat.keys()])
    print [(k,np.mean(car_stat[k]["time"]),np.median(car_stat[k]["time"]),np.std(car_stat[k]["time"])) for k in car_stat.keys() if car_stat[k]["success"]==max([car_stat[i]["success"] for i in car_stat.keys()])]
    print max([car_stat[i]["cum_reward"] for i in car_stat.keys()])
    f.write(str(([(k,"mean",np.mean(car_stat[k]["time"]),"median",np.median(car_stat[k]["time"]),"std",np.std(car_stat[k]["time"])) for k in car_stat.keys() if car_stat[k]["success"]==max([car_stat[i]["success"] for i in car_stat.keys()])]
,"max" ,max([car_stat[i]["success"] for i in car_stat.keys()]), max([car_stat[i]["cum_reward"] for i in car_stat.keys()])  , [(k,car_stat[k]["time"]) for k in car_stat.keys() if car_stat[k]["success"]==max([car_stat[i]["success"] for i in car_stat.keys()])]))+'\n')
    f.close()
