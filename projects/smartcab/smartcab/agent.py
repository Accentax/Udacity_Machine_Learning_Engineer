import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.iteration=0
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.action_set=[None,'forward','left','right']

        self.previous_state = None
        self.previous_action = None
        self.current_state = None
        self.current_action=None

        self.deadline = self.env.get_deadline(self)

        self.previous_reward = None

        self.destination=None
        #set debug to 1 to print
        self.debug=0

    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        ###########
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing==True:
            self.epsilon=0
            self.alpha=0
        else:

            self.iteration+=1
            self.epsilon=0.9**self.iteration
        self.destination=destination
        self.previous_state = None
        self.previous_action = None
        self.current_state = None

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ###########
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        state= (inputs['light'],
                      inputs['oncoming'],
                      #inputs['right'],
                      #inputs['left'],
                      waypoint)


        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ###########
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        if state not in self.Q.keys():
            maxQ= 0
        else:
            value = [self.Q[state][i] for i in self.Q[state].keys()]
            maxQ= max(value)


        return maxQ


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ###########
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if state not in self.Q.keys():
            self.Q[state]={}
        for i in self.action_set:

            if i not in self.Q[state].keys():

                self.Q[state][i]=0
        return None


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """
        print "in choose"
        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if self.learning!=True:
            action=random.choice(self.action_set)
            print "random action"
        else:
            action=self.policy(state)
            print "state"
            print state

        return action

    #Returns the action with the highest value from state s, or if the value is 0 then returns a random action.
    def policy(self,s):

        print "in policy"
        if random.random()<self.epsilon:
            print "random eps"
            return random.choice(self.action_set)

        elif s not in self.Q.keys():
            print "not in keys Q:"
            print s in self.Q.keys()
            print self.Q.keys()
            return random.choice(self.action_set)
        else:
            # Check for duplicate maxes
            print "getting max Q"
            print self.Q
            print "getting max_value"
            max_value=self.get_maxQ(s)
            duplicates=[i for i in self.Q[s].keys() if self.Q[s][i]==max_value]
            #if there exists several actions with the same value select one at random
            if len(duplicates)>1:

                return random.choice(duplicates)
            #else select the highest


            else:

                max_action= max(self.Q[s],key=self.Q[s].get)
                if self.Q[s][max_action]==0:
                    return random.choice(self.action_set)
                else:
                    return max_action

    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards
            when conducting learning. """

        ###########
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning==True:
            #self.initialize_qDict(state,action)
            #updates the q table using Q function approximation


            currentQ = self.Q[state][action]
            self.Q[state][action] = reward*self.alpha + currentQ*(1-self.alpha)

        else:
            return False



    def update(self):
        """ The update function is called when a time step is completed in the
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return None


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning=True,alpha=0.4 )

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline=True)


    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01,log_metrics=True)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=10,tolerance=0.01)


if __name__ == '__main__':
    run()
