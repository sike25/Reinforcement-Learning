# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state

          The practical objective of QLearning is to eliminate the simultaneous 
          record of utilities/values, and rely only on the Qs.
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Q Values - a Counter is a dict with default 0
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.

          This is the inverse of the QValueFromValues function
          in valueIterationAgent. 
          The maximum of each legal action's qValue is the 
          utility/value of that state
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0
        maxQValue = -sys.maxsize
        for action in actions:
            qValue = self.getQValue(state, action)
            maxQValue = max(maxQValue, qValue)
        return maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
          The best action produces the greatest Q value
        """
        actions = self.getLegalActions(state)
        # Find the maximum Q value
        if len(actions) == 0:
            return None
        maxQValue = -sys.maxsize
        for action in actions:
            qValue = self.getQValue(state, action)
            maxQValue = max(maxQValue, qValue)
        # Populate list of actions that map to the greatest Q values
        maxActions = []
        for action in actions:
            qValue = self.getQValue(state, action)
            if maxQValue == qValue:
                maxActions.append(action) 
        # Return randomly from the list of actions
        return random.choice(maxActions)
            

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)

          Epsilon-greedy. Epsilon percent of the time, we take a random
          action. The other times, we take the best action.
          This algorithm is for striking a balance between curious and greedy.
          If we find something good, we want to exploit it, yes, but not at the expense
          of learning new stuff (we have to discover the magic of dance!).
          So sometimes, we do what it best, and other times, we take a chance to explore, and learn
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        if (util.flipCoin(self.epsilon)):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf.
        """
        maxNextQValue = -sys.maxsize
        nextActions = self.getLegalActions(nextState)
        # Find the max Q from each legal action's Q
        for nextAction in nextActions:
            nextQValue = self.getQValue(nextState, nextAction)
            maxNextQValue = max(maxNextQValue, nextQValue)
        # If we are at the fugue terminal state, after the GUI-displayed terminal
        # This state allows our GUI-displayed terminal to be updated.
        if len(nextActions) == 0:
            maxNextQValue = 0
        # Q(s,a) = Q(s,a) + alpha(r + (dmaxQ(s', a') - Q(s,a)))
        self.qValues[(state, action)] = self.qValues[(state, action)] + self.alpha * (reward + ((self.discount * maxNextQValue) - self.qValues[(state, action)]))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.

       The goal of this process to replace states (which can be many)
       with features, which are usually much fewer.
       Number of ghosts nearby, number of foods nearby...
       instead of just states.
       So the map: state -> QValue (q learning) or 
       state -> Utility (value iteration)
       becomes: feature -> weight
       We increase or decrease the weights of each feature depending on the 
       rewards or penalties it led us to.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator.
          Now, we have to deduce the Q from the weight and the 
          feature vector
        """
        # Q(s, a) = sum (fi(s,a) * wi)
        qValue = 0
        features = self.featExtractor.getFeatures(state, action)
        for feature, fValue in features.items():
            qValue += fValue * self.weights[feature]
        return qValue

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        # Find the Utility/ the maxQValue of the next state
        maxNextQValue = -sys.maxsize
        nextActions = self.getLegalActions(nextState)
        for nextAction in nextActions:
            nextQValue = self.getQValue(nextState, nextAction)
            maxNextQValue = max(maxNextQValue, nextQValue)
        if len(nextActions) == 0:
            maxNextQValue = 0

        # difference = 
        # (the reward of moving from state to nextState via input action) 
        # + (discount * nextState's utility) - (this state's utility)
        difference = reward + (self.discount * maxNextQValue) - self.getQValue(state, action) 

        # collect features with given class and function
        features = self.featExtractor.getFeatures(state, action)
        # update wi = wi + alpha (difference * fi(s,a))
        for feature, fValue in features.items():
            self.weights[feature] = self.weights[feature] + self.alpha * (difference * fValue)

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.getWeights())
            pass
