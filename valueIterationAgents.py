# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util, sys

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.

          We use batch update-- that is, when we need to update a value using a value
          from another state, we use the value from the last iteration, even though, the
          value might already have changed in the present iteration.
          Hence, the variable oldValues

          Furthermore, our completion condition is simply the number of iterations. 
          Not the complicated formula from class.
        """
        # Update the utility/value of each state using the potential new states 
        # (their utilities and probabilities)
        for i in range(self.iterations):
            oldValues = self.values.copy()
            for state in self.mdp.getStates():
                newVal = self.computeNewValue(state, oldValues)
                self.values[state] = newVal

    def computeNewValue(self, state, oldValues):
        # The fugue state we end up, after we have left the GUI-displayed terminal
        # This state allows our GUI-displayed terminal to be updated 
        if state == "TERMINAL_STATE":
            return 0
        # to hold the maximum Q value or the new utility
        maxValue = -sys.maxsize
        for action in self.mdp.getPossibleActions(state):
            # What states could we potentially end up if we take this action
            # at this present state? And how likely are each of them?
            transAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            # For each potential state, find its Q value...
            # The sum of all states' (probability * utility) 
            # times the discount + the reward
            sumQ = 0
            for i in range(len(transAndProbs)):
                nextState = transAndProbs[i][0]
                probOfNextState = transAndProbs[i][1]
                reward = self.mdp.getReward(state, action, nextState)
                sumQ += (probOfNextState * oldValues[nextState] * self.discount) + reward
            # The max Q value out of all the actions' Q values
            maxValue = max(maxValue, sumQ)
        return maxValue

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
    

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          For each state, we find its potential next states,
          given the input actions.
          We take the product of the utilities and the 
          probabilities of these potential next states,
          and add them all together. That is the Q-value.

          This function only works after runValueIteration() 
          has populated self.values, giving each state a utility.
          This is done in the constructor.
        """
        if state == "TERMINAL_STATE":
            return 0
        transAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for i in range(len(transAndProbs)):
            nextState = transAndProbs[i][0]
            probNextState = transAndProbs[i][1]
            reward = self.mdp.getReward(state, action, nextState)
            qValue += (probNextState * self.values[nextState] * self.discount) + reward
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

          Here, we check which action would give us the greatest
          qValue, and return that. Ties are not broken. The latest
          max qValue is what gets returned.
        """
        maxAction = None
        maxQValue = -sys.maxsize
        for action in self.mdp.getPossibleActions(state):
            transAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            qValue = 0      
            for i in range(len(transAndProbs)):
                nextState = transAndProbs[i][0]
                probNextState = transAndProbs[i][1]
                qValue += (self.values[nextState] * probNextState) 
            maxQValue = max(maxQValue, qValue)
            if maxQValue == qValue:
                maxAction = action
        return maxAction
  
  
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
