# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).

      A very low discount means a quicker reward (+1) is preferred, 
      even over a much bigger but farther one (+10).
    """
    answerDiscount = 0.01
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).

      The noise being as high as 50 percent means that taking
      the shorter path that is close to the cliff is really risky.
      There is a very good chance we will fall into it.

      The tiny living penalty encourages our agent to move 
      towards positive termination, as it seemed to prefer hanging around.
      We kept it tiny so it did not get it into its head to suddenly be 
      willing to settle for the cliff,

      The larger-than-expected discount also provides incentive to terminate, because 
      while the living penalty provided some push, especially initially, 
      it proved insuffcient when the agent was due to go east. We think the
      the  reluctant to go east was because agent did not want
      to incur the living penalty of moving towards any terminal,
      (this was after the noise pushed it away from the lower path, 
      and the penalty pushed it north).
      Increasing the discount made it worth it to brave the penalties towards a terminal,
      but we were careful to increase it just enough that +1 was worth it, but +10 was not.
      We also ensured to keep the total cost east and south towards +1 lower than +1, so again,
      the trip was worth it.
    """
    answerDiscount = 0.6
    answerNoise = 0.5
    answerLivingReward = -0.2
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
      
      High discount means that we care more about a future, larger
      reward (+10) than an immediate, smaller one (+1).
    """
    answerDiscount = 0.7
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).

      High discount means that we care more about a future, larger
      reward (+10) than an immediate, smaller one (+1).
      The noise being as high as 50 percent means that taking
      the shorter path that is close to the cliff is extra risky.
      There is a good chance we will fall into it.
    """
    answerDiscount = 0.7
    answerNoise = 0.5
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).

      Discount is neutral. We don't particularly care more for present pleasure 
      nor delayed gratification.
      The living reward is ridiculously high. We have no reason to go to any terminal.
    """
    answerDiscount = 0.5
    answerNoise = 0
    answerLivingReward = 1000
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
