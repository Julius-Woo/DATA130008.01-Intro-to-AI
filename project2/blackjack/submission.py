import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [-1, 1]
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [] if (state == -1 or state == 1) else [(1, 0.1, 10), (-1, 0.9, 1)]
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        sum_cur, nextpeek_cur, deck_cur = state
        if deck_cur is None:  # terminal state
            return []
        if action == 'Take':
            if nextpeek_cur is not None:  # peeked in last action
                sum_next = sum_cur + self.cardValues[nextpeek_cur]
                if sum_next > self.threshold:  # bust
                    return [((sum_next, None, None), 1, 0)]
                else:
                    deck_next = list(deck_cur)
                    deck_next[nextpeek_cur] -= 1
                    if sum(deck_next) == 0:  # empty deck
                        return [((sum_next, None, None), 1, sum_next)]
                    else:
                        return [((sum_next, None, tuple(deck_next)), 1, 0)]
            else:  # not peeked in last action
                list_next = []  # initalize the return list
                totalcards = sum(deck_cur)  # total number of cards in deck
                for i in range(len(self.cardValues)):
                    if deck_cur[i] > 0:
                        prob = deck_cur[i] / totalcards
                        sum_next = sum_cur + self.cardValues[i]
                        if sum_next > self.threshold:  # bust
                            list_next.append(((sum_next, None, None), prob, 0))
                        else:
                            deck_next = list(deck_cur)
                            deck_next[i] -= 1
                            if sum(deck_next) == 0:  # empty deck
                                list_next.append(
                                    ((sum_next, None, None), prob, sum_next))
                            else:
                                list_next.append(((sum_next, None, tuple(deck_next)), prob, 0))
                return list_next
        elif action == 'Peek':
            if nextpeek_cur is not None:  # cannot peek twice
                return []
            list_next = []  # initalize the return list
            totalcards = sum(deck_cur)  # total number of cards in deck
            for i in range(len(self.cardValues)):
                if deck_cur[i] > 0:
                    prob = deck_cur[i] / totalcards
                    list_next.append(((sum_cur, i, deck_cur), prob, -self.peekCost))
            return list_next
        else:  # 'Quit'
            return [((sum_cur, None, None), 1, sum_cur)]
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    return BlackjackMDP(cardValues=[1, 3, 5, 6, 50], multiplicity=1, threshold=20, peekCost=1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q_cur learning

# Performs Q_cur-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q_cur function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        Q_cur = self.getQ(state, action)
        if newState is not None:
            Qmax = max(self.getQ(newState, a) for a in self.actions(newState))
            for f, v in self.featureExtractor(state, action):
                self.weights[f] += self.getStepSize() * (reward + self.discount * Qmax - Q_cur) * v
        else:  # terminal state
            for f, v in self.featureExtractor(state, action):
                self.weights[f] += self.getStepSize() * (reward - Q_cur) * v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q_cur-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q_cur-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    random.seed(123)  # set random seed for reproducibility
    Q_learn = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)
    value_iter = ValueIteration()  # use value iteration to get optimal policy
    value_iter.solve(mdp)
    
    util.simulate(mdp, Q_learn, 30000)  # simulate 30000 times with Q-learning
    Q_learn.explorationProb = 0  # no exploration, then .getAction() will return the optimal action
    
    n_state = len(mdp.states)  # number of total states
    n_diff = 0  # number of states with different actions
    for state in mdp.states:
        if Q_learn.getAction(state) != value_iter.pi[state]:
            n_diff += 1
    
    if mdp.multiplicity == 2:  # small MDP
        print("For small MDP, the match rate is: %.4f"  %(1 - n_diff / n_state))
        print("Number of total states is: %d; Number of diferent actions is: %d" %(n_state, n_diff))
    else:  # large MDP
        print("For large MDP, the match rate is: %.4f"  %(1 - n_diff / n_state))
        print("Number of total states is: %d; Number of diferent actions is: %d" % (
            n_state, n_diff))
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q_cur-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    featurepairs = [((action, total),1)]
    if counts is not None:
        counts = list(counts)
        for i in range(len(counts)):
            featurepairs.append(((action, i, counts[i]),1))
            if counts[i] > 0:
                counts[i] = 1
            else:
                counts[i] = 0
        featurepairs.append(((action, tuple(counts)),1))
    return featurepairs
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    random.seed(123)  # set random seed for reproducibility
    value_iter = ValueIteration()  # use value iteration to get optimal policy
    value_iter.solve(original_mdp)
    
    fixedrl = util.FixedRLAlgorithm(value_iter.pi)  # use optimal policy to simulate
    r_val_s = util.simulate(modified_mdp, fixedrl, numTrials=30)  # small trial number
    print("The average reward for value iteration with 30 trials is: %.4f"  %(sum(r_val_s) / len(r_val_s)))
    r_val_l = util.simulate(modified_mdp, fixedrl, numTrials=10000)  # large trial number
    print("The average reward for value iteration with 10000 trials is: %.4f"  %(sum(r_val_l) / len(r_val_l)))
    
    Qrl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor)  # use Q-learning to simulate
    r_ql_s = util.simulate(modified_mdp, Qrl, numTrials=30)
    print("The average reward for Q-learning with 30 trials is: %.4f"  %(sum(r_ql_s) / len(r_ql_s)))
    r_ql_l = util.simulate(modified_mdp, Qrl, numTrials=10000)
    print("The average reward for Q-learning with 10000 trials is: %.4f"  %(sum(r_ql_l) / len(r_ql_l)))
    # END_YOUR_CODE

