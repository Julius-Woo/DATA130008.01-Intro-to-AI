# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# my code also
class Node():
    '''
    define a class for node information
    parameters:
    state: the state that corresponds to this node
    parent: the node which generates this node
    action: the action that was applied to the parent’s state to generate this node
    cost_path: the total cost of the path from root to this node, or g(node)
    '''
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    path = []
    root = Node(state=problem.getStartState())
    frontier = util.Stack()
    frontier.push(root)
    visited = []  # store states corresponding to expanded nodes
    while not frontier.isEmpty():
        node_cur = frontier.pop()
        if problem.isGoalState(node_cur.state):  # if current state is Goal, return the path
            while node_cur.parent:
                path.append(node_cur.action)
                node_cur = node_cur.parent
            path.reverse()
            return path
        if node_cur.state in visited:  # graph search: ignore visited states
            continue
        visited.append(node_cur.state)
        # consider current node's successors
        successors = problem.getSuccessors(node_cur.state)
        for successor in successors:  # successor: a list of triples, (successor, action, stepCost)
            if successor[0] not in visited:  # ignore reached states
                child = Node(state=successor[0], parent=node_cur, action=successor[1])  # path_cost can be ignored in DFS
                frontier.push(child)
    return "Failure"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    path = []
    root = Node(state=problem.getStartState())
    if problem.isGoalState(root.state):  # check if root node is Goal
        return path
    frontier = util.Queue()
    frontier.push(root)
    visited = []  # store states corresponding to expanded nodes
    while not frontier.isEmpty():
        node_cur = frontier.pop()
        if node_cur.state in visited:  # graph search: ignore visited states
            continue
        visited.append(node_cur.state)
        # consider current node's successors
        successors = problem.getSuccessors(node_cur.state)
        for successor in successors:  # successor: a list of triples, (successor, action, stepCost)
            s = successor[0]
            if problem.isGoalState(s):  # if current state's child is Goal, return the path
                path.append(successor[1])
                while node_cur.parent:
                    path.append(node_cur.action)
                    node_cur = node_cur.parent
                path.reverse()
                return path
            if s not in visited:  # ignore reached states
                child = Node(state=s, parent=node_cur, action=successor[1])  # path_cost can be ignored in DFS
                frontier.push(child)
    return "Failure"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path = []
    root = Node(state=problem.getStartState())
    frontier = util.PriorityQueue()
    frontier.push(root, root.path_cost)
    reached = {root.state: root}  # store mappings between states and corresponding nodes
    while not frontier.isEmpty():
        node_cur = frontier.pop()
        if problem.isGoalState(node_cur.state):  # if current state is Goal, return the path
            while node_cur.parent:
                path.append(node_cur.action)
                node_cur = node_cur.parent
            path.reverse()
            return path
        # consider current node's successors
        successors = problem.getSuccessors(node_cur.state)
        for successor in successors:  # successor: a list of triples, (successor, action, stepCost)
            s = successor[0]
            nwpc = successor[2]+node_cur.path_cost  # new path cost
            if s not in reached or nwpc < reached[s].path_cost:  # ignore reached states or update
                child = Node(state=s, parent=node_cur,
                            action=successor[1], path_cost=nwpc)
                reached[s] = child
                frontier.update(item=child, priority=child.path_cost)
    return "Failure"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    class Node():
        '''
        define a class for node information
        parameters:
        state: the state that corresponds to this node
        parent: the node which generates this node
        action: the action that was applied to the parent’s state to generate this node
        path_cost: the total cost of the path from root to this node, or g(node)
        heuristic: the heuristic value, or h(node)
        '''

        def __init__(self, state, parent, action, path_cost, heuristic_val):
            self.state = state
            self.parent = parent
            self.action = action
            self.path_cost = path_cost
            self.heuristic_val = heuristic_val

    path = []
    root = Node(problem.getStartState(), None, None, 0, heuristic(problem.getStartState(), problem))
    frontier = util.PriorityQueue()
    frontier.push(root, root.path_cost+root.heuristic_val)
    visited = []  # store states corresponding to expanded nodes
    while not frontier.isEmpty():
        node_cur = frontier.pop()
        if problem.isGoalState(node_cur.state):  # if current state is Goal, return the path
            while node_cur.parent:
                path.append(node_cur.action)
                node_cur = node_cur.parent
            path.reverse()
            return path
        if node_cur.state in visited:
            continue
        visited.append(node_cur.state)
        # consider current node's successors
        successors = problem.getSuccessors(node_cur.state)
        for successor in successors:  # successor: a list of triples, (successor, action, stepCost)
            s = successor[0]
            if s not in visited:  # ignore reached states or update
                nwpc = successor[2] + node_cur.path_cost  # new path cost
                child = Node(state=s, parent=node_cur,
                             action=successor[1], path_cost=nwpc, heuristic_val=heuristic(s, problem))
                frontier.update(item=child, priority=child.path_cost+child.heuristic_val)
                #  priority = node.path_cost + node.heuristic_cost = g(n) + h(n)
    return "Failure"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
