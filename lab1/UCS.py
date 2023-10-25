import heapq
import sys

class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            assert type(i) == node, 'i must be node'
            if i.state == item.state:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class node:
    """define node"""

    def __init__(self, state, parent, path_cost, action,):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.action = action


class problem:
    """searching problem"""

    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.actions = actions

    def search_actions(self, state):
        """Search actions for the given state.
        Args:
            state: a string e.g. 'A'

        Returns:
            a list of action string list
            e.g. [['A', 'B', '2'], ['A', 'C', '3']]
        """
        return [action for action in self.actions if action[0] == state]
        raise Exception	

    def solution(self, node):
        """Find the path & the cost from the beginning to the given node.

        Args:
            node: the node class defined above.

        Returns:
            ['Start', 'A', 'B', ....], Cost
        """
        path = []
        current_node = node  # Create a variable to iterate through nodes
        while current_node:  # Traverse till the root node
            path.append(current_node.state)
            current_node = current_node.parent
        path.reverse()
        return path, node.path_cost
        raise Exception	

    def transition(self, state, action):
        """Find the next state from the state adopting the given action.

        Args:
            state: 'A'
            action: ['A', 'B', '2']

        Returns:
            string, representing the next state, e.g. 'B'
        """
        return action[1]
        raise Exception

    def goal_test(self, state):
        """Test if the state is goal

        Args:
            state: string, e.g. 'Goal' or 'A'

        Returns:
            a bool (True or False)
        """

        return state == "Goal"
        raise Exception	

    def step_cost(self, state1, action, state2):
        if (state1 == action[0]) and (state2 == action[1]):
            return int(action[2])
        else:
            print("Step error!")
            sys.exit()

    def child_node(self, node_begin, action):
        """Find the child node from the node adopting the given action

        Args:
            node_begin: the node class defined above.
            action: ['A', 'B', '2']

        Returns:
            a node as defined above
        """
        state = self.transition(node_begin.state, action)
        path_cost = node_begin.path_cost + self.step_cost(node_begin.state, action, state)
        return node(state, node_begin, path_cost, action)
        raise Exception


def UCS(problem):
    """Using Uniform Cost Search to find a solution for the problem.

    Args:
        problem: problem class defined above.

    Returns:
        a list of strings representing the path, along with the path cost as an integer.
            e.g. ['A', 'B', '2'], 5
        if the path does not exist, return 'Unreachable'
    """
    node_test = node(problem.initial_state, '', 0, '')
    frontier = PriorityQueue()
    frontier.push(node_test, node_test.path_cost)
    state2node = {node_test.state: node_test}
    explored = []

    while not frontier.isEmpty():
        current_node = frontier.pop()

        if problem.goal_test(current_node.state):
            return problem.solution(current_node)

        explored.append(current_node.state)

        for action in problem.search_actions(current_node.state):
            child = problem.child_node(current_node, action)

            if child.state not in explored and child.state not in state2node:
                state2node[child.state] = child
                frontier.push(child, child.path_cost)
            elif child.state in state2node and child.path_cost < state2node[child.state].path_cost:
                state2node[child.state] = child
                frontier.update(child, child.path_cost)

    return "Unreachable", -99
    raise Exception


if __name__ == '__main__':
    Actions = []
    while True:
        a = input().strip()
        if a != 'END':
            a = a.split()
            Actions += [a]
        else:
            break
    graph_problem = problem('Start', Actions)
    answer, path_cost = UCS(graph_problem)
    s = "->"
    if answer == 'Unreachable':
        print(answer)
    else:
        path = s.join(answer)
        print(path)
        print(path_cost)
