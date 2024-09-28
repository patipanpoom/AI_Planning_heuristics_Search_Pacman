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
from game import Actions
from game import Directions
from util import foodGridtoDic
import itertools
import copy

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


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    This heuristic is trivial.
    """
    return 0


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"
    pacmanPosition, foodGrid = state
    foodPositions = foodGrid.asList()
    number_of_food_left = len(foodPositions)
    if number_of_food_left == 0:
        return 0  # If no food left, heuristic is 0, indicating the goal state

    # # Version1

    # # Compute the Manhattan distance to the closest food pellet
    # min_distance = float('inf')
    # for food in foodPositions:
    #     distance = abs(pacmanPosition[0] - food[0]) + abs(pacmanPosition[1] - food[1])
    #     min_distance = min(min_distance, distance)

    # return min_distance

    # # Version2

    # # Compute the total Manhattan distance to all remaining food pellets
    # total_distance = 0

    # for food in foodPositions:
    #     total_distance += abs(pacmanPosition[0] - food[0]) + abs(pacmanPosition[1] - food[1])
    # return total_distance

    # # Version3
    # min_distance = float('inf')
    # for food in foodPositions:
    #     distance = abs(pacmanPosition[0] - food[0]) + abs(pacmanPosition[1] - food[1])
    #     min_distance = min(min_distance, distance)

    # # Multiply the minimum distance by the number of remaining food pellets
    # heuristic_value = min_distance * len(foodPositions)

    # return heuristic_value

    # Version4
    # Compute the Manhattan distance to the nearest food
    min_distance = float('inf')
    for food in foodPositions:
        distance = abs(pacmanPosition[0] - food[0]) + abs(pacmanPosition[1] - food[1])
        min_distance = min(min_distance, distance)

    # Compute the sum of Manhattan distances between the remaining food
    total_distance = 0
    for food1 in foodPositions:
        for food2 in foodPositions:
            distance = abs(food1[0] - food2[0]) + abs(food1[1] - food2[1])
            total_distance += distance

    # Multiply the minimum distance by the number of remaining food 
    heuristic_value = min_distance * number_of_food_left + total_distance / 2

    return heuristic_value
    # comment the below line after you implement the algorithm
    # util.raiseNotDefined()


class MAPFProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPositions, foodGrid ) where
      pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
      foodGrid:         a Grid (see game.py) of either pacman_name or False, specifying the target food of that pacman_name. For example, foodGrid[x][y] == 'A' means pacman A's target food is at (x, y). Each pacman have exactly one target food at start
    """

    def __init__(self, startingGameState):
        "Initial function"
        "*** WARNING: DO NOT CHANGE!!! ***"
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()

    def getStartState(self):
        "Get start state"
        "*** WARNING: DO NOT CHANGE!!! ***"
        return self.start

    def isGoalState(self, state):
        "Return if the state is the goal state"
        "*** YOUR CODE HERE for task2 ***"
        pacmanPositions, foodGrid = state
        food_for_pacman = set()
        for i in list(foodGrid):
            for pacman_name, position in pacmanPositions.items():
                if pacman_name in i:
                    food_for_pacman.add(pacman_name)
        
        for pacman_name, position in pacmanPositions.items():
            if pacman_name in food_for_pacman:
                if foodGrid[position[0]][position[1]] != pacman_name:
                    return False

        return True
        # comment the below line after you implement the function
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
            Returns successor states, the actions they require, and a cost of 1.
            Input: search_state
            Output: a list of tuples (next_search_state, action_dict, 1)

            A search_state in this problem is a tuple consists of two dictionaries ( pacmanPositions, foodGrid ) where
              pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
              foodGrid:    a Grid (see game.py) of either pacman_name or False, specifying the target food of each pacman.

            An action_dict is {pacman_name: direction} specifying each pacman's move direction, where direction could be one of 5 possible directions in Directions (i.e. Direction.SOUTH, Direction.STOP etc)
        """
        "*** YOUR CODE HERE for task2 ***"
        pacmanPositions, foodGrid = state
        successors = []
        number_of_pacman = len(pacmanPositions)
        # Possible directions for Agent
        directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        
        # Generate all possible combinations movements for Agent
        for action_combination in itertools.product(directions, repeat=number_of_pacman):
            # Initialize the next positions
            next_positions = {}
            for (pacman_name, position), action in zip(pacmanPositions.items(), action_combination):
                if action != Directions.STOP:  # If action is not STOP, calculate next position
                    dx, dy = Actions.directionToVector(action)
                    next_x, next_y = int(position[0] + dx), int(position[1] + dy)

                    # Check if the next position is a wall or not
                    if not self.walls[next_x][next_y]:
                        next_positions[pacman_name] = (next_x, next_y)
                else:  # If action is STOP, keep the current position
                    next_positions[pacman_name] = position

            collision = False

            # Check for vertex collision
            if len(set(next_positions.values())) != number_of_pacman:
                collision = True
            else:
                # Check for swapping collision
                swapped_positions = set()
                for agent, position in pacmanPositions.items():
                    next_position = next_positions[agent]
                    if next_position != position:
                        if (next_position, position) in swapped_positions or (position, next_position) in swapped_positions:
                            collision = True
                            break
                        swapped_positions.add((position, next_position))

            if not collision:
                # Append the successor state, action dictionary, and cost
                successors.append(((next_positions, foodGrid), dict(zip(pacmanPositions.keys(), action_combination)), 1))
        return successors
        # comment the below line after you implement the function
        # util.raiseNotDefined()


def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores the path for each pacman as a list {pacman_name: [action1, action2, ...]}.

    """
    "*** YOUR CODE HERE for task3 ***"
    def low_level_heuristic(state, constraints):
        pacmanPositions, foodGrid = state
        foodPositions = {}
        for row_idx, row in enumerate(foodGrid):
            for col_idx, value in enumerate(row):
                if value != False:
                    foodPositions[value] = (row_idx, col_idx)

        for pacman_name, position in pacmanPositions.items():
            distance = abs(position[0] - foodPositions[pacman_name][0]) + abs(position[1] - foodPositions[pacman_name][1])
        
        heuristic_values = distance
        return heuristic_values


    def lowLevelSearch(state, constraints=None, heuristic=low_level_heuristic):
        myPQ = util.PriorityQueue()
        startNode = (state, 0, [])
        myPQ.push(startNode, heuristic(state, constraints))
        # print("QUE",startNode, heuristic(state, constraints))
        best_g = dict()

        while not myPQ.isEmpty():
            node = myPQ.pop()
            state, cost, path = node
            state_hashable = tuple(state[0].values())[0]
            if (not state_hashable in best_g) or (cost < best_g[state_hashable]):
                best_g[state_hashable] = cost
                if problem.isGoalState(state):
                    return path
                
                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    new_cost = cost + succCost
                    newNode = (succState, new_cost, path + [succAction])
                    # Check if the successor state violates any constraint
                    if constraints and violates_constraints(newNode, constraints):
                        print("VIOLATE CONSTRAINT")
                        continue  # Skip this successor if it violates constraints
                    myPQ.push(newNode, heuristic(succState, constraints) + new_cost)
                    print("QUE", newNode, heuristic(succState, constraints) + new_cost)

        return None  # Goal not found
    
    # (Agent, Coordinate, Step)
    def violates_constraints(node, constraints):
        if not constraints:
            return False
        else:
            succState, new_cost, _ = node
            for constrain in constraints:
                pacman, coordinate, cost = constrain
                # print(succState[0], pacman, coordinate, new_cost, cost)
                if succState[0][pacman] == coordinate and new_cost == cost - 1:
                    # print("HELLOWORLD")
                    return True
        return False

    def calculatePathCost(path):
        return len(path) - 1
    
    def sic(name, startCoordinate, path):
        coordinates = [startCoordinate]
        for index, action in enumerate(path):
            dx, dy = Actions.directionToVector(action[name])
            next_x, next_y = int(coordinates[index][0] + dx), int(coordinates[index][1] + dy)
            coordinates.append((next_x, next_y))

        return coordinates
        
    def findFirstConflict(paths):
        # Iterate over each time step
        for timestep, time_step_positions in enumerate(zip(*paths), start=1):
            # Create a dictionary to track the agents at each location
            location_agents = {}
            # Iterate over each agent's position at the current time step
            for agent, position in enumerate(time_step_positions, start=1):
                # Check if the current position is already occupied by another agent
                if position in location_agents:
                    # Conflict detected, return agent, conflict location, and timestep
                    conflict_location = position
                    conflicting_agents = (location_agents[position], agent)
                    return conflicting_agents, conflict_location, timestep
                # Update the location_agents dictionary with the current agent's position
                location_agents[position] = agent
        # No conflict found
        return None
    

    def calculateSolutionCost(solution):
        total_cost = sum(len(agent_path) for agent_path in solution)
        return total_cost - len(solution)
    

    def generatePathDictionary(paths, pacman_names):
        path_dict = {}

        # Iterate over each agent's path and corresponding pacman name
        for path, pacman_name in zip(paths, pacman_names):
            actions = []

            # Iterate over each step in the path and calculate the action
            for i in range(len(path) - 1):
                current_position = path[i]
                next_position = path[i + 1]

                # Calculate the action based on the change in position
                dx = next_position[0] - current_position[0]
                dy = next_position[1] - current_position[1]

                action = Actions.vectorToDirection((dx, dy))
                actions.append(action)

            # Add the list of actions to the path dictionary for the current pacman
            path_dict[pacman_name] = actions

        return path_dict
    

    # Find individual paths using the low-level search     
    startState, food = problem.getStartState()
    low_level_coordinates = {}
    costs = {}
    pacman_names = []

    for state, value in startState.items():
        pacman_names.append(state)
        low_level_path_each = lowLevelSearch(({state: value}, food))
        low_level_coordinates[state] = []
        costs[state] = []
        # Calculate the low-level solution
        low_level_coordinate_each = sic(state, value, low_level_path_each)
        low_level_coordinates[state].append(low_level_coordinate_each)
        # Calculate the cost of the solution
        costs[state].append(calculatePathCost(low_level_coordinate_each))

    sum_cost = 0
    solution = []
    for i in range(len(costs[pacman_names[0]])):
        for pacman_name in pacman_names:
            sum_cost += costs[pacman_name][i]
            solution.append(low_level_coordinates[pacman_name][i])

    # # Create a node representing the initial solution
    initial_node = {"constraints": None, "solution": solution, "cost": sum_cost}
    # Initialize a priority queue for the search
    open_set = [initial_node]
    # Iterate until OPEN is not empty
    while open_set:
        # Get the best node from OPEN (lowest solution cost)
        best_node = min(open_set, key=lambda x: x["cost"])
        # Validate the paths in the current node until a conflict occurs
        if not findFirstConflict(best_node["solution"]):
            # If there is no conflict, return the solution
            return generatePathDictionary(best_node["solution"], pacman_names)
        # Find the first conflict in the current solution
        conflict = findFirstConflict(best_node["solution"])
        # For each agent involved in the conflict
        for agent in conflict[0]:
            agent_name = pacman_names[int(agent)-1]
            # Create a new node
            copy_node = copy.deepcopy(best_node)
            new_node = {"constraints": copy_node["constraints"], "solution": copy_node["solution"]}

            # Add constraint for the conflicting agent
            if not new_node["constraints"]:
                new_node["constraints"] = [(agent_name, conflict[1], conflict[2])]
            else:
                new_node["constraints"].append((agent_name, conflict[1], conflict[2]))

            # Update the solution for the conflicting agent
            new_node["solution"][agent-1] = sic(agent_name, startState[agent_name], 
                                                lowLevelSearch(({agent_name: startState[agent_name]}, food), 
                                                               constraints=new_node["constraints"]))
            # Update the cost of the new node
            new_node["cost"] = calculateSolutionCost(new_node["solution"])
            open_set.append(new_node)

        open_set.remove(best_node)
        # break
    # If no conflict-free solution is found, return None
    return None
    # comment the below line after you implement the function
    # util.raiseNotDefined()


"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"


class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                nextFood = state[1].copy()
                nextFood[next_x][next_y] = False
                successors.append((((next_x, next_y), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class SingleFoodSearchProblem(FoodSearchProblem):
    """
    A special food search problem with only one food and can be generated by passing pacman position, food grid (only one True value in the grid) and wall grid
    """

    def __init__(self, pos, food, walls):
        self.start = (pos, food)
        self.walls = walls
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch
