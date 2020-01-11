from copy import deepcopy
from random import choice, randint
import random
import pandas as pd

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

tsp_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [0,1],
    "Data Type"    : "Choices"
}


# REMARK: There is no constraint

# -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem( ProblemTemplate ):
    """
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = tsp_encoding_rule):
        """
        """
        # optimize the access to the decision variables

        self._cities = decision_variables["City"]
        self._xcoords = decision_variables["X"]
        self._ycoords = decision_variables["Y"]
        self._distancematrix = pd.DataFrame(decision_variables).drop(["City_id", "City", "X", "Y"], axis=1)
        self._all_fitnesses = [] 
        self._best_fitness = 999999999

        encoding_rule["Size"] = len( self._cities )
        self._encoding_rule = encoding_rule

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Travelling Salesman Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

    # Build Solution for Travelling Salesman Problem
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        """
        solution_representation = []
        for i in range(0, self._encoding.size):
            solution_representation.append(i)
            
        random.shuffle(solution_representation)
        solution_representation.append(solution_representation[0])

        solution = LinearSolution(
            representation = solution_representation,
            encoding_rule = self._encoding_rule
        )

        return solution


    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        Solution length is always 90 so let's just return True.
        """
        return True


    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        total_fit = 0

        for i in range(len(solution.representation)):
            origin = solution.representation[i-1]
            destination = solution.representation[i]
            
            total_fit += self._distancematrix.loc[origin, destination]
        
        solution.fitness = total_fit

        # self._all_fitnesses.append(solution.fitness)

        # if solution.fitness < self._best_fitness:
        #     self._best_fitness = solution.fitness
        #     self._best_fitness_solution = solution.representation

        return solution        


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it only is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def tsp_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    #the nearest city that has not yet been visited
    pass
