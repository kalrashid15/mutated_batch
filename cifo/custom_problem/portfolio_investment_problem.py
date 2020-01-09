from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

from collections import Counter

import numpy as np

pip_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : False,
    "Can repeat"   : True,
    "Data"         : [0,1],
    "Data Type"    : "Choices"
}

pip_constraints_example = {
    "Risk-Tolerance" : 1,
    "Budget": 10000
}

# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem 
# -------------------------------------------------------------------------------------------------
class PortfolioInvestmentProblem( ProblemTemplate ):
    """
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = pip_encoding_rule):
        """
        #keys include: stock, stock_name, price, exp_ret, stdiv

        """
        # optimize the access to the decision variables
        self._stocks = []

        if "stock" in decision_variables:
            self._stocks = decision_variables["stock"]

        self._stock_names = []
        if "stock_name" in decision_variables:
            self._stock_names = decision_variables["stock_name"]

        self._prices = []
        if "price" in decision_variables:
            self._prices = decision_variables["price"]
        
        self._exp_rets = []
        if "exp_ret" in decision_variables:
            self._exp_rets = decision_variables["exp_ret"]

        self._stdivs = []
        if "stdiv" in decision_variables:
            self._stdivs = decision_variables["stock_name"]

        self._rf_rate = []
        if "Risk-free-rate" in constraints:
            self._rf_rate = constraints["Risk-free-rate"]

        self._max_investment = 0 # "Max-Weight"
        if "Max-Investment" in constraints:
            self._max_investment = constraints["Max-Investment"]
        
        self._risk_tolerance = 0 # "Max-Weight"
        if "Risk-Tolerance" in constraints:
            self._risk_tolerance = constraints["Risk-Tolerance"]

        encoding_rule["Size"] = len( self._stocks )

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Portfoilo Optimisation Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization

    # Build Solution for PIP
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        To bui
        """
        solution_representation = []
        encoding_data = self._encoding.encoding_data

        for _ in range(0, self._encoding.size):
            solution_representation.append(choice(encoding_data))
        
        solution = LinearSolution(
            representation = solution_representation,
            encoding_rule = self._encoding_rule
        )

        return solution
    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        1. investment budget <= $100,000
        2. Rish Tolerance/Sharpe Ratio >= 1

        Formula for SR:

        SR = E[Ra - Rb] / α
            
            α = Std deviation of Portfolio = root(var(Ra - Rb))
            Ra = portfolio Retrun 
            Rb = Risk-Free return = 1.56%

        calculation of α
            Find the std. of each asset
            Find the weight of each asset in the portfolio
            FInd the Corr between assets in the portfoilo

        """
        #defining a function to calcuate total investment of portfolio
        def cal_total_investment(porfolio):
            total_investment = 0
            for stock in porfolio:
                i = self._stocks.index(stock)
                total_investment += float(self._prices[i])
            return total_investment 

        #calculating total return
        def cal_total_return(porfolio):
            total_return = 0
            for stock in porfolio:
                i = self._stocks.index(stock)
                total_return += float(self._exp_rets[i])
            return total_return

        def cal_total_std(porfolio):
            total_std = 0
            for stock in porfolio:
                i = self._stocks.index(stock)
                total_return += float(self._stdivs[i])
            return total_std
        
        #defining a function to calcuate total risk of a porfoilio
        def cal_pfolio_sR(porfolio):
            #Sharpe Ratio = (Rx – Rf) / StdDev Rx

            #I dont think we need the weight here. But
            count = Counter(porfolio).items()
            percentages = {x: float(float(y) / len(porfolio) * 100.00) for x, y in count}
            
            list_of_stocks = percentages.keys()
            weight_of_stocks = np.array(percentages.values())

            #calcuating sharpe ratio
            pfolio_ret = cal_total_return(porfolio)
            rf_ret = self._rf_rate
            pfolio_std = cal_total_std(porfolio)

            pfolio_sR = float((pfolio_ret - rf_ret) / pfolio_std)

            return pfolio_sR
            
        #actual sharpe ratio calc.
        # calcuate covariance of stocks from sp_12_weeks
        #then pfolio_risk = np.sqrt(np.dot(weight_of_stocks.T, np.dot(cov_mat, weight_of_stocks)))                          

        
        #admissibility test
        stocks = self._stocks
        current_pfolio = []

        for  i in range(0, len( stocks )):
            if solution.representation[ i ] == 1:
                current_pfolio.append(stocks[ i ])

        current_pfolio_investment = cal_total_investment(current_pfolio)
        current_pfolio_sR = cal_pfolio_sR(current_pfolio) 
        if current_pfolio_investment <= self._max_investment & current_pfolio_sR >= self._rf_rate:
            result = True
        else:
            result = False


        return result 

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        pass     


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    pass