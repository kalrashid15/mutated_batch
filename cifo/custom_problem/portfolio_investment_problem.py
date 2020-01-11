from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

from collections import Counter

import numpy as np
import pandas as pd

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
        #declaring variables

        self._stocks = []
        self._stock_names = []
        self._prices = []
        self._exp_rets = []
        self._stdivs = []
        self._rf_rate = 0
        self._max_investment = 0 # "Max-Investment"
        self._risk_tolerance = 0 # "Max-width"
        self._df_stocks = pd.DataFrame() # a dataframe to store historical prices to generate Cov table.


        if (len(decision_variables) != 2):
            print(f'we do not have all the variables')


        if "stock" in decision_variables[0]:
            self._stocks = decision_variables[0]["stock"]

        
        if "stock_name" in decision_variables[0]:
            self._stock_names = decision_variables[0]["stock_name"]

        if "price" in decision_variables[0]:
            self._prices = decision_variables[0]["price"]
        
        
        if "exp_ret" in decision_variables[0]:
            self._exp_rets = decision_variables[0]["exp_ret"]

        
        if "stdiv" in decision_variables[0]:
            self._stdivs = decision_variables[0]["stock_name"]

        
        if len(decision_variables) == 2:
            self._df_stocks = decision_variables[1]

        if "Risk-free-rate" in constraints:
            self._rf_rate = constraints["Risk-free-rate"]

        
        if "Max-Investment" in constraints:
            self._max_investment = constraints["Max-Investment"]
        
        
        if "Risk-Tolerance" in constraints:
            self._risk_tolerance = constraints["Risk-Tolerance"]
        
        #sanity check of the assignments
        """
        print(f'stocks: {self._stocks[100:210]}, stock names {self._stock_names[100:210]}, prices {self._stock_names[100:210]}')
        print(f'risk free rates: {self._rf_rate}, limit {self._max_investment}, risk tolerate {self._risk_tolerance}')
        print(f'dataframe for cov {self._df_stocks.head()}')

        breakpoint
        """
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

        #print(self._stocks)

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
            #print(solution_representation.append(choice(encoding_data)))
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
        #calcuate cov_matrix
        df_stocks = pd.read_excel(r'./data/sp_12_weeks.xlsx')

        def cal_pfolio_risk(portfolio):
            #find the list of stocks and their weights

            count = Counter(portfolio).items()
            percentages = {x: float(float(y) / len(portfolio) * 100.00) for x, y in count}
            
            list_of_stocks = percentages.keys()
            weight_of_stocks = np.array([float(x) for x in list(percentages.values())])

            cov_mat = df_stocks[list_of_stocks].cov()
            pfolio_risk = np.sqrt(np.dot(weight_of_stocks.T, np.dot(cov_mat, weight_of_stocks)))

            return pfolio_risk
        
        #defining a function to calcuate total investment of portfolio
        def cal_total_investment(porfolio):
            total_investment = 0
            for stock in porfolio:
                i = self._stocks.index(stock)
                total_investment += float(self._prices[i])
            return total_investment 

        #calculating total return
        def cal_total_return(portfolio):
            total_return = 0
            for stock in portfolio:
                i = self._stocks.index(stock)
                total_return += float(self._exp_rets[i])
            return total_return

        
        #defining a function to calcuate total risk of a porfoilio
        def cal_pfolio_sR(portfolio):
            #Sharpe Ratio = (Rx – Rf) / StdDev Rx
            pfolio_sR = 0

            #not efficient way of calcuating sharpe ratio
            # pfolio_ret = cal_total_return(porfolio)
            # rf_ret = self._rf_rate
            # pfolio_std = cal_total_std(porfolio)
            # pfolio_sR = float((pfolio_ret - rf_ret) / pfolio_std)
            
            #calcuating the nominitor
            pfolio_exp_ret = cal_total_return(portfolio) -  self._rf_rate

            #calcuating the denominitor
            pfolio_std_div = cal_pfolio_risk(portfolio)

            if(pfolio_std_div != 0):
                pfolio_sR = pfolio_exp_ret / pfolio_std_div

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
        if (current_pfolio_investment <= self._max_investment) & (current_pfolio_sR >= self._rf_rate):
            result = True
        else:
            result = False


        return result 

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        highest fitness, based on highest return
        """
        stocks = self._stocks
        #print(f'prining stocks: ', *stocks)

        fitness = 0
        #print(f'len of stocks ', len(stocks))
        #for  i in range(0, len( stocks )):
            #if solution.representation[ i ] == 1:
            #print(solution.representation[i])
                #fitness += float(self._exp_rets[self._stocks.index(solution.representation[ i ])])#need to check this
        
        solution.fitness = fitness

        return solution      


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    pass
    """
    neighbors = []

    # Generate all neighbors considering a bit flip
    for position in range(0, len(solution.representation)):
        n = deepcopy(solution) # solution.clone()
        if n.representation[ position ]  ==  1 : 
            n.representation[ position ] = 0
        else: 
            n.representation[ position ] = 1
        
        neighbors.append(n)

    # return all neighbors
    if neighborhood_size == 0:
        return neighbors
    # return a RANDOM subset of all neighbors (in accordance with neighborhood size)    
    else:     
        subset_neighbors = []
        indexes = list( range( 0, len( neighbors ) ) )
        for _ in range(0, neighborhood_size):
            selected_index = choice( indexes )

            subset_neighbors.append( neighbors[ selected_index ] )
            indexes.remove( selected_index )

        return subset_neighbors  
    """