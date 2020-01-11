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
        self._optimum_stocks = 0

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
            #self._cov_mat = self._df_stocks.cov()

        if "Risk-free-rate" in constraints:
            self._rf_rate = constraints["Risk-free-rate"]

        
        if "Max-Investment" in constraints:
            self._max_investment = constraints["Max-Investment"]
        
        
        if "Risk-Tolerance" in constraints:
            self._risk_tolerance = constraints["Risk-Tolerance"]

        if "Optimum_num" in constraints:
            self._optimum_stocks = constraints["Optimum_num"]
        else:
            self._optimum_value = len(self._stocks)
        

        encoding_rule["Size"] = len( self._stocks )
        #print(encoding_rule["Size"])
        encoding_rule["Data"] = list(range(1,5))
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
        To build the solution
        we pick 32 stocks from the list as par Fisher and laurie (1970)
        """
        solution_representation = []
        
        #for _ in range(0, total_stock_size):
        #calculate portfolio investment
        def cal_total_investment(porfolio):
            total_investment = 0
            for i in porfolio:
                if i == 1:
                    total_investment += float(self._prices[i])
            return total_investment


        solution_representation = [0]*self._encoding_rule['Size']
        #len(solution_representation) sum(solution_representation) < 32) & 
        investment_limit = 0
        tolerance = 0.1 #limit before reaching the full investment
        while True:
            n = choice(self._encoding_rule['Data'])
            solution_representation[randint(0,len(solution_representation)-1)] = n
            solution_investment = cal_total_investment(solution_representation)
            investment_limit += solution_investment
            
            if (investment_limit >= self._max_investment*(1-tolerance)):
                break
        #print('investment limit', investment_limit)
        #solution_representation[1] = 15
        #print((solution_representation))
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

        def cal_pfolio_risk(portfolio, weights):
            #find the list of stocks and their weights

            #count = Counter(portfolio).items()
            #percentages = {x: float(float(y) / len(portfolio) * 100.00) for x, y in count}
            #print(percentages)
            #list_of_stocks = percentages.keys()
            #weight_of_stocks = np.array([float(x) for x in list(percentages.values())])
            list_of_stocks = portfolio
            weight_of_stocks = np.array(weights)
            
            cov_mat = self._df_stocks[list_of_stocks].cov()
            pfolio_risk = np.sqrt(np.dot(weight_of_stocks.T, np.dot(cov_mat, weight_of_stocks)))


            return pfolio_risk
        
        #defining a function to calcuate total investment of portfolio
        def cal_total_investment(porfolio, weights):
            total_investment = 0
            j = 0 #j index
            for stock in porfolio:
                i = self._stocks.index(stock)
                total_investment += (float(self._prices[i]) * weights[j])
                j += 1
            #print('check:  ',total_investment)
            return total_investment 

        #calculating total return
        def cal_total_return(portfolio, weights):
            total_return = 0
            j = 0 #j index
            for stock in portfolio:
                i = self._stocks.index(stock)
                total_return += float(self._exp_rets[i] * weights[j])
                j += 1
            return total_return

        
        #defining a function to calcuate total risk of a porfoilio
        def cal_pfolio_sR(portfolio, weights):
            #Sharpe Ratio = (Rx – Rf) / StdDev Rx
            pfolio_sR = 0

            #not efficient way of calcuating sharpe ratio
            # pfolio_ret = cal_total_return(porfolio)
            # rf_ret = self._rf_rate
            # pfolio_std = cal_total_std(porfolio)
            # pfolio_sR = float((pfolio_ret - rf_ret) / pfolio_std)
            
            #calcuating the nominitor
            pfolio_exp_ret = cal_total_return(portfolio, weights) -  self._rf_rate

            #calcuating the denominitor
            pfolio_std_div = cal_pfolio_risk(portfolio, weights)

            if(pfolio_std_div != 0):
                pfolio_sR = pfolio_exp_ret / pfolio_std_div
            #print('sharp ratio ', pfolio_sR)
            return pfolio_sR
            
        #actual sharpe ratio calc.
        # calcuate covariance of stocks from sp_12_weeks
        #then pfolio_risk = np.sqrt(np.dot(weight_of_stocks.T, np.dot(cov_mat, weight_of_stocks)))                          

        
        #admissibility test
        stocks = self._stocks
        current_pfolio = []
        weights = []

        for  i in range(0, len(solution.representation)):
            #print(solution.representation)
            if solution.representation[ i ] >= 1:
                current_pfolio.append(stocks[ i ])
                weights.append(solution.representation[ i ])



        current_pfolio_investment = cal_total_investment(current_pfolio, weights)
        current_pfolio_sR = cal_pfolio_sR(current_pfolio, weights)

        #print(self._max_investment)
        if (current_pfolio_investment <= self._max_investment) and (current_pfolio_sR >= self._rf_rate):
            result = True


        return result 

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        highest fitness, based on highest return
        """
        #stocks = self._stocks
        stocks_picked = deepcopy(solution.representation)
        #print(f'prining stocks: ', *stocks_picked)

        fitness = 0
        money_spent = 0
        #print(f'len of stocks ', len(self._stocks))
        for i in range(0, len( self._stocks )):
            if stocks_picked[ i ] == 1:
                #find the index of that solution
                #print(solution.representation[i])
                fitness += float(self._exp_rets[ i ])#need to check this
            elif stocks_picked[ i ] > 1:
                fitness += i*float(self._exp_rets[ i ])
                money_spent += i*float(self._prices[i])
        
        #print('money spent ', money_spent)
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