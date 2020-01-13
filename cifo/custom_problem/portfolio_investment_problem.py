from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

from collections import Counter

import numpy as np
import pandas as pd


import openpyxl
from itertools import takewhile, count



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
        self._best_fit = 0

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
        encoding_rule["Data"] = list(range(0,10))
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
        
        #calculate portfolio investment
        def cal_total_investment(portfolio):
            total_investment = 0
            for i in range (0, len(portfolio)):
                if portfolio[i] >= 1:
                    #print(i, 'printing i')
                    total_investment += float(self._prices[i])*portfolio[i]
            return total_investment


        solution_representation = [0]*self._encoding_rule['Size']
        solution_investment = 0
        tolerance = 0.20 #limit before reaching the full investment
        stock_counter = self._optimum_stocks

        for i in takewhile(lambda i:i<stock_counter and solution_investment<self._max_investment*(1-tolerance), count()):
            n = choice(self._encoding_rule['Data'])
            solution_representation[randint(0,len(solution_representation)-1)] = n
            solution_investment += cal_total_investment(solution_representation)
            i+=1

        """
        while True: #((investment_limit <= self._max_investment*(1-tolerance)) ):#and (sum(solution_representation) < self._optimum_stocks)
            n = choice(self._encoding_rule['Data'])
            solution_representation[randint(0,len(solution_representation)-1)] = n
            solution_investment = cal_total_investment(solution_representation)            
            if (solution_investment >= self._max_investment*(1-tolerance)):
                break
        #print('investment limit :',investment_limit)
        """
        solution = LinearSolution(
            representation = solution_representation,
            encoding_rule = self._encoding_rule
        )
        #print(f'Build Solution: total stocks picked in {sum(solution.representation)}, at cost: {solution_investment}')

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

            list_of_stocks = []
            weight_of_stocks = []
            for i in range(0, len(portfolio)):
                if portfolio[i] >= 1:
                    #print(f'printing stocks {stock}')
                    list_of_stocks.append( self._stocks[i])
                    weight_of_stocks.append(weights[i])
            
            weight_of_stocks = np.array(weight_of_stocks)
            cov_mat = self._df_stocks[list_of_stocks].cov()
            pfolio_risk = np.sqrt(np.dot(weight_of_stocks.T, np.dot(cov_mat, weight_of_stocks)))


            return pfolio_risk
        
        #defining a function to calcuate total investment of portfolio
        def cal_total_investment(portfolio, weights):
            #print(f'calculating weights at investment calc{sum(portfolio)}')
            total_investment = 0
            for i in range(0, len( portfolio )):
                if portfolio[ i ] >= 1:
                    #print(f'printing stock details {self._stocks[i]}, price: {self._prices[i]}, {self._exp_rets[i]} and weight {weights[i]}')

                    #print(f'printing cal_total_invest_: {i}, {self._prices[i]}, and {weights[i]}')
                    total_investment += (float(self._prices[i]) * weights[i])
            return total_investment 

        #calculating total return
        def cal_total_return(portfolio, weights):
            total_return = 0
            for i in range(0, len( portfolio )):
                if portfolio[ i ] >= 1:
                    total_return += float(self._exp_rets[i] * weights[i])
            return total_return

        
        #defining a function to calcuate total risk of a porfoilio
        def cal_pfolio_sR(portfolio, weights):
            #Sharpe Ratio = (Rx – Rf) / StdDev Rx
            pfolio_sR = 0

            
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
        current_pfolio = solution.representation
        weights = [0] * (len(solution.representation))
        
        for  i in range(0, len(solution.representation)):
            #print(solution.representation)
            if solution.representation[ i ] >= 1:
                #current_pfolio[i] = stocks[ i ]
                weights[i]= solution.representation[ i ]
                #/sum(current_pfolio)prin
        #print(current_pfolio)

        #print(f'weigth calculated at admissibilty {sum(weights)}')
        #print(current_pfolio,'current pfolio')
        current_pfolio_investment = cal_total_investment(current_pfolio, weights)
        current_pfolio_sR = cal_pfolio_sR(current_pfolio, weights)
        
        if (current_pfolio_investment <= self._max_investment) and (current_pfolio_sR >= self._risk_tolerance):
            result = True
            #print(f'Accepted!  porfolio Investment: {current_pfolio_investment}, and Sharp Ratio {current_pfolio_sR}')
        else:
            #print(f'rejected: {current_pfolio_investment}, and Sharp Ratio {current_pfolio_sR}')
            result = False
        



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

        """
        portfolio return with i stocks, Rp = sum (W_i x R_i)

        """

        total_weight = sum(stocks_picked)
        fitness = 0

        for i in range(0, len( stocks_picked )):
            if stocks_picked[ i ] >= 1:
                fitness += round((stocks_picked[ i ]/ total_weight)*(self._exp_rets[ i ])/100,2)#need to check this
                #fitness /= 100
                #fitness = round(fitness/100, 2)

        solution.fitness = fitness
        if solution.fitness > self._best_fit:
            self._best_fit = solution.fitness

            dict_toWrite = {'fitness': solution.fitness,
                            'stocks': solution.representation}
            f = open('stocks-PIP_PMX-09_Inv-09_RankS-15_p-20_I-1000.txt', "w+")
            
            f.write('stocks: ' + repr(dict_toWrite) + '\n')
            f.close()

        
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