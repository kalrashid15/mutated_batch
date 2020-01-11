from random import uniform, randint, choices
from copy import deepcopy
import numpy as np

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population


###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>( problem, population_size ):

# -------------------------------------------------------------------------------------------------
# Random Initialization 
# -------------------------------------------------------------------------------------------------
def initialize_randomly( problem, population_size ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = problem.build_solution()
        
        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = problem.build_solution()
        
        s.id = [0, i]
        i += 1
        problem.evaluate_solution ( s )
        
        solution_list.append( s )

    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list )
    
    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_hc( problem, population_size ):
    pass

# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Simulated Annealing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_sa( problem, population_size ):
    pass

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
# TODO: implement Roulette Wheel for Minimization: Done by Rash and Hugo @21/12/19
class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """
    def select(self, population, objective, params):
        """
        select two different parents using roulette wheel
        """
        index1 = self._select_index(population, objective)
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( population, objective )

        return population.get( index1 ), population.get( index2 )



    def _select_index(self, population, objective):
        # for Min: calculate fitness 1/50, 1/30, 1/20 new values. Calculate values (sum of new values). Divide new values by sum of new values

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        solution_fitness_min = 0
        for solution in population.solutions:
            total_fitness += solution.fitness
            solution_fitness_min += 1/solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        
        """
        if objective == ProblemObjective.Maximization:

            print(objective)
            breakpoint
        """
        for solution in population.solutions :
            
            if objective == ProblemObjective.Maximization:
                stop_position += (solution.fitness / total_fitness)
                #stop_position += solution.fitness
            elif objective == ProblemObjective.Minimization:
                stop_position += 1 / solution.fitness / solution_fitness_min
            """
            Temp fix below
            
            stop_position += 1 / solution.fitness / solution_fitness_min
            """
            #stop_position +=(1/solution.fitness)
            if stop_position > wheel_position :
                break
            index += 1    

        return index
        
# -------------------------------------------------------------------------------------------------
# class RankSelection
# -------------------------------------------------------------------------------------------------
class RankSelection:
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is allocated selection probability with respect to its rank. Individuals are selected as per their selection probability. Rank selection is an exploration technique of selection.
    """
    def select(self, population, objective, params):
        # Step 1: Sort / Rank
        population = self._sort( population, objective )

        # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]
        rank_list = []

        for index in range(0, len(population)):
            for _ in range(0, index + 1):
                rank_list.append( index )

        print(f" >> rank_list: {rank_list}")       

        # Step 3: Select solution index
        index1 = randint(0, len( rank_list )-1)
        index2 = index1
        
        while index2 == index1:
            index2 = randint(0, len( rank_list )-1)

        return population.get( rank_list [index1] ), population.get( rank_list[index2] )

    
    def _sort( self, population, objective ):

        if objective == ProblemObjective.Maximization:
            for i in range (0, len( population )):
                for j in range (i, len (population )):
                    if population.solutions[ i ].fitness > population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap
                        
        else:    
            for i in range (0, len( population )):
                for j in range (i, len (population )):
                    if population.solutions[ i ].fitness < population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap

        return population

# -------------------------------------------------------------------------------------------------
# class TournamentSelection
# -------------------------------------------------------------------------------------------------
class TournamentSelection:  
    """
    """
    def select(self, population, objective, params):
        tournament_size = 2
        if "Tournament-Size" in params:
            tournament_size = params[ "Tournament-Size" ]

        index1 = self._select_index( objective, population, tournament_size )    
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( objective, population, tournament_size )

        return population.solutions[ index1 ], population.solutions[ index2 ]


    def _select_index(self, objective, population, tournament_size ): 
        
        index_temp      = -1
        index_selected  = randint(0, population.size - 1)

        if objective == ProblemObjective.Maximization: 
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness > population.solutions[ index_selected ].fitness:
                    index_selected = index_temp
        elif objective == ProblemObjective.Minimization:
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness < population.solutions[ index_selected ].fitness:
                    index_selected = index_temp            

        return index_selected         

###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover( problem, solution1, solution2):
    print(f"Problem: {problem}\nSolution1: {solution1}\nSolution2: {solution2}")
    singlepoint = randint(0, len(solution1.representation)-1)


    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]


    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Partially Mapped Crossover: Hugo
def pmx_crossover(problem, solution1, solution2):
    
    parent1 = deepcopy(solution1.representation)
    parent2 = deepcopy(solution2.representation)
    
    firstCrossPoint = np.random.randint(0,len(parent1)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(parent1)-1)

    #firstCrossPoint = 3
    #secondCrossPoint = 6

    print(firstCrossPoint, secondCrossPoint)

    parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

    temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]

    temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

    relations = []
    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    print(relations)

    def recursion1 (temp_child , firstCrossPoint , secondCrossPoint , parent1MiddleCross , parent2MiddleCross) :
        child = np.array([0 for i in range(len(parent1))])
        while True:
            for i,j in enumerate(temp_child[:firstCrossPoint]):
                c=0
                for x in relations:
                    if j == x[0]:
                        child[i]=x[1]
                        c=1
                        break
                if c==0:
                    child[i]=j
            j=0
            for i in range(firstCrossPoint,secondCrossPoint):
                child[i]=parent2MiddleCross[j]
                j+=1

            for i,j in enumerate(temp_child[secondCrossPoint:]):
                c=0
                for x in relations:
                    if j == x[0]:
                        child[i+secondCrossPoint]=x[1]
                        c=1
                        break
                if c==0:
                    child[i+secondCrossPoint]=j
            child_unique=np.unique(child)
            if len(child) <= len(child_unique):
                break
                #child=recursion1(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross)
        return(child)

    """
    def recursion2(temp_child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross):
        child = np.array([0 for i in range(len(parent1))])
        for i,j in enumerate(temp_child[:firstCrossPoint]):
            c=0
            for x in relations:
                if j == x[1]:
                    child[i]=x[0]
                    c=1
                    break
            if c==0:
                child[i]=j
        j=0
        for i in range(firstCrossPoint,secondCrossPoint):
            child[i]=parent1MiddleCross[j]
            j+=1

        for i,j in enumerate(temp_child[secondCrossPoint:]):
            c=0
            for x in relations:
                if j == x[1]:
                    child[i+secondCrossPoint]=x[0]
                    c=1
                    break
            if c==0:
                child[i+secondCrossPoint]=j
        child_unique=np.unique(child)
        if len(child)>len(child_unique):
            child=recursion2(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross)
        return(child)
    
    """
    child1=recursion1(temp_child1,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross)
    child2=recursion1(temp_child2,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross)

    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)
    
    offspring1.representation = child1
    offspring2.representation = child2
    
    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Cycle Crossover: Rashid

def cycle_crossover(problem, solution1, solution2):
    parent1 = deepcopy(solution1)
    parent2 = deepcopy(solution2)
    
    print(parent1)
    print(parent2)
    
    chrom_length = len(parent1.representation)
    print(chrom_length)
    
    offspring1 = deepcopy(parent1)
    offspring2 = deepcopy(parent2)
    
    offspring1.representation = [-1] * chrom_length
    offspring2.representation = [-1] * chrom_length

    
    p1_copy = deepcopy(parent1)
    p2_copy = deepcopy(parent2)
    swap = True
    count = 0
    pos = 0
    
    while True:
        if count > chrom_length:
            break
        for i in range(chrom_length):
            if offspring1.representation[i] == -1:
                pos = i
                break
    
        if swap:
                while True:
                    offspring1.representation[pos] = parent1.representation[pos]
                    count += 1
                    pos = parent2.representation.index(parent1.representation[pos])
                    if p1_copy.representation[pos] == -1:
                        swap = False
                        break
                    p1_copy.representation[pos] = -1
        else:
            while True:
                parent1.representation[pos] = parent2.representation[pos]
                count += 1
                pos = parent1.representation.index(parent2.representation[pos])
                if p2_copy.representation[pos] == -1:
                    swap = True
                    break
                p2_copy.representation[pos] = -1

        for i in range(chrom_length): #for the second child
            if offspring1.representation[i] == parent1.representation[i]:
                offspring2.representation[i] = parent2.representation[i]
            else:
                offspring2.representation[i] = parent1.representation[i]

        for i in range(chrom_length): #Special mode
            if offspring1.representation[i] == -1:
                if p1_copy.representation[i] == -1: #it means that the ith gene from p1 has been already transfered
                    offspring1.representation[i] = parent2.representation[i]
                else:
                    offspring1.representation[i] = parent1.representation[i]

    else:  # if pc is less than random number then don't make any change
        offspring1 = deepcopy(parent1)
        offspring2 = deepcopy(parent2)
    return offspring1, offspring2
###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation( problem, solution):
    singlepoint = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        try:
            temp = deepcopy( encoding.encoding_data )

            temp.pop( solution.representation[ singlepoint ] )

            gene = temp[0]
            if len(temp) > 1 : gene = choices( temp )  

            solution.representation[ singlepoint ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )     

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
#TODO: Implement Swap mutation: Done by Rashid @21/12/2019
def swap_mutation( problem, solution):
    first_point = randint(0, len(solution.representation) -1)
    second_point = first_point

    while(second_point == first_point):
        second_point = randint(0, len(solution.representation) -1)
    
    gene_1 = solution.representation[first_point]
    gene_2 = solution.representation[second_point]

    solution.representation[first_point] = gene_2
    solution.representation[second_point] = gene_1
 
    return solution

    #I am pretty sure we dont need to bring in encoding here.
    #But if we do... I will take a look at it later.
    

###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population ):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population ):
    

    if problem.objective == ProblemObjective.Minimization :
        if current_population.fittest.fitness < new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]
    
    elif problem.objective == ProblemObjective.Maximization : 
        if current_population.fittest.fitness > new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)
