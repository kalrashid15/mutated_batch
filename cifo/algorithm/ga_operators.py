from random import uniform, randint, choices
from copy import deepcopy
import random
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
            if solution.fitness > 0:

                solution_fitness_min += 1/solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0

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

        for index in range(0, len(population.solutions)):
            for _ in range(0, index + 1):
                rank_list.append( index )

        #print(f" >> rank_list: {rank_list}")       

        # Step 3: Select solution index
        index1 = randint(0, len( rank_list )-1)
        index2 = index1
        
        while index2 == index1:
            index2 = randint(0, len( rank_list )-1)

        return population.get( rank_list [index1] ), population.get( rank_list[index2] )

    
    def _sort( self, population, objective ):

        if objective == ProblemObjective.Maximization:
            for i in range (0, len( population.solutions )):
                for j in range (i, len (population.solutions )):
                    if population.solutions[ i ].fitness > population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap
                        
        else:    
            for i in range (0, len( population.solutions )):
                for j in range (i, len (population.solutions )):
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
    #print(f"Singlepoint Crossover\nProblem: {problem}\nSolution1: {solution1}\nSolution2: {solution2}")
    singlepoint = randint(0, len(solution1.representation)-1)


    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    #offspring2.representation[4] = 7
    #print(offspring1.representation, offspring2.representation)
    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Partially Mapped Crossover: Hugo and Pedro
def pmx_crossover(problem, solution1, solution2):
    """
    """
    #print('chopped!!!!')
    parent1 = solution1.representation
    parent2 = solution2.representation
    
    size = min(len(parent1), len(parent2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[parent1[i]] = i
        p2[parent2[i]] = i
    
    # Choose crossover points
    crossoverpoint1 = random.randint(0, size)
    crossoverpoint2 = random.randint(0, size - 1)
    if crossoverpoint2 >= crossoverpoint1:
        crossoverpoint2 += 1
    else:  # Swap the two crossover points
        crossoverpoint1, crossoverpoint2 = crossoverpoint2, crossoverpoint1
  
    offspring1 = deepcopy(solution1)
    offspring2 = deepcopy(solution2)   
    
    # Apply crossover between cx points
    for i in range(crossoverpoint1, crossoverpoint2):
        # Keeping track of the selected values
        aux1 = parent1[i]
        aux2 = parent2[i]
        # Swapping the matched value
        offspring1.representation[i], offspring1.representation[p1[aux2]] = aux2, aux1
        offspring2.representation[i], offspring2.representation[p2[aux1]] = aux1, aux2
        # Position bookkeeping
        p1[aux1], p1[aux2] = p1[aux2], p1[aux1]
        p2[aux1], p2[aux2] = p2[aux2], p2[aux1]

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Cycle Crossover: Rashid

def cycle_crossover(problem, solution1, solution2):
    parent1 = deepcopy(solution1)
    parent2 = deepcopy(solution2)
    
    
    chrom_length = len(parent1.representation)
    #print(chrom_length)
    
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
                    if (parent1.representation[pos] in parent2.representation):
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

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        temp = deepcopy( solution.representation )
        # temp = [x for i,x in enumerate(temp) if i!= solution.representation[ singlepoint ]]
        # print("\n\n", temp, "\n\n")
        temp.pop(solution.representation[ singlepoint ])


        gene = temp[0]
        if len(temp) > 1 : gene = choices( temp )  

        solution.representation[ singlepoint ] = gene

        return solution

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
#TODO: Implement Swap mutation: Done by Rashid @21/12/2019

def swap_mutation(problem, solution):
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
    


# -------------------------------------------------------------------------------------------------
# Scramble mutation
# -----------------------------------------------------------------------------------------------
#TODO: Implement Swap mutation: Done by Rashid @11/01/2020
def scramble_mutation( problem, solution):
    first_point = randint(0, len(solution.representation) -1)
    second_point = first_point + randint(0, len(solution.representation) -1)

    chromosome = solution.representation

    #selecting set of genes for scrumbling
    genes = chromosome[first_point:second_point]

    #shuffling the genes
    random.shuffle(genes)

    #assigning the shuffled genes back to the chromosome
    chromosome[first_point:second_point] = genes

    solution.representation = chromosome

    return solution


# -------------------------------------------------------------------------------------------------
# Inverse mutation
# -----------------------------------------------------------------------------------------------

def inverse_mutation( problem, solution):
    first_point = randint(0, len(solution.representation) -1)
    second_point = first_point + randint(0, len(solution.representation) -1)

    chromosome = solution.representation

    #selecting set of genes for scrumbling
    genes = chromosome[first_point:second_point]

    #inversing the genes
    genes.reverse()

    #assigning the inverted genes back to the chromosome
    chromosome[first_point:second_point] = genes

    solution.representation = chromosome

    return solution







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
