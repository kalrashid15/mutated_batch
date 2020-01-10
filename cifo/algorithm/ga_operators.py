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
        index1 = self._select_index(population = population)
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( population = population )

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
        for solution in population.solutions :
            if objective == ProblemObjective.Maximization:
                stop_position += (solution.fitness / total_fitness)
                #stop_position += solution.fitness
            elif objective == ProblemObjective.Minimization:
                stop_position += 1 / solution.fitness / solution_fitness_min

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
<<<<<<< HEAD
    #print(f" >> singlepoint: {singlepoint}")
    #print(f'type of solution ', type(solution1))
=======
    # print(f" >> singlepoint: {singlepoint}")

>>>>>>> c867c5c994dd5236608fc8a904d568e541666fa8
    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

        #print(f'print offspring1', offspring1)
        #print(f'print offspring2', offspring2)

        #print(f'types:', type(offspring1))

    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Partially Mapped Crossover: Hugo
def pmx_crossover( problem, solution1, solution2):

    
    StartCrossBar = np.random.randint(0,len(solution1.representation)-2)
    EndCrossBar = np.random.randint(StartCrossBar+1,len(solution1.representation)-1)

    print(StartCrossBar, EndCrossBar)

    parent1MidCross = solution1.representation[StartCrossBar:EndCrossBar]
    parent2MidCross = solution2.representation[StartCrossBar:EndCrossBar]



    temp_child1 = solution1.representation[:StartCrossBar] + parent2MidCross + solution1.representation[EndCrossBar:]

    temp_child2 = solution2.representation[:StartCrossBar] + parent1MidCross + solution2.representation[EndCrossBar:]

    switches = {}


    for i in range(len(parent1MidCross)):

        switches.setdefault(parent1MidCross[i], []).append(parent2MidCross[i])
        switches.setdefault(parent2MidCross[i], []).append(parent1MidCross[i])

    def solve(child, parentMidCross):

        for i in range(len(child)):
            if i in [x for x in range(StartCrossBar, EndCrossBar)]:
                continue
            if child[i] in switches.keys():

                if switches.get(child[i])[0] in parentMidCross:
                    #print(child[i])
                    num=switches.get(child[i])[0]
                    if switches.get(num)[0] in parentMidCross:
                        print(child[i])
                        child[i]=switches.get(num)[1]    

                    else:
                        child[i]=switches.get(num)[0]
                else:
                    child[i]=switches.get(child[i])[0]

        return child   
    
    offspring1 = solve(temp_child2,parent1MidCross)
    offspring2 = solve(temp_child1,parent2MidCross)
    
    return offspring1, offspring2


# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Cycle Crossover: Natalia

def cycle_crossover(problem, solution1, solution2):
    """
    This function takes two parents, and performs Cycle crossover on them. 
    pc: The probability of crossover (control parameter)
    """
    chrom_length = len(solution1.representation)
    #print(f" >> singlepoint: {singlepoint}")
    #print(f'type of solution ', type(solution1))
    parent1 = deepcopy(solution1) #solution1.clone()
    parent2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    #chrom_length = Chromosome.get_chrom_length(parent_one)
    print("\nParents")
    print("=================================================")
    print(parent1.representation)
    print(parent2.representation)
    
    #Chromosome.describe(parent_one)
    #Chromosome.describe(parent_two)
    
    #offspring1 = Chromosome(genes=np.array([-1] * chrom_length), id_=0, fitness=125.2)
    #offspring2 = Chromosome(genes=np.array([-1] * chrom_length), id_=1, fitness=125.2)

    Off1 = {
        'genes': np.array([-1] * chrom_length),
        'id': 0,
        'fitness': 132.2
    }
    
    Off2 = {
        'genes': np.array([-1] * chrom_length),
        'id': 1,
        'fitness': 132.2
    }
    
    
    
    if np.random.random() < 1:  # if pc is greater than random number
        p1_copy = P1['genes']
        p2_copy = P2['genes']
        swap = True
        count = 0
        pos = 0

        while True:
            if count > chrom_length:
                break
            for i in range(chrom_length):
                if Off1['genes'][i] == -1:
                    pos = i
                    break

            if swap:
                while True:
                    Off1['genes'][pos] = P1['genes'][pos]
                    count += 1
                    pos = P2['genes'].index(P1['genes'][pos])
                    if p1_copy[pos] == -1:
                        swap = False
                        break
                    p1_copy[pos] = -1
            else:
                while True:
                    Off1['genes'][pos] = P2['genes'][pos]
                    count += 1
                    pos = P1['genes'].index(P2['genes'][pos])
                    if p2_copy[pos] == -1:
                        swap = True
                        break
                    p2_copy[pos] = -1

        for i in range(chrom_length): #for the second child
            if Off1['genes'][i] == P1['genes'][i]:
                Off2['genes'][i] = P2['genes'][i]
            else:
                Off2['genes'][i] = P1['genes'][i]

        for i in range(chrom_length): #Special mode
            if Off1['genes'][i] == -1:
                if p1_copy[i] == -1: #it means that the ith gene from p1 has been already transfered
                    Off1['genes'][i] = P2['genes'][i]
                else:
                    Off1['genes'][i] = P1['genes'][i]
        offspring1 = Off1['genes']
        offspring2 = Off2['genes']
        print(f'type of offprint CC', offspring1)

    else:  # if pc is less than random number then don't make any change
        offspring1 = deepcopy(parent_one)
        offspring2 = deepcopy(parent_two)
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
