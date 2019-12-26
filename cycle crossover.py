#!/usr/bin/env python
# coding: utf-8

import numpy as np
from copy import deepcopy

#https://codereview.stackexchange.com/questions/226179/easiest-way-to-implement-cycle-crossover


solution1 = [1,2,3,4,5,6,7,8,9,10,11,12]
solution2 = [2,3,4,5,1,7,6,8,9,10,11,12]




class Chromosome():
    """
    Description of class `Chromosome` goes here
    """
    def __init__(self, genes, id_=None, fitness=-1):
        self.id_ = id_
        self.genes = genes
        self.fitness = fitness

    def describe(self):
        """
        Prints the ID, fitness, and genes
        """
        #print('ID=#{}, fitenss={}, \ngenes=\n{}'.format(self.id, self.fitness, self.genes))
        print(f"ID=#{self.id_}, Fitness={self.fitness}, \nGenes=\n{self.genes}")

    def get_chrom_length(self):
        """
        Returns the length of `self.genes`
        """
        return len(self.genes)

def cycle_crossover(pc):
    """
    This function takes two parents, and performs Cycle crossover on them. 
    pc: The probability of crossover (control parameter)
    """
    parent_one = Chromosome(genes=np.array(solution1), id_=0, fitness=125.2)
    parent_two = Chromosome(genes=np.array(solution2), id_=1, fitness=125.2)
    chrom_length = Chromosome.get_chrom_length(parent_one)
    print("\nParents")
    print("=================================================")
    Chromosome.describe(parent_one)
    Chromosome.describe(parent_two)
    offspring1 = Chromosome(genes=np.array([-1] * chrom_length), id_=0, fitness=130)
    offspring2 = Chromosome(genes=np.array([-1] * chrom_length), id_=1, fitness=130)

    if np.random.random() < pc:  # if pc is greater than random number
        p1_copy = parent_one.genes.tolist()
        p2_copy = parent_two.genes.tolist()
        swap = True
        count = 0
        pos = 0

        while True:
            if count > chrom_length:
                break
            for i in range(chrom_length):
                if offspring1.genes[i] == -1:
                    pos = i
                    break

            if swap:
                while True:
                    offspring1.genes[pos] = parent_one.genes[pos]
                    count += 1
                    pos = parent_two.genes.tolist().index(parent_one.genes[pos])
                    if p1_copy[pos] == -1:
                        swap = False
                        break
                    p1_copy[pos] = -1
            else:
                while True:
                    offspring1.genes[pos] = parent_two.genes[pos]
                    count += 1
                    pos = parent_one.genes.tolist().index(parent_two.genes[pos])
                    if p2_copy[pos] == -1:
                        swap = True
                        break
                    p2_copy[pos] = -1

        for i in range(chrom_length): #for the second child
            if offspring1.genes[i] == parent_one.genes[i]:
                offspring2.genes[i] = parent_two.genes[i]
            else:
                offspring2.genes[i] = parent_one.genes[i]

        for i in range(chrom_length): #Special mode
            if offspring1.genes[i] == -1:
                if p1_copy[i] == -1: #it means that the ith gene from p1 has been already transfered
                    offspring1.genes[i] = parent_two.genes[i]
                else:
                    offspring1.genes[i] = parent_one.genes[i]

    else:  # if pc is less than random number then don't make any change
        offspring1 = deepcopy(parent_one)
        offspring2 = deepcopy(parent_two)
    return offspring1, offspring2

if __name__ == '__main__':

    CROSS = cycle_crossover(1)
    print("\nChildren")
    print("=================================================")
    for index, _ in enumerate(CROSS):
        Chromosome.describe(CROSS[index])




x = np.random.random()
print(x)






