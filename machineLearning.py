# NeuroEvolution of Augmenting Topologies (NEAT) implementation in Python 3
# Created by Keaton Zang
# Feel free to modify, redistribute, and use this code as you wish.
# Credit is nice but not required!
# 9 - 16 - 2017

import random
import math
from datetime import datetime

random.seed(datetime.now())

#numero de neurons
inputNodes = 3
outputNodes = 2

#numero de genomas por especie
population = 30

deltaDisjoint = 2.0
deltaWeights = 0.4
deltaThreshold = 1.0

#parametros de mutação
mutateConnections = .25
mutateWeights = 0.9
mutateNode = 0.25
mutateBias = 0.2
mutateLink = 2.0
mutateRemoveNode = 0.1
mutateRemoveLink = 0.4
stepSize = 0.1

staleThreshold = 15
crossoverChance = .75
mateRate = 0.2

class Snake:
    def __init__(self):
        self.ml = 1
        self.xs = 0.5
        self.cord_y = 53
        self.cord_x = 47
        self.score = 0
        self.direction = 0
        self.perdeu = False
        self.qtdStep = 0

        self.directions = {
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
            "UP": (0, 1),
            "DOWN": (0, -1),
        }

        self.dirs = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        snake, fruit = None, None
        pass

    def init(self):
        global snake
        snake = [ (10, 2), (10, 1), (10, 0)]
        self.score = 0
        self.ml = 1
        self.xs = 0.5
        self.direction = 0
        self.perdeu = False

        self.place_fruit()

    def incrementSteps(self, steps):
        self.qtdStep += steps

    def getPerdeu(self):
        return self.perdeu

    def getSteps(self):
        return self.qtdStep

    def getSnake(self):
        return snake

    def getFruit(self):
        return fruit

    def setSnake(self, snakeNew):
        snake = snakeNew

    def setFruit(self, fruitNew):
        fruit = fruitNew

    def getDistance(self):
        return math.hypot(fruit[0]-snake[0][0], fruit[0]-snake[0][1])

    def getOverview(self):
        head = snake[0]

        dirDefaut   = self.directions[self.dirs[self.direction]]
        dirDireita  = self.directions[self.dirs[(self.direction+1) % 4]]
        dirEsquerda = self.directions[self.dirs[(self.direction+3) % 4]]

        #frente, direita e esquerda
        headF = (head[0]+dirDefaut[0], head[1]+dirDefaut[1])
        headD = (head[0]+dirDireita[0], head[1]+dirDireita[1])
        headE = (head[0]+dirEsquerda[0], head[1]+dirEsquerda[1])

        esquerda = 0
        direita  = 0
        frente   = 0

        #algo a frente
        if (headF[0] < 0 or headF[0] >= self.cord_x or headF[1] < 0 or headF[1] >= self.cord_y or headF in snake
            ):
            frente = 1

        #algo a esquerda
        if (headE[0] < 0 or headE[0] >= self.cord_x or headE[1] < 0 or headE[1] >= self.cord_y or headE in snake
            ):
            esquerda = 1

        #algo a direita
        if (headD[0] < 0 or headD[0] >= self.cord_x or headD[1] < 0 or headD[1] >= self.cord_y or headD in snake
            ):
            direita = 1


        return (frente, esquerda, direita)

    def mapaLogistico(self, xn, r):
        return r * xn * (1 - xn)

    def getScore(self):
        return self.score

    def place_fruit(self, coord=None):
        global fruit
        if coord:
            fruit = coord
            return

        while True:
            xN = self.mapaLogistico(self.ml, self.xs)
            self.ml += 1
            self.xs += 0.5
            x = int((xN%self.cord_x))
            y = int((xN%self.cord_y))
            if (x, y) not in snake:
               fruit = x, y
               return

    def step(self, direction):
        old_head = snake[0]
        movement = self.directions[direction]
        new_head = (old_head[0]+movement[0], old_head[1]+movement[1])

        if (
                new_head[0] < 0 or
                new_head[0] >= self.cord_x or
                new_head[1] < 0 or
                new_head[1] >= self.cord_y or
                new_head in snake
            ):
            return False
            
        if new_head == fruit:
            self.score += 1
            self.place_fruit()
        else:
            tail = snake[-1]
            del snake[-1]

        snake.insert(0, new_head)
        return True

    def make_step(self, directionToMake):
        if directionToMake == 0: #Direita
            self.direction = (self.direction+1) % 4
        elif directionToMake == 1: #Esquerda
            self.direction = (self.direction+3) % 4

        if not self.step(self.dirs[self.direction]):
            self.perdeu = True

class Pool(object):
  
    def __init__(self):
        self.species = []
        self.generation = 0
        self.innovation = 0
        self.maxFitness = 0
        self.minComplexity = 50
    
class Species(object):
  
    def __init__(self):
        self.genomes = []
        self.topFitness = 0
        self.averageFitness = 0
        self.staleness = 0
    
class Genome(object):
  
    def __init__(self):
        self.neurons = []
        self.links = []
        self.fitness = 0
        self.mutationRates = [mutateConnections, mutateNode, mutateBias, mutateLink, mutateRemoveNode, mutateRemoveLink, stepSize]
    
        for n in range(1 + inputNodes + outputNodes):
            self.neurons.append(Neuron())
    
class Neuron(object):
  
    def __init__(self):
        self.value = 0.0
    
class Link(object):
  
  def __init__(self):
    self.into = 0
    self.out = 0
    self.weight = 0.0
    self.enabled = True
    self.innovation = 0

# Create a new pool  
snakeGame = Snake()
pool = Pool()
pool.species.append(Species())

# Create an initial "blank" population
for p in range(population):
    pool.species[0].genomes.append(Genome())

# Partially truncated sigmoid function
def sigmoid(x):
  x = float("{0:.4f}".format(x))
  return 2 / (1 + (math.e ** (-4.9 * x))) - 1

# Increment the innovation number for the pool 
def newInnovation():
  global pool
  pool.innovation += 1
  return pool.innovation

# Copy a link from one genome to another (ensuring that they've got different identities)
def copyLink(genome, newgenome, link):
  newlink = Link()
  newlink.into = newgenome.neurons[genome.neurons.index(link.into)]
  newlink.out = newgenome.neurons[genome.neurons.index(link.out)]
  newlink.weight = link.weight
  newlink.enabled = link.enabled
  newlink.innovation = link.innovation
  return newlink

# Copy a genome with a new identity
def copyGenome(genome):
  newgen = Genome()
  
  # Copy neurons
  for n in range(len(genome.neurons) - (inputNodes + outputNodes + 1)):
    newgen.neurons.append(Neuron())
  
  # Copy links
  for l in genome.links:
    newgen.links.append(copyLink(genome, newgen, l))
  
  # Copy mutation rates
  for r in range(len(genome.mutationRates)):
    newgen.mutationRates[r] = genome.mutationRates[r]
    
  return newgen
  

# Choose a random neuron, where "shift" dictates whether it should include input neurons or output neurons.
def randomNeuron(genome, shift, forceBias):
  
  # Include input nodes
  if shift:
    return random.choice(genome.neurons[0 : len(genome.neurons) - outputNodes])
  
  # Include output nodes
  else:
    return random.choice(genome.neurons[inputNodes + 1 : len(genome.neurons)])
    
  if forceBias:
    return genome.neurons[inputNodes + 1]

# Check if the given genome has a link
def hasLink(genome, link):
  for l in genome.links:
    if link.into == l.into and link.out == l.out:
      return True
  return False

# Check if the given link exists in the pool yet
def linkExists(genome, link):
  global pool
  
  for s in pool.species:
    for g in s.genomes:
      for l in g.links:
        if g.neurons.index(l.into) == genome.neurons.index(link.into) and g.neurons.index(l.out) == genome.neurons.index(link.out):
          return l.innovation

  return 0

# Evaluate a network with a given set of inputs  
def evaluateNetwork(genome, inputs):
  
    outs = []
  
    # Fill the input nodes
    for i in range(inputNodes):
        genome.neurons[i].value = inputs[i]
    
    # Include the bias node
    genome.neurons[inputNodes].value = 1
  
    # Iterate through the rest of the nodes, pulling along the links
    for n in range(inputNodes + 1, len(genome.neurons)):
        genome.neurons[n].value = 0.0
        for l in genome.links:
            if l.out == genome.neurons[n]:
                genome.neurons[n].value += l.weight * l.into.value
        
        # Apply the sigmoid function
        genome.neurons[n].value = sigmoid(genome.neurons[n].value)
    
    # Check the output nodes
    for o in range(len(genome.neurons) - outputNodes, len(genome.neurons)):
    
        if genome.neurons[o].value > 0:
            outs.append(True)
        else:
            outs.append(False)
      
    return outs
  
def checkNet(genome, into, out):
    output = evaluateNetwork(genome, into)
    if output == out:
        return True
    return False

# Score the fitness of a genome
def scoreFitness(genome, snake):
  
    # Mark fitness as scored
    fitness = 1
  
    # Evaluate several inputs, and compare them to expected outputs

    #output
    #[False, False] vira para trás, teoricamente nunca deve acontecer
    #[True, False] vira a esquerda
    #[False, True] vira a direita
    #[True, True] segue reto
    x = 0
    while(not snake.getPerdeu() and x < snake.getSteps()):
        

        #checa se chegou mais perto do objetivo
        snakeTemp = snake.getSnake()
        fruitTemp = snake.getFruit()
        distancia = snake.getDistance()
        atualSitu = snake.getOverview()

        if checkNet(genome, [0, 0, 0], [True, False]):
            snake.make_step(1)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 0, 0], [True, True]):
            snake.make_step(2)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 0, 0], [False, True]):
            snake.make_step(0)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 0, 1], [True, True]):
            snake.make_step(2)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 0, 1], [True, False]):
            snake.make_step(1)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 1, 0], [True, True]):
            snake.make_step(2)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [0, 1, 0], [False, True]):
            snake.make_step(0)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [1, 0, 0], [False, True]):
            snake.make_step(0)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)

        if checkNet(genome, [1, 0, 0], [True, False]):
            snake.make_step(1)
            if(distancia > snake.getDistance()): fitness += 10
            elif(distancia < snake.getDistance()): fitness -= 5
            snake.setSnake(snakeTemp)
            snake.setFruit(fruitTemp)


        if checkNet(genome, [0, 1, 0], [False, False]): fitness = 1


        snake.perdeu = False
        if checkNet(genome, atualSitu, [True, True]):
            snake.make_step(2)
        if checkNet(genome, atualSitu, [True, False]):
            snake.make_step(1)
        if checkNet(genome, atualSitu, [False, True]):
            snake.make_step(0)
        x += 1
  
    # Give the fitness special boosters, based on how well it does
    '''if fitness == 13*snake.getSteps():
        fitness += int(fitness*0.5)
    
    if fitness == 17*snake.getSteps():
        fitness = fitness*2'''

    fitness += 2*snake.getScore()
    
    genome.fitness = fitness
  
# Mutate a link's weights
def connectionMutation(genome):
  
  # Iterate through links
  for c in genome.links:
    # Slightly modify the weights
    if random.random() < mutateWeights:
      c.weight += random.random() * genome.mutationRates[6] * 2 - genome.mutationRates[6]
    # Assign a new, totally random weight
    else:
      c.weight = random.random() * 2 - 1
  
# Create a new link between two nodes
def linkMutation(genome, bias):
  
  # Create a new link
  newlink = Link()
  
  # Assign it a random in and out
  if bias:
    newlink.into = randomNeuron(genome, True, True)
    newlink.out = randomNeuron(genome, False, True)
    
    while genome.neurons.index(newlink.into) >= genome.neurons.index(newlink.out):
      newlink.into = randomNeuron(genome, True, True)
      newlink.out = randomNeuron(genome, False, True)
      
  else:
    newlink.into = randomNeuron(genome, True, False)
    newlink.out = randomNeuron(genome, False, False)
    
    while genome.neurons.index(newlink.into) >= genome.neurons.index(newlink.out):
      newlink.into = randomNeuron(genome, True, False)
      newlink.out = randomNeuron(genome, False, False)
  
  # Check if this genome already has the link
  if hasLink(genome, newlink):
    return genome
  
  # If it doesn't, create it.
  else:
    newlink.weight = random.random() * 2 - 1
    
    # Check if the link exists
    existance = linkExists(genome, newlink)
    
    # If it does, give it the same innovation number
    if existance != 0:
      newlink.innovation = existance
    # Otherwise, give it a new innovation number
    else:
      newlink.innovation = newInnovation()
    
    genome.links.append(newlink)
  
def nodeMutation(genome):
  if len(genome.links) > 0:
    
    # Create a new neuron and pick a link
    genome.neurons.insert(len(genome.neurons) - outputNodes, Neuron())
    lnk = random.choice(genome.links)
    
    # Generate the new link
    newlink = Link()
    newlink.weight = lnk.weight
    newlink.into = lnk.into
    newlink.out = genome.neurons[len(genome.neurons) - (outputNodes + 1)]
    
    # Check if the link exists
    existance = linkExists(genome, newlink)
    
    # If it does, give it the same innovation number
    if existance != 0:
      newlink.innovation = existance
    # Otherwise, give it a new innovation number
    else:
      newlink.innovation = newInnovation()
    
    genome.links.append(newlink)
    
    # Create the second link
    newlink = Link()
    newlink.weight = 1.0
    newlink.into = genome.neurons[len(genome.neurons) - (outputNodes + 1)]
    newlink.out = lnk.out
    
    # Check if the link exists
    existance = linkExists(genome, newlink)
    
    # If it does, give it the same innovation number
    if existance != 0:
      newlink.innovation = existance
    # Otherwise, give it a new innovation number
    else:
      newlink.innovation = newInnovation()
    
    genome.links.append(newlink)
    
    # Remove the old link
    genome.links.remove(lnk)

# Mutation to remove a random link
def linkDeleteMutation(genome):
  '''
  # Pick and remove a random link
  lnk = random.choice(genome.links)
  genome.links.remove(lnk)
  '''

# Mutation to remove a random node (and connected links)
def nodeDeleteMutation(genome):
  '''
  # Pick a random node
  if len(genome.neurons) > inputNodes + outputNodes + 1:
    node = random.choice(genome.neurons[inputNodes + 1 : len(genome.neurons) - outputNodes])
    
    # Remove all connected links
    for l in genome.links:
      if l.out == node or l.into == node:
        genome.links.remove(l)
    
    # Remove the node
    genome.neurons.remove(node)
  '''

def mutate(genome):
  
  # Modify the mutation rates
  for r in genome.mutationRates:
    if random.random() >= 0.5:
      r *= 1.05
    else:
      r *= 0.95
  
  # Weight mutation
  mval = genome.mutationRates[0]
  
  while mval > 0:
    if random.random() < mval:
      connectionMutation(genome)
    mval -= 1
  
  # Node mutation
  mval = genome.mutationRates[1]
  
  while mval > 0:
    if random.random() < mval:
      nodeMutation(genome)
    mval -= 1
  
  # Link mutation
  mval = genome.mutationRates[2]
  
  while mval > 0:
    if random.random() < mval:
      linkMutation(genome, False)
    mval -= 1
  
  # Bias mutation
  mval = genome.mutationRates[3]
  
  while mval > 0:
    if random.random() < mval:
      linkMutation(genome, True)
    mval -= 1
    
  # Node removal mutation
  mval = genome.mutationRates[4]
  
  while mval > 0:
    if random.random() < mval:
      nodeDeleteMutation(genome)
    mval -= 1
    
  # Link removal mutaiton
  mval = genome.mutationRates[5]
  
  while mval > 0:
    if random.random() < mval:
      linkDeleteMutation(genome)
    mval -= 1
    
# Crossover between two genomes
def crossover(pone, ptwo):
  
  child = copyGenome(pone)
  
  # Ensure that tempone is more fit
  if pone.fitness < ptwo.fitness:
    temptwo = pone
    tempone = ptwo
  else:
    tempone = pone
    temptwo = ptwo
  
  child = Genome()
  
  # Give the child more neurons
  for n in range(max(len(pone.neurons), len(ptwo.neurons)) - (inputNodes + outputNodes + 1)):
    child.neurons.append(Neuron())
  
  # Determine what links the second parent has (the less fit one)
  linkstwo = []
  
  for l in temptwo.links:
    linkstwo.append(l.innovation)
    
  # Create the links for the child
  for l in tempone.links:
    # Add this link from the weaker if both parents have it and luck allows
    if (l.innovation in linkstwo) and random.random() < 0.5:
      child.links.append(copyLink(temptwo, child, l))
    # Add this link from the stronger if only the stronger has it or luck allows
    else:
      child.links.append(copyLink(tempone, child, l))
      
  # Give the child mutation rates from the stronger parent
  for r in range(len(tempone.mutationRates)):
    child.mutationRates[r] = tempone.mutationRates[r]
  
  return child

# Create a child from a given species
def breedChild(species):
  
  totalfit = 0
  
  # Total the fitnesses of the genomes
  for g in species.genomes:
    totalfit += g.fitness
  
  # Find parents for the children (more fit parents have a higher chance of being chosen)
  pone = 0
  ptwo = 0
  
  ponechance = random.random()
  ptwochance = random.random()
  
  fitpar = 0
  
  # Iterate through the genomes
  for g in species.genomes:
    # Those with a higher fitness are more likely to reproduce
    if ponechance < fitpar + (g.fitness / totalfit) and pone == 0:
      pone = g
    else:
      fitpar += (g.fitness / totalfit)
  
  fitpar = 0
  
  # Repeat for the second parent 
  for g in species.genomes:
    if ponechance < fitpar + (g.fitness / totalfit) and ptwo == 0:
      ptwo = g
    else:
      fitpar += (g.fitness / totalfit)
  
  # Crossover, mutate, and speciate
  if random.random() < mateRate:
    child = crossover(pone, ptwo)
  else:
    child = copyGenome(pone)
  
  # Mutate and speciate the child
  mutate(child)
  speciate(child)
  
  return child
  
# Find the difference between the contained links
def disjointDifference(gone, gtwo):
  
  disjoints = 0
  
  # Fill the first innovations
  innovone = []
  for l in gone.links:
    innovone.append(l.innovation)
  
  # Fill the second innovations
  innovtwo = []
  for l in gtwo.links:
    innovtwo.append(l.innovation)
  
  # Find out how many extra links genome one has
  for d in innovone:
    if d not in innovtwo:
      disjoints += 1
  
  # Find out how many extra links genome two has
  for d in innovtwo:
    if d not in innovone:
      disjoints += 1
  
  # Find out who has the most extra links
  mostlinks = max(len(gone.links), len(gtwo.links))
  if mostlinks == 0:
    mostlinks = 1
  
  # Divide the total extras over the most
  return disjoints / mostlinks

# Determine the difference in link weights
def weightDifference(gone, gtwo):
  
  totalWeight = 0
  overlap = 0
  
  # Iterate through all of the links
  for lone in gone.links:
    for ltwo in gtwo.links:
      # Compare the innovations
      if lone.innovation == ltwo.innovation:
        # Add the difference of the weights
        totalWeight += abs(lone.weight - ltwo.weight)
        overlap += 1
  
  if overlap == 0:
    overlap = 1
  
  return totalWeight / overlap

# Check of two genomes are the same species
def sameSpecies(gone, gtwo):
  disjoint = deltaDisjoint * disjointDifference(gone, gtwo)
  weights = deltaWeights * weightDifference(gone, gtwo)
  
  return disjoint + weights < deltaThreshold
  
# Add a genome to a species
def speciate(genome):
  global pool
  
  foundSpecies = False
  
  # Iterate through the species
  for s in pool.species:
    # If the genome can be added, then add them
    if not foundSpecies and sameSpecies(genome, s.genomes[0]):
      s.genomes.append(genome)
      foundSpecies = True
  
  # If there are no species for this genome, then create a new one
  if not foundSpecies:
    newSpecies = Species()
    newSpecies.genomes.append(genome)
    pool.species.append(newSpecies)
  
# Find a species' average fitness
def speciesAverageFitness(species):
  total = 0
  
  # Add up each genome's global rank
  for g in species.genomes:
    total += g.fitness
  
  # Divide total rank by genomes
  species.averageFitness = total / len(species.genomes)

# Create one evolutionary iteration, removing and replacing one genome
def nextStep(snake):
  global pool
  worstAgent = pool.species[0].genomes[0]
  
  # Search the pool for the weakest genome and score average species' fitnesses
  for s in pool.species:
    for g in s.genomes:
      snake.init()
      scoreFitness(g, snake)
      if g.fitness < worstAgent.fitness:
        worstAgent = g
    speciesAverageFitness(s)
  
  # Remove the weakest organism
  for s in pool.species:
    if worstAgent in s.genomes:
      s.genomes.remove(worstAgent)
      if len(s.genomes) == 0:
        pool.species.remove(s)
      else:
        speciesAverageFitness(s)
  
  # Find a new species to breed the child from
  totalAvg = 0
  for s in pool.species:
    totalAvg += s.averageFitness
    
  specchance = random.random()
  
  spec = 0
  fitspec = 0
  
  # Iterate through the genomes
  for s in pool.species:
    # Those with a higher fitness are more likely to reproduce
    if specchance < fitspec + (s.averageFitness / totalAvg) and spec == 0:
      spec = s
    fitspec += (s.averageFitness / totalAvg)
  
  breedChild(s)

# Infinite loop to generate better and better genomes
snakeGame.incrementSteps(10000)
while True:
    

    pool.generation += 1
  
    # Create a new generation
    print("Generation " + str(pool.generation) + " on the way...")
    #input()
  
    # Iterate through each genome
    #while
    topfit = 0
    for s in pool.species:   
        for g in s.genomes:
            if topfit == 0: topfit = g
            snakeGame.init()
            # Score fitness and print out the strength and complexity
            scoreFitness(g, snakeGame)
            #print(str(g.fitness) + " ~ " + str(len(g.links)) + " ~ " + str(len(g.neurons)))
      
            # See if we have a new strongest genome
            if g.fitness > pool.maxFitness or (g.fitness == pool.maxFitness and len(g.links) + len(g.neurons) < pool.minComplexity):
                #print("Fitness: " + str(g.fitness))
                #print("Complexity: " + str(len(g.links) + len(g.neurons)))
                pool.maxFitness = g.fitness
                pool.minComplexity = len(g.links) + len(g.neurons)
                topfit = g
        
            if g.fitness > s.topFitness:
                s.topFitness = g.fitness

    file = open("redeneural.txt","w")
    file.write(str(len(topfit.neurons)) + " " + str(len(topfit.links)) + " " + str(topfit.fitness) + "\n")
    for l in topfit.links:
        file.write(str(topfit.neurons.index(l.into)) + " " + str(topfit.neurons.index(l.out)) + " " + str(l.weight) + "\n")
    file.close()


    '''print("Pool " + str(pool))
    print("Max fitness: " + str(pool.maxFitness))
    print("Species count: " + str(len(pool.species)))'''
    for s in pool.species:
        '''print(' ' * 4, end = '')
        print("Species " + str(s))
        print(' ' * 4, end = '')'''
        speciesAverageFitness(s)
        '''print("Average fitness: " + str(s.averageFitness))
        print(' ' * 4, end = '')
        print("Genome count: " + str(len(s.genomes)))'''
        '''for g in s.genomes:
            print(' ' * 8, end = '')
            print("Genome " + str(g))
            print(' ' * 8, end = '')
            print("Fitness: " + str(g.fitness))'''
  
    # Create a new generation
    for i in range(population):
        nextStep(snakeGame)

