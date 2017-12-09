import pygame
import pyautogui
import time
import math
from pygame.locals import *
from random import randint
#from math import hypot
import os, sys

#numero de neurons
inputNodes = 3
outputNodes = 2

class Genome(object):
  
    def __init__(self):
        self.neurons = []
        self.links = []


class Neuron(object):
  
    def __init__(self):
        self.value = 0.0
    
class Link(object):
  
  def __init__(self):
    self.into = 0
    self.out = 0
    self.weight = 0.0


file = open("redeneural.txt", "r") 
genome = Genome()
firstLine = file.readline()
arguments = firstLine.split()

for x in range(int(arguments[0])):
    genome.neurons.append(Neuron())

for i in range(0,int(arguments[1])):
    genome.links.append(Link())

i = 0
for line in file:
    arguments = line.split()
    genome.links[i].into = genome.neurons[int(arguments[0])]
    genome.links[i].out = genome.neurons[int(arguments[1])]
    genome.links[i].weight = float(arguments[2])
    i += 1


class Snake:
    def __init__(self):
        self.ml = 1
        self.xs = 0.5
        self.array_size = 50
        self.cord_y = 53
        self.cord_x = 47
        self.score = 0
        self.direction = 0
        self.naoPerdeu = True

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

        self.place_fruit()

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


        return [frente, esquerda, direita]

    def mapaLogistico(self, xn, r):
        return r * xn * (1 - xn)

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
            print("Score: " + str(self.score))
            print(self.getOverview())
            return False
            
        if new_head == fruit:
            self.score += 1
            self.place_fruit()
        else:
            tail = snake[-1]
            del snake[-1]

        snake.insert(0, new_head)
        return True

    def evaluateNetwork(self, genome, inputs):
  
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
            genome.neurons[n].value = self.sigmoid(genome.neurons[n].value)
        
        # Check the output nodes
        for o in range(len(genome.neurons) - outputNodes, len(genome.neurons)):
        
            if genome.neurons[o].value > 0:
                outs.append(True)
            else:
                outs.append(False)
          
        return outs

    def sigmoid(self, x):
        x = float("{0:.4f}".format(x))
        return 2 / (1 + (math.e ** (-4.9 * x))) - 1


    def artificial_step(self, genome):
        #time.sleep(0.1)
        output = self.evaluateNetwork(genome, self.getOverview())
        #time.sleep(0.1)
        print(output)
        print(self.getOverview())

        if(output == [True, False]):
            return 1
        elif (output == [False, True]):
            return 0
        else:
            return 2

    def make_step(self, pygame, genome):
        #e = pygame.event.wait()
        time.sleep(0.2)
        dirc = self.artificial_step(genome);
        if (dirc == 0):
            self.direction = (self.direction+1) % 4
        elif (dirc == 1):
            self.direction = (self.direction+3) % 4
        

        if not self.step(self.dirs[self.direction]):
            #pygame.quit()
            self.naoPerdeu = False


    def run(self, genome):

        pygame.init()
        global s, img, appleimage
        s = pygame.display.set_mode((self.cord_x * 10, self.cord_y * 10))
        pygame.display.set_caption('Snake')
        appleimage = pygame.Surface((10, 10))
        appleimage.fill((0, 255, 0))
        img = pygame.Surface((10, 10))
        img.fill((255, 0, 0))
        clock = pygame.time.Clock()

        qtdJogo = 1
        it = 0

        while it < qtdJogo:
            self.init()

            self.naoPerdeu = True
            while self.naoPerdeu:
                self.make_step(pygame, genome)
                s.fill((255, 255, 255)) 
                for bit in snake:
                    s.blit(img, (bit[0] * 10, (self.cord_y - bit[1] - 1) * 10))
                s.blit(appleimage, (fruit[0] * 10, (self.cord_y - fruit[1]-1) * 10))
                pygame.display.flip()

            it += 1
        pygame.quit()
            

if __name__ == "__main__":
    snake = Snake()
    snake.run(genome)
