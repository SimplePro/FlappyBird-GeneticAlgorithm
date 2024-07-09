"""
입력 변수들간의 관계를 학습하기 어려운 경우에는 입력 변수들 사이의 관계를 내포하고 있는 정보를 제공하는 것이 도움이 된다.
입력 변수로 (bird_y_velocity, pipe_x, pipe_y, bird_y) 를 주었을 때 학습을 거의 못하였는데,
입력 변수로 (bird_y_velocity, horizontal distance between pipe and bird, vertical distance between pipe and bird)
를 주었을 때 바로 학습을 잘했다.
"""
import pygame
import random

import torch

from model import Model

from numpy.random import choice

from time import time

import pickle

from copy import deepcopy

import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Bird settings
BIRD_WIDTH = 46.875
BIRD_HEIGHT = 33.225
BIRD_X = 50
BIRD_Y = 300
BIRD_GRAVITY = 0.6
BIRD_JUMP_STRENGTH = -10
BIRD_MAX_ROTATION = 25
BIRD_ROTATION_SPEED = 2

# Pipe settings
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 170
PIPE_SPEED = 3

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

font = pygame.font.Font("assets/FlappyFont.TTF", 80)
gene_font = pygame.font.Font("assets/FlappyFont.TTF", 40)
# font = pygame.font.SysFont(None, 100)

background = pygame.image.load("assets/background.png")
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
screen.blit(background, (0, 0))

# Pipe class
class Pipe:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect_top = pygame.Rect(x, y - PIPE_HEIGHT - PIPE_GAP // 2, PIPE_WIDTH, PIPE_HEIGHT)
        self.rect_bottom = pygame.Rect(x, y + PIPE_GAP // 2, PIPE_WIDTH, PIPE_HEIGHT)

    def move(self):
        global score, score_text

        self.x -= PIPE_SPEED
        self.rect_top.x = self.x
        self.rect_bottom.x = self.x

        # print(self.x + PIPE_WIDTH)
        if (self.x + PIPE_WIDTH) == 120:
            score += 1
            score_text = font.render(str(score), True, (0, 0, 0))
            screen.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, SCREEN_HEIGHT * 0.13))

            # print(score)

    def draw(self):
        bottom_pipe_image = pygame.image.load("assets/pipe.png")
        bottom_pipe_image = pygame.transform.scale(bottom_pipe_image, (PIPE_WIDTH, PIPE_HEIGHT))
        top_pipe_image = pygame.transform.flip(bottom_pipe_image, False, True)

        bottom_rect = bottom_pipe_image.get_rect(center=self.rect_bottom.center)
        top_rect = top_pipe_image.get_rect(center=self.rect_top.center)

        screen.blit(bottom_pipe_image, bottom_rect.topleft)
        screen.blit(top_pipe_image, top_rect.topleft)

class Bird:

    def __init__(self):
        self.bird_rect = pygame.Rect(BIRD_X, BIRD_Y, BIRD_WIDTH, BIRD_HEIGHT)
        self.bird_y_velocity = 0

        self.model = Model()

        self.fitness = 0.01

    def run(self):

        pipe = sorted([pipe for pipe in pipes if (pipe.rect_top.x+PIPE_WIDTH) > 50], key=lambda pipe: pipe.rect_top.x)[0]
        
        # print(self.bird_rect.x)
        # if (self.bird_rect.x + BIRD_WIDTH) >= pipe.rect_top.x: self.fitness += 1
            
        # bird_center = (self.bird_rect.x + BIRD_WIDTH // 2, self.bird_rect.y + BIRD_HEIGHT // 2)
        pipe_center = (pipe.rect_top.x + PIPE_WIDTH // 2, pipe.rect_top.y + PIPE_HEIGHT + PIPE_GAP // 2)


        pred = self.model(
            torch.tensor([
                self.bird_y_velocity,
                pipe_center[0] - self.bird_rect.centerx,
                pipe_center[1] - self.bird_rect.centery,
            ]).type(torch.float)
        )

        if pred.item() > 0.5:
            self.bird_y_velocity = BIRD_JUMP_STRENGTH

        self.bird_y_velocity += BIRD_GRAVITY
        self.bird_rect.y += self.bird_y_velocity

        self.bird_angle = min(BIRD_MAX_ROTATION, max(-BIRD_MAX_ROTATION, -self.bird_y_velocity * BIRD_ROTATION_SPEED))
        
        center_distance = math.sqrt((pipe_center[0] - self.bird_rect.centerx)**2 + (pipe_center[1] - self.bird_rect.centery) ** 2)
        fitness = (time() - start_time) * PIPE_SPEED - center_distance

        if score >= 2024:
            with open("./weights/best_bird.pickle", "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def draw_rotated_bird(self):
        rotated_image = pygame.image.load("assets/bird.png")
        # rotated_image.fill(BLUE)
        rotated_image = pygame.transform.scale(rotated_image, (BIRD_WIDTH, BIRD_HEIGHT))
        rotated_image = pygame.transform.rotate(rotated_image, self.bird_angle)
        new_rect = rotated_image.get_rect(center=self.bird_rect.center)
        screen.blit(rotated_image, new_rect.topleft)

    def check_collision(self):
        global best_fitness, best_bird

        for pipe in pipes:
            if self.bird_rect.colliderect(pipe.rect_top) or self.bird_rect.colliderect(pipe.rect_bottom):

                # bird_center = (self.bird_rect.x + BIRD_WIDTH // 2, self.bird_rect.y + BIRD_HEIGHT // 2)
                pipe_center = (pipe.rect_top.x + PIPE_WIDTH // 2, pipe.rect_top.y + PIPE_HEIGHT + PIPE_GAP // 2)

                center_distance = math.sqrt((self.bird_rect.centerx - pipe_center[0])**2 + (self.bird_rect.centery - pipe_center[1])**2)

                self.fitness = (time() - start_time) * PIPE_SPEED - center_distance
                # print(self.fitness)
                if self.fitness >= best_fitness:
                    best_fitness = self.fitness
                    best_bird = self

                return True
        
        if self.bird_rect.top <= 0 or self.bird_rect.bottom >= SCREEN_HEIGHT:
            pipe = sorted([pipe for pipe in pipes if (pipe.rect_top.x+PIPE_WIDTH) > 50], key=lambda pipe: pipe.rect_top.x)[0]

            # bird_center = (self.bird_rect.x + BIRD_WIDTH // 2, self.bird_rect.y + BIRD_HEIGHT // 2)
            pipe_center = (pipe.rect_top.x + PIPE_WIDTH // 2, pipe.rect_top.y + PIPE_HEIGHT + PIPE_GAP // 2)

            center_distance = math.sqrt((self.bird_rect.centerx - pipe_center[0])**2 + (self.bird_rect.centery - pipe_center[1])**2)

            self.fitness = (time() - start_time) * PIPE_SPEED - center_distance

            if self.fitness >= best_fitness:
                best_bird = self
                best_fitness = self.fitness
            # dead_birds.append(self)
            return True
        
        return False

def crossover_model(model1, model2):
    new_model1 = deepcopy(model1)
    new_model2 = deepcopy(model2)

    # if random.randint(0, 1) == 1:
    new_model1.weight1, new_model2.weight1 = new_model2.weight1, new_model1.weight1
    # else:
    #     new_model1.weight2, new_model2.weight2 = new_model2.weight2, new_model2.weight2

    return new_model1, new_model2

# ------ TRAIN -------
# BIRDS_N = 20
# MUTATION_N = 18
# EPOCH = 200

# ------ TEST -------
BIRDS_N = 1
EPOCH = 1


# Define birds
# --------- TRAIN --------
# birds = [Bird() for i in range(BIRDS_N)]


# ------ TEST -------
birds = []
with open("weights/epoch80.pickle", "rb") as f:
    birds.append(pickle.load(f))



best_bird = None
best_fitness = -1e4

score = 0
# score_text = font.render(str(score), True, (255, 255, 255))
# screen.bilt(score_text, (SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT * 0.2))

for epoch in range(EPOCH):
    start_time = time()
    # Initialize variables
    pipes = []
    pipe_timer = 0
    running = True
    clock = pygame.time.Clock()

    pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
    new_pipe = Pipe(SCREEN_WIDTH, pipe_y)
    pipes.append(new_pipe)
    pipe_timer = 0


    # Game loop
    while len(birds) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
        for bird in birds:
            bird.run()
        
        # Add new pipes
        pipe_timer += 1
        if pipe_timer > 100:
            pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
            new_pipe = Pipe(SCREEN_WIDTH, pipe_y)
            pipes.append(new_pipe)
            pipe_timer = 0
        
        # Move pipes
        for pipe in pipes:
            pipe.move()
        
        # Remove off-screen pipes
        pipes = [pipe for pipe in pipes if pipe.rect_top.right > 0]
        
        birds = [bird for bird in birds if not bird.check_collision()]

        # Draw everything
        # screen.fill(WHITE)
        screen.blit(background, (0, 0))
        for pipe in pipes:
            pipe.draw()

        for bird in birds:
            bird.draw_rotated_bird()
        
        score_text = font.render(str(score), True, (0, 0, 0))
        screen.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, SCREEN_HEIGHT * 0.13))

        gene_text = gene_font.render("gene: " + str(epoch+1), True, (0, 0, 0))
        screen.blit(gene_text, (10, 10))

        pygame.display.flip()
        # clock.tick(60)

    # ver1
    # population_fitness = [bird.fitness for bird in dead_birds]

    # parent_probabilities = [fitness / sum(population_fitness) for fitness in population_fitness]

    # # print(sum(parent_probabilities))
    # for i in range((BIRDS_N - int(max(MUTATION_N, MIN_MUTATION_N)))//2):
    #     parents_idx = choice(list(range(len(dead_birds))), size=2, p=parent_probabilities)

    #     new_model1, new_model2 = crossover_model(
    #         dead_birds[parents_idx[0]].model,
    #         dead_birds[parents_idx[1]].model
    #     )

    #     new_bird1 = Bird()
    #     new_bird1.model = new_model1
        
    #     new_bird2 = Bird()
    #     new_bird2.model = new_model2

    #     birds.append(new_bird1)
    #     birds.append(new_bird2)

    # mutation_idx = choice(list(range(len(dead_birds))), size=int(max(MUTATION_N, MIN_MUTATION_N)), p=parent_probabilities)

    # for i in mutation_idx:
    #     new_weight1, new_weight2 = dead_birds[i].model.mutate()

    #     new_bird = Bird()
    #     new_bird.model.weight1 = new_weight1
    #     new_bird.model.weight2 = new_weight2

    #     birds.append(new_bird)    

    # MUTATION_N *= MUTATION_DECAY


    ### ver2
    # for i in range(BIRDS_N - MUTATION_N):
    #     new_bird = Bird()
    #     new_bird.model.weight1 = deepcopy(best_bird.model.weight1)
    #     new_bird.model.weight2 = deepcopy(best_bird.model.weight2)

    #     birds.append(new_bird)

    # for i in range(MUTATION_N):
    #     new_bird = Bird()
    #     mutated_weights = best_bird.model.mutate()

    #     new_bird.model.weight1 = mutated_weights[0]
    #     new_bird.model.bias1 = mutated_weights[1]
    #     new_bird.model.weight2 = mutated_weights[2]
    #     new_bird.model.bias2 = mutated_weights[3]

    #     birds.append(new_bird)

    # print(f"EPOCH: {epoch+1}/{EPOCH}, best_fitness: %d"%(best_fitness), "score:", score)

    # with open(f"weights/epoch{epoch+1}.pickle", mode="wb") as f:
    #     pickle.dump(best_bird, f, protocol=pickle.HIGHEST_PROTOCOL)

    # best_fitness = -1e4
    # score = 0

pygame.quit()

