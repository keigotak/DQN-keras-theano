# -*- coding:utf-8 -*-
import pygame
import random
from PIL import Image
from PIL import ImageOps
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from game import Game
from agent import Agent


class Block(pygame.sprite.Sprite):
    def __init__(self, color, width, height):
        super(Block, self).__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()


class Gleaning(Game):
    def __init__(self):
        pygame.init()
        self.frame = 0
        self.screen_width = 400
        self.screen_height = 400
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        self.block_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()
        self.block_num = 50
        self.block_width = 20
        self.block_height = 20
        self.done = False
        self.won = False
        self.score = 0
        self.earned = 0
        self.grid_size = 80

        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

        for i in xrange(self.block_num):
            block = Block(self.BLACK, self.block_width, self.block_height)

            block.rect.x = random.randrange(self.screen_width - self.block_width)
            block.rect.y = random.randrange(self.screen_height - self. block_height)

            self.block_list.add(block)
            self.all_sprites_list.add(block)
        self.player = Block(self.GREEN, self.block_width, self.block_height)
        self.player.rect.x = random.randrange(self.screen_width)
        self.player.rect.y = random.randrange(self.screen_height)
        self.all_sprites_list.add(self.player)

        self.start_block_list = self.block_list

        self.clock = pygame.time.Clock()

    @property
    def name(self):
        return "Gleaning"

    @property
    def nb_actions(self):
        return 4

    def reset(self):
        self.score = 0
        self.frame = 0
        self.done = False
        self.won = False
        self.earned = 0

        self.block_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()

        for s_block in self.start_block_list:
            block = Block(self.BLACK, self.block_width, self.block_height)

            block.rect.x = s_block.rect.x
            block.rect.y = s_block.rect.y

            self.block_list.add(block)
            self.all_sprites_list.add(block)
        self.player = Block(self.GREEN, self.block_width, self.block_height)
        self.player.rect.x = random.randrange(self.screen_width)
        self.player.rect.y = random.randrange(self.screen_height)
        self.all_sprites_list.add(self.player)

        self.clock = pygame.time.Clock()

    def play(self, action):

        self.screen.fill(self.WHITE)

        action_size = 10
        if action == 0:
            pos_x = action_size
            pos_y = 0
        elif action == 1:
            pos_x = 0
            pos_y = action_size
        elif action == 2:
            pos_x = -action_size
            pos_y = 0
        elif action == 3:
            pos_x = 0
            pos_y = -action_size

        if self.player.rect.x < 0 or self.player.rect.x > self.screen_width - self.block_width:
            pass
        else:
            self.player.rect.x += pos_x

        if self.player.rect.y < 0 or self.player.rect.y > self.screen_height - self.block_height:
            pass
        else:
            self.player.rect.y += pos_y

        self.all_sprites_list.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

        self.frame += 1

    def get_state(self):
        fname = 'screenshot.bmp'
        pygame.image.save(self.screen, fname)
        img = Image.open(fname).resize((self.grid_size, self.grid_size))
        gray_img = ImageOps.grayscale(img)
        array_gray_img = np.array(gray_img)
        return array_gray_img

    def get_score(self):
        score = 0
        if self.player.rect.x < 0 or self.player.rect.x > self.screen_width - self.block_width:
            score -= 10

        if self.player.rect.y < 0 or self.player.rect.y > self.screen_height - self.block_height:
            score -= 10

        self.blocks_hit_list = pygame.sprite.spritecollide(self.player, self.block_list, True)
        if self.frame is not 0:
            if len(self.blocks_hit_list) > 0:
                score += 20 * len(self.blocks_hit_list)
                self.earned += len(self.blocks_hit_list)
        score -= 1
        self.score += score
        return score

    def is_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.done = True

        if self.score < -200:
            self.done = True
        return self.done

    def is_won(self):
        if len(self.blocks_hit_list) == self.block_num:
            self.won = True
        return self.won

    def terminate(self):
        pygame.quit()

    def get_earned(self):
        return self.earned

if __name__ == "__main__":

    print "create agent"


    print "initialize game"
    g = Gleaning()

    nb_frames = 4
    agent = Agent(model=model, memory_size=100000, nb_frames=nb_frames, grid_size=g.grid_size, nb_action=g.nb_actions)
    agent.train(g, batch_size=64, nb_epoch=10000, gamma=0.99, epsilon=0.1)
    agent.play(g)


    # num_episodes = 2000
    # for e in xrange(num_episodes):
    #     g.reset()
    #     observation = g.get_state()
    #     done = False
    #     agent.new_episode()
    #     total_cost = 0.0
    #     total_reward = 100
    #     while not done:
    #         action, value = agent.act(observation)
    #         g.play(action)
    #         observation = g.get_state()
    #         reward = g.get_score()
    #         done = g.is_over()
    #         total_cost = agent.observe(reward)
    #         if g.is_won():
    #             done = True
    #     print "episode,", e, ", earned blocks,", g.earned, ", mean cost,", total_cost/g.frame
    # g.terminate()