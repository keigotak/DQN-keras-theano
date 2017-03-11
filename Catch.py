# -*- coding:utf-8 -*-
import pygame
import random
from PIL import Image
from PIL import ImageOps
from game import Game
from agent import Agent
import numpy as np

class Block(pygame.sprite.Sprite):
    def __init__(self, color, width=20, height=20):
        super(Block, self).__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()


class Bar(pygame.sprite.Sprite):
    def __init__(self, color, width=100, height=20):
        super(Bar, self).__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()


class Catch(Game):
    def __init__(self):
        pygame.init()
        self.screen_width = 400
        self.screen_height = 400
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])
        self.clock = pygame.time.Clock()

        self.grid_size = 80

        self.score = 0
        self.frame = 0
        self.done = False
        self.won = False
        self.earned = 0
        self.dropped = False
        self.drop_freq = 0.8
        self.drop_speed = 10

        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

        self.block_width = 20
        self.block_height = 20
        self.bar_width = 100
        self.bar_height = 20

        self.block_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()
        self.block = Block(self.BLACK, self.block_width, self.block_height)
        self.player = Bar(self.BLACK)
        self.player.rect.x = (self.screen_width - self.bar_width) / 2
        self.player.rect.y = self.screen_height - self.bar_height
        self.all_sprites_list.add(self.player)

    @property
    def name(self):
        return "Catch"

    @property
    def nb_actions(self):
        return 3

    def reset(self):
        self.clock = pygame.time.Clock()

        self.score = 0
        self.frame = 0
        self.done = False
        self.won = False
        self.earned = 0
        self.dropped = False

        self.block_list = pygame.sprite.Group()
        self.all_sprites_list = pygame.sprite.Group()
        self.player = Bar(self.BLACK)
        self.player.rect.x = (self.screen_width - self.bar_width) / 2
        self.player.rect.y = self.screen_height - self.bar_height
        self.all_sprites_list.add(self.player)

    def play(self, action):

        self.screen.fill(self.WHITE)

        action_size = 20
        if action == 0:
            pos_x = action_size
        elif action == 1:
            pos_x = -action_size
        elif action == 2:
            pos_x = 0

        if self.player.rect.x + pos_x < 0 or self.player.rect.x + pos_x > self.screen_width - self.bar_width:
            pass
        else:
            self.player.rect.x += pos_x

        # drop the block
        if np.random.random() < self.drop_freq and not self.dropped:
            self.block.rect.x = np.random.random() * (self.screen_width - self.block_width)
            self.block.rect.y = 0
            self.block_list.add(self.block)
            self.all_sprites_list.add(self.block)
            self.dropped = True
        elif self.dropped:
            self.block.rect.y += self.drop_speed

        self.all_sprites_list.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

        self.frame += 1

    def get_state(self):
        fname = 'screenshot.bmp'
        pygame.image.save(self.screen, fname)
        img = Image.open(fname).resize((self.grid_size, self.grid_size))
        gray_img = ImageOps.grayscale(img)
        gray_img.save("screenshot_gray.bmp")
        array_gray_img = np.float32(np.array(gray_img) / 255.0)
        return array_gray_img

    def get_score(self):
        score = 0
        self.blocks_hit_list = pygame.sprite.spritecollide(self.player, self.block_list, True)
        if self.frame is not 0:
            if len(self.blocks_hit_list) > 0:
                score += 20 * len(self.blocks_hit_list)
                self.earned += len(self.blocks_hit_list)
                self.block_list.empty()
                self.dropped = False
        self.score += score
        return score

    def is_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
            elif event.type == pygame.MOUSEBUTTONUP:
                self.done = True

        if self.dropped and self.block.rect.y > self.screen_height:
            self.done = True
        return self.done

    def is_won(self):
        if self.earned > 100:
            self.won = True
        return self.won

    def terminate(self):
        pygame.quit()

    def get_earned(self):
        return self.earned

if __name__ == "__main__":

    print "initialize game"
    g = Catch()

    print "create agent"
    nb_frames = 4
    agent = Agent(memory_size=10000, nb_frames=nb_frames, nb_action=g.nb_actions, grid_size=g.grid_size)
    for ep in xrange(10000):
        agent.train(g, batch_size=64, nb_epoch=100, gamma=0.99, epsilon_rate=10000)
        agent.play(g)