import pygame
import random
import numpy as np

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 100
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_GAP = 150

class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.vel = 0
        self.gravity = 0.5
        self.lift = -8
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT

    def update(self):
        self.vel += self.gravity
        self.y += self.vel
        if self.y < 0:
            self.y = 0
            self.vel = 0
        if self.y > SCREEN_HEIGHT - GROUND_HEIGHT - self.height:
            self.y = SCREEN_HEIGHT - GROUND_HEIGHT - self.height
            self.vel = 0

    def flap(self):
        self.vel = self.lift

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.top = random.randint(50, SCREEN_HEIGHT - GROUND_HEIGHT - self.gap - 50)
        self.bottom = self.top + self.gap
        self.passed = False

    def update(self):
        self.x -= 3

    def get_top_rect(self):
        return pygame.Rect(self.x, 0, self.width, self.top)

    def get_bottom_rect(self):
        return pygame.Rect(self.x, self.bottom, self.width, SCREEN_HEIGHT - GROUND_HEIGHT - self.bottom)

    def off_screen(self):
        return self.x + self.width < 0

class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.reset()

    def reset(self):
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH + 100)]
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        # action: 0 = do nothing, 1 = flap
        reward = 1  # reward for staying alive
        if action == 1:
            self.bird.flap()
        self.bird.update()
        add_pipe = False
        rem = []
        for pipe in self.pipes:
            pipe.update()
            if pipe.off_screen():
                rem.append(pipe)
            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward += 10  # reward for passing a pipe
                add_pipe = True
            if self.bird.get_rect().colliderect(pipe.get_top_rect()) or self.bird.get_rect().colliderect(pipe.get_bottom_rect()):
                self.done = True
                reward = -100
        if add_pipe:
            self.pipes.append(Pipe(SCREEN_WIDTH))
        for r in rem:
            self.pipes.remove(r)
        if self.bird.y >= SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.height:
            self.done = True
            reward = -100
        if self.render_mode:
            self.render()
        return self.get_state(), reward, self.done, {"score": self.score}

    def get_state(self):
        # State: bird y, bird vel, next pipe x dist, next pipe top, next pipe bottom
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            next_pipe = Pipe(SCREEN_WIDTH)
        state = np.array([
            self.bird.y / SCREEN_HEIGHT,
            self.bird.vel / 10.0,
            (next_pipe.x - self.bird.x) / SCREEN_WIDTH,
            next_pipe.top / SCREEN_HEIGHT,
            next_pipe.bottom / SCREEN_HEIGHT
        ], dtype=np.float32)
        return state

    def render(self):
        if not self.render_mode:
            return
        BLUE = (0, 150, 255)
        BLACK = (0, 0, 0)
        self.screen.fill(BLUE)
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), pipe.get_top_rect())
            pygame.draw.rect(self.screen, (0, 255, 0), pipe.get_bottom_rect())
        pygame.draw.rect(self.screen, (255, 255, 0), self.bird.get_rect())
        pygame.draw.rect(self.screen, (222, 184, 135), (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        pygame.display.update()
        self.clock.tick(60)
