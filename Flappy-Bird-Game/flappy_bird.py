import pygame
import sys
import random

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 100
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP = 150

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 150, 255)

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

def draw_window(screen, bird, pipes, score):
    screen.fill(BLUE)
    for pipe in pipes:
        pygame.draw.rect(screen, (0, 255, 0), pipe.get_top_rect())
        pygame.draw.rect(screen, (0, 255, 0), pipe.get_bottom_rect())
    pygame.draw.rect(screen, (255, 255, 0), bird.get_rect())
    pygame.draw.rect(screen, (222, 184, 135), (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
    font = pygame.font.SysFont(None, 36)
    score_text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))
    pygame.display.update()

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 100)]
    score = 0
    running = True
    game_over = False
    first_game = True
    while not game_over:
        if first_game:
            countdown_font = pygame.font.SysFont(None, 72)
            for i in range(3, 0, -1):
                screen.fill(BLUE)
                draw_window(screen, bird, pipes, score)
                count_text = countdown_font.render(str(i), True, (255, 0, 0))
                screen.blit(count_text, (SCREEN_WIDTH // 2 - count_text.get_width() // 2, SCREEN_HEIGHT // 2 - count_text.get_height() // 2))
                pygame.display.update()
                pygame.time.delay(1000)
            # Small 'Go!' message
            screen.fill(BLUE)
            draw_window(screen, bird, pipes, score)
            go_text = countdown_font.render("Go!", True, (0, 200, 0))
            screen.blit(go_text, (SCREEN_WIDTH // 2 - go_text.get_width() // 2, SCREEN_HEIGHT // 2 - go_text.get_height() // 2))
            pygame.display.update()
            pygame.time.delay(700)
            first_game = False
            running = True
        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        bird.flap()
            bird.update()
            add_pipe = False
            rem = []
            for pipe in pipes:
                pipe.update()
                if pipe.off_screen():
                    rem.append(pipe)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    score += 1
                    add_pipe = True
                if bird.get_rect().colliderect(pipe.get_top_rect()) or bird.get_rect().colliderect(pipe.get_bottom_rect()):
                    running = False
            if add_pipe:
                pipes.append(Pipe(SCREEN_WIDTH))
            for r in rem:
                pipes.remove(r)
            if bird.y >= SCREEN_HEIGHT - GROUND_HEIGHT - bird.height:
                running = False
            draw_window(screen, bird, pipes, score)

        # Game Over Alert
        font = pygame.font.SysFont(None, 64)
        text = font.render("Game Over!", True, (255, 0, 0))
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 60))
        font2 = pygame.font.SysFont(None, 36)
        info = font2.render("Press R to Restart or Q to Quit", True, (0, 0, 0))
        screen.blit(info, (SCREEN_WIDTH // 2 - info.get_width() // 2, SCREEN_HEIGHT // 2))
        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        waiting = False
                    if event.key == pygame.K_r:
                        # Restart game
                        bird = Bird()
                        pipes = [Pipe(SCREEN_WIDTH + 100)]
                        score = 0
                        first_game = True
                        waiting = False

if __name__ == "__main__":
    main()
