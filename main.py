import numpy as np
import pygame
import random

from PIL import Image

h = 1        # spatial step width
k = 1        # time step width
dimx = 300   # width of the simulation domain
dimy = 300   # height of the simulation domain
cellsize = 2 # display size of a cell in pixel

walls = None
source = None

PULSE_AMPLITUDE = 120
WAVE_PROPAGATION_SPEED = 0.5   # The "original" wave propagation speed
ENERGY_CONSERVATION = 0.995

RANDOM_DROPS = False
DROP_RATE = 0.01

SOURCE_PULSE_DELAY = 120


def init_simulation():
    global dimx, dimy, walls, source
    img = Image.open('img/wall.png')      # Image representing non-passable walls in simulation
    dimx = img.width
    dimy = img.height
    img_matrix = np.asarray(img)
    img_matrix = np.transpose(img_matrix, (1, 0, 2))
    # 0 - of pixel is closer to black, 1 - if closer to white (judging by red channel value only)
    walls = np.zeros((dimx, dimy))
    walls[1:dimx-1, 1:dimy-1] = img_matrix[1:dimx-1, 1:dimy-1, 0] // 128

    source = np.zeros((dimx, dimy))
    source[1:dimx-1, 1:dimy-1] = (img_matrix[1:dimx-1, 1:dimy-1, 0] -
                                  img_matrix[1:dimx-1, 1:dimy-1, 1] -
                                  img_matrix[1:dimx-1, 1:dimy-1, 2]) // 170

    u = np.zeros((3, dimx, dimy))           # The three dimensional simulation grid
    alpha = np.zeros((dimx, dimy))          # wave propagation velocities of the entire simulation domain
    alpha[0:dimx, 0:dimy] = ((WAVE_PROPAGATION_SPEED*k) / h)**2  # will be set to a constant value of tau
    return u, alpha


def update(u, alpha):
    u[2] = u[1]
    u[1] = u[0]

    # This switch is for educational purposes. The fist implementation is approx 50 times slower in python!
    use_terribly_slow_implementation = False
    if use_terribly_slow_implementation:
        # Version 1: Easy to understand but terribly slow!
        for c in range(1, dimx-1):
            for r in range(1, dimy-1):
                u[0, c, r]  = alpha[c,r] * (u[1, c-1, r] + u[1, c+1, r] + u[1, c, r-1] + u[1, c, r+1] - 4*u[1, c, r])
                u[0, c, r] += 2 * u[1, c, r] - u[2, c, r]
    else:
        # Version 2: Much faster by eliminating loops
        u[0, 1:dimx-1, 1:dimy-1]  = (alpha[1:dimx-1, 1:dimy-1] * (u[1, 0:dimx-2, 1:dimy-1] +
                                            u[1, 2:dimx,   1:dimy-1] +
                                            u[1, 1:dimx-1, 0:dimy-2] +
                                            u[1, 1:dimx-1, 2:dimy] - 4*u[1, 1:dimx-1, 1:dimy-1])
                                        + 2 * u[1, 1:dimx-1, 1:dimy-1] - u[2, 1:dimx-1, 1:dimy-1]) * walls[1:dimx-1, 1:dimy-1]

    # Not part of the wave equation but I need to remove energy from the system.
    # The boundary conditions are closed. Energy cannot leave and the simulation keeps adding energy.
    u[0, 1:dimx-1, 1:dimy-1] *= ENERGY_CONSERVATION


def source_raindrops(u):
    u[0, 1:dimx-1, 1:dimy-1] = (
        (1 - source[1:dimx-1, 1:dimy-1]) * u[0, 1:dimx-1, 1:dimy-1] +
        source[1:dimx - 1, 1:dimy - 1] * PULSE_AMPLITUDE
    )


def place_raindrops(u):
    if random.random() < DROP_RATE:
        x = random.randrange(5, dimx-5)
        y = random.randrange(5, dimy-5)
        u[0, x-2:x+2, y-2:y+2] = PULSE_AMPLITUDE


def mouse_raindrops(x, y, u):
    u[0, x - 2:x + 2, y - 2:y + 2] = PULSE_AMPLITUDE


def main():
    u, alpha = init_simulation()
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8 )

    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    source_counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                mouse_raindrops(x//cellsize, y//cellsize, u)

        if RANDOM_DROPS:
            place_raindrops(u)

        source_counter += 1
        if source_counter >= SOURCE_PULSE_DELAY:
            source_raindrops(u)
            source_counter = 0

        update(u, alpha)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 0] *= walls[1:dimx, 1:dimy].astype(np.uint8)
        pixeldata[1:dimx, 1:dimy, 1] *= walls[1:dimx, 1:dimy].astype(np.uint8)
        pixeldata[1:dimx, 1:dimy, 2] *= walls[1:dimx, 1:dimy].astype(np.uint8)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    main()