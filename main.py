import numpy as np
import pygame
import random

from PIL import Image

h = 1        # spatial step width
k = 1        # time step width
dimx = 300   # width of the simulation domain
dimy = 300   # height of the simulation domain
cellsize = 2 # display size of a cell in pixel
walls = np.zeros((dimx, dimy))


def init_simulation():
    walls_img = Image.open('img/wall.png')      # Image representing non-passable walls in simulation
    global dimx, dimy, walls
    dimx = walls_img.width
    dimy = walls_img.height
    walls_data = np.asarray(walls_img)
    print(type(walls_data))
    print(walls_data.shape)
    print(walls_data)
    walls = np.zeros((dimx, dimy))
    # 0 - of pixel is closer to black, 1 - if closer to white (judging by red channel value only)
    walls[1:dimx-1, 1:dimy-1] = walls_data[1:dimx-1, 1:dimy-1, 0] // 128
    walls = walls.transpose()
    print(type(walls))
    print(walls.shape)
    print(walls)

    u = np.zeros((3, dimx, dimy))           # The three dimensional simulation grid
    c = 0.5                                 # The "original" wave propagation speed
    alpha = np.zeros((dimx, dimy))          # wave propagation velocities of the entire simulation domain
    alpha[0:dimx, 0:dimy] = ((c*k) / h)**2  # will be set to a constant value of tau
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
    u[0, 1:dimx-1, 1:dimy-1] *= 0.995


def place_raindrops(u):
    if random.random() < 0.02:
        x = random.randrange(5, dimx-5)
        y = random.randrange(5, dimy-5)
        u[0, x-2:x+2, y-2:y+2] = 120


def mouse_raindrops(x, y, u):
    u[0, x - 2:x + 2, y - 2:y + 2] = 120
def main():
    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))
    pygame.display.set_caption("Solving the 2d Wave Equation")

    u, alpha = init_simulation()
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8 )

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                mouse_raindrops(x//cellsize, y//cellsize, u)

        place_raindrops(u)
        update(u, alpha)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    main()