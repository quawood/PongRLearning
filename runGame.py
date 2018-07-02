from game.Game import Game
import pygame


game = Game()

while game.isRunning:

    if game.isPlaying:
        game.move_player()
        game.move_ball()

    for event in pygame.event.get():
            game.check(event)

    if game.trainingIterations > 10000:
        game.draw()
        pygame.display.update()
        game.clock.tick(60)
