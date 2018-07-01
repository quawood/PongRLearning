from game.Game import Game
import pygame


game = Game()

while game.isRunning:
    game.draw()
    if game.isPlaying:
        game.move_player()
        game.move_ball()

    for event in pygame.event.get():
            game.check(event)

    pygame.display.update()
    game.clock.tick(60)



