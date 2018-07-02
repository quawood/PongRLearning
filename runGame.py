from game.Game import Game
import pygame


game = Game(loading=True)

while game.isRunning:

    if game.isPlaying:
        game.move_player()
        game.move_ball()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            for i in range(len(game.players)):
                game.players[i].save('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)

            print(game.trainingIterations)
            game.isRunning = False

    if game.trainingIterations > 0:
        for event in pygame.event.get():
            game.check(event)
        game.draw()
        pygame.display.update()
        game.clock.tick(60)