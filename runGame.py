from game.Game import Game
import pygame


game = Game(loading=False)
game.players[0].isAi = True
game.players[1].isAi = True
game.training = True

while game.isRunning:
    if game.training:
        game.players[0].isTraining = True
        game.players[1].isTraining = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for i in range(len(game.players)):
                    game.players[i].save('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)

                print(game.trainingIterations)
                game.isRunning = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.training = False

    else:
        game.players[0].isTraining = False
        game.players[1].isTraining = False
        for event in pygame.event.get():
            game.check(event)
        game.draw()
        pygame.display.update()
        game.clock.tick(60)

    if game.isPlaying:
        game.move_players()
