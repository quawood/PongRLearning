from game.Game import Game
import pygame


game = Game(loading=True)
game.players[0].isAi = True
game.players[0].isTraining = True
game.players[1].isAi = True
game.players[1].isTraining = True
training = True

while game.isRunning:
    if game.isPlaying:
        game.move_ball()
        game.move_players()

    if training:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for i in range(len(game.players)):
                    game.players[i].save('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)

                print(game.trainingIterations)
                game.isRunning = False
            game.check(event)
        game.draw()
        pygame.display.update()
        game.clock.tick(60)
