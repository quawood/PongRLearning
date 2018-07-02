from game.Game import Game
import pygame


game = Game(loading=True)
game.players[0].isAi = True
game.players[0].isTraining = False

training = False

for i in range(len(game.players)):
    if game.players[i].isTraining:
        training = True
        game.players[i - 1].isAi = True
        game.players[i - 1].agentAI = game.players[i].agentAI

while game.isRunning:

    if game.isPlaying:
        game.move_player()
        game.move_ball()

    if training:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                for i in range(len(game.players)):
                    game.players[i].save('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)

                print(game.trainingIterations)
                game.isRunning = False
    else:
        for event in pygame.event.get():
            game.check(event)

        game.draw()
        pygame.display.update()
        game.clock.tick(60)










