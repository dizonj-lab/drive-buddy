
from email import header
from operator import truediv
from turtle import window_height, window_width
import pygame


class Player:
    def __init__(self,name,fightertype,avatar):
        #print("init Placeholder only")
        self.name = name
        self.type = fightertype
        self.maxhealth = 100
        self.currenthealth = 100
        self.maxattack = 20
        self.maxdamage = 15
        self.experience = 20
        self.defense = 20
        self.rank = 2
        self.avatar = avatar
    
    def methodsGoesHere(self):
        pass

class BattleUI:
    def __init__(self):
 
        self.header = "/Users/joeldizon/Documents/Project_BattleGame/images/header.png"
        self.stage = "/Users/joeldizon/Documents/Project_BattleGame/images/stage1.png"
        self.scoreboard = "/Users/joeldizon/Documents/Project_BattleGame/images/scoreboard.png"
        self.window_width = 1185
        self.window_height = 800

        self.screen_header_size = (1185,124)
        self.screen_headerX = 0
        self.screen_headerY = 0

        self.screen_stage_size = (1185,550)
        self.screen_stageX = 0
        self.screen_stageY = 125

        self.screen_scoreboard_size = (1185,174)
        self.screen_scoreboardX = 0
        self.screen_scoreboardY = 626

    
    def paintScreen(self):
        pygame.display.set_caption("Game of Champions")
        screen = pygame.display.set_mode((self.window_width,self.window_height))
        
        screen_header = pygame.image.load(self.header)
        screen_header = pygame.transform.scale(screen_header,self.screen_header_size)
        screen.blit(screen_header,(self.screen_headerX,self.screen_headerY))

        screen_stage = pygame.image.load(self.stage)
        screen_stage = pygame.transform.scale(screen_stage,self.screen_stage_size)
        screen.blit(screen_stage,(self.screen_stageX,self.screen_stageY))
        
        screen_scoreboard = pygame.image.load(self.scoreboard)
        screen_scoreboard = pygame.transform.scale(screen_scoreboard,self.screen_scoreboard_size)
        screen.blit(screen_scoreboard,(self.screen_scoreboardX,self.screen_scoreboardY))

        pygame.display.update()



## MAIN PROGRAM ##

#player1 = Player("AAA","warrior","/locationOfPNG/PNGfile.png")
#print("test")
#print(player1.avatar)



pygame.init()

myUI = BattleUI()
myUI.paintScreen()
running = True

while running:
        #set RGB
        #screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                    running = False
          #Update display
        pygame.display.update()


