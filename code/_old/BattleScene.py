import pygame

global screen
global window_width
global window_height

class Player:
    def __init__(self):
        #print("init Placeholder only")
        self.name = ""
        self.type = ""
        self.maxhealth = 100
        self.currenthealth = 100
        self.maxattack = 20
        self.maxdamage = 15
        self.experience = 20
        self.defense = 20
        self.rank = 2
        self.playerAvatar = "/Users/joeldizon/Documents/Project_BattleGame/assets/char1.png"
        self.playerProfile = "/Users/joeldizon/Documents/Project_BattleGame/assets/char1_head.png"
   
    def methodsGoesHere(self):
        pass
 
class BattleUI:
    def __init__(self):
 
        self.header = "/Users/joeldizon/Documents/Project_BattleGame/assets/header.png"
        self.screen_header_size = (1185,124)
        self.screen_headerX = 0
        self.screen_headerY = 0

        self.stage = "/Users/joeldizon/Documents/Project_BattleGame/assets/stage1.png"
        self.screen_stage_size = (1185,500)
        self.screen_stageX = 0
        self.screen_stageY = 125

        self.scoreboard = "/Users/joeldizon/Documents/Project_BattleGame/assets/scoreboard.png"
        self.screen_scoreboard_size = (1185,174)
        self.screen_scoreboardX = 0
        self.screen_scoreboardY = 626

    def setHeader(self):
        screen_header = pygame.image.load(self.header)
        screen_header = pygame.transform.scale(screen_header,self.screen_header_size)
        screen.blit(screen_header,(self.screen_headerX,self.screen_headerY))

    def setStage(self):
        screen_stage = pygame.image.load(self.stage)
        screen_stage = pygame.transform.scale(screen_stage,self.screen_stage_size)
        screen.blit(screen_stage,(self.screen_stageX,self.screen_stageY))
        
    def setScoreboard(self):
        screen_scoreboard = pygame.image.load(self.scoreboard)
        screen_scoreboard = pygame.transform.scale(screen_scoreboard,self.screen_scoreboard_size)
        screen.blit(screen_scoreboard,(self.screen_scoreboardX,self.screen_scoreboardY))

class PlayerCharacters():
    def __init__(self):
        self.name = ""
        self.playerType = ["Human","Machine"]
        self.locX = 0
        self.locY = 100
        self.charAvatar = "/Users/joeldizon/Documents/Project_BattleGame/assets/char{}.png"
        self.charHead = "/Users/joeldizon/Documents/Project_BattleGame/assets/char{}_head.png"
        self.charAvatarSize = (450,580)
        self.charHeadSize = (65,7)
        self.SCREEN_FLIP_FACTOR = 600

    def SetCharater(self,playerType,charIcon,locX,locY,charSize):
        self.locY = locY
        self.locX = locX

        if (playerType=="MACHINE"):
            self.locX = locX + self.SCREEN_FLIP_FACTOR
 
        playChar = pygame.image.load(charIcon)
        playChar = pygame.transform.scale(playChar,charSize)
        screen.blit(playChar,(self.locX,self.locY))

 
class PlayerTeams:
    def __init__(self):
        self.name = ""
        self.playerType = ["Human","Machine"]
        self.locX = 0
        self.locY = 0
        self.AvatarSize = (100,100)
        self.AvatarHuman = "/Users/joeldizon/Documents/Project_BattleGame/assets/Human_Head.png"
        self.AvatarMachine = "/Users/joeldizon/Documents/Project_BattleGame/assets/Machine_Head.png"
        self.Avatar = ""
        self.coinLevel = 0
        self.GOLD = (255, 215, 0)

    def test_func(self):
        return False

    def SetAvatar(self, name,teamType,locX,locY):
        self.locX = locX
        self.locY = locY
        x = self.test_func()

        if (teamType=="HUMAN"):
            self.Avatar = self.AvatarHuman
        else:
            self.Avatar = self.AvatarMachine
 
        avatar = pygame.image.load(self.Avatar)
        avatar = pygame.transform.scale(avatar,self.AvatarSize)
        screen.blit(avatar,(self.locX,self.locY))

    def SetCoins(self,coins,locX,locY):
        self.coinLevel = coins
        #pygame.display.set_caption(self.coinLevel)

        font = pygame.font.SysFont(None, 24)
        img = font.render(coins, True, self.GOLD)
        screen.blit(img, (locX, locY))

class PlayerSelectionButtons(pygame.sprite.Sprite):
    def __init__(self, name, playerType,direction,locX, locY):
        self.name = name
        self.playerType = playerType
        self.direction = direction
        self.locX = locX
        self.locY = locY
        self.btnLeft= "/Users/joeldizon/Documents/Project_BattleGame/assets/arrow_left.png"
        self.btnRight="/Users/joeldizon/Documents/Project_BattleGame/assets/arrow_right.png"
        self.btnSelect_size = (40,80)
        self.btnIcon = ""
    
    def SetButtons(self):
        if (self.direction=="LEFT"):
            self.btnIcon = self.btnLeft
        else:
            self.btnIcon = self.btnRight

        btn = pygame.image.load(self.btnIcon).convert()
        btn = pygame.transform.scale(btn,self.btnSelect_size)
        screen.blit(btn,(self.locX,self.locY))

class ProgressBarScores:
    def __init__(self):
        self.name = ""
        self.playerType = ""
        self.playerIndex = ""
        self.pvalue = 100
        self.locX = 0
        self.locY = 0
        self.pbarIconGreen = "/Users/joeldizon/Documents/Project_BattleGame/assets/progress_green.png"
        self.pbarIconRed="/Users/joeldizon/Documents/Project_BattleGame/assets/progress_red.png"
        self.pbarIconWhite="/Users/joeldizon/Documents/Project_BattleGame/assets/progress_white.png"
        self.pbarSize = (65,6)
        self.pbarSizeW = (65,7)
        self.pbarIcon = ""
    
    def SetProgressBar(self,pvalue,locX,locY):
        self.locX = locX
        self.locY = locY
        if (pvalue > 45):
            self.pbarIcon = self.pbarIconGreen
        else:
            self.pbarIcon = self.pbarIconRed
        self.pbarSize = ((65/100)*pvalue,6)
        #Erase existing progress bar
        progress = pygame.image.load(self.pbarIconWhite)
        progress = pygame.transform.scale(progress,self.pbarSizeW)
        screen.blit(progress,(self.locX,self.locY))
        #Draw latest progress bar
        progress = pygame.image.load(self.pbarIcon)
        progress = pygame.transform.scale(progress,self.pbarSize)
        screen.blit(progress,(self.locX,self.locY))
       # pass


## Global ##

window_width = 1185
window_height = 800
screen = pygame.display.set_mode((window_width,window_height))
pygame.display.set_caption("Game of Champions")       

#pygame.display.flip()

pygame.init()

myUI = BattleUI()
#myUI.paintScreen()
myUI.setHeader()
myUI.setStage()
myUI.setScoreboard()


# players = [Player() for i in range(6)]
# player[0] = Player("AAA","warrior","/locationOfPNG/PNGfile.png")
# player2 = Player("AAA","warrior","/locationOfPNG/PNGfile.png")
# player3 = Player("AAA","warrior","/locationOfPNG/PNGfile.png")
# player4 = Player("AAA","warrior","/locationOfPNG/PNGfile.png")

#myUI.setProgress(1,"H",60)

# # Set navigation buttons for Player Selections
# btnLeftHuman = PlayerSelectionButtons("HumanLeftSel","Human","LEFT",80,300)
# btnLeftHuman.SetButtons()
# btnRightHuman = PlayerSelectionButtons("HumanRightSel","Human","RIGHT",450,300)
# btnRightHuman.SetButtons()
# btnLeftMachine = PlayerSelectionButtons("MachineLeftSel","Machine","LEFT",700,300)
# btnLeftMachine.SetButtons()
# btnRightMachine = PlayerSelectionButtons("MachineRightSel","Machine","RIGHT",1050,300)
# btnRightMachine.SetButtons()

#Set Progress Bars coordinates for the scoreboard
pbarX = [265,265,265,265,265,265,438,438,438,438,438,438,751,751,751,751,751,751,924,924,924,924,924,924]
pbarY = [652,669,703,721,756,773,652,669,703,721,756,773,652,669,703,721,756,773,652,669,703,721,756,773]
#Instantiate a list of progress bars
pbars = [ProgressBarScores() for i in range(24)]
#Set the initial values of progress bars
x=0
for pbar in pbars:
    pbar.SetProgressBar(100,pbarX[x],pbarY[x])
    x +=1
#Update progress on progress bars
i=8
p=25
pbars[i].SetProgressBar(p,pbarX[i],pbarY[i])
#** end of progress bar**

#Set the Team's Avatar
avatar = PlayerTeams()
avatar.SetAvatar("Human","HUMAN",50,650)
avatar.SetAvatar("Machine","MACHINE",1010,650)
avatar.SetCoins("0",115,770)
avatar.SetCoins("0",1070,770)
# End of Avatar module

#Characters
charAvatar = [
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char1.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char2.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char3.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char4.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char5.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char6.png",
    "/Users/joeldizon/Documents/Project_BattleGame/assets/char7.png"]

charX = [80,90,100,100,100,180,110]
charY = [100,105,140,150,120,130,130]
charSize = [(395,559),(349,505),(252,449),(388,439),(300,500),(178,484),(298,459)]

playerMachineChar = PlayerCharacters()
playerHumanChar = PlayerCharacters()
m = 6
h = 0
playerHumanChar.SetCharater("HUMAN",charAvatar[h],charX[h],charY[h],charSize[h])
playerMachineChar.SetCharater("MACHINE",charAvatar[m],charX[m],charY[m],charSize[m])

#Navigation Buttons
# btnLeft= "/Users/joeldizon/Documents/Project_BattleGame/assets/arrow_left2.png"
# #btnLeft= "/Users/joeldizon/Documents/Project_BattleGame/assets/button_left2.png"
# btnRight="/Users/joeldizon/Documents/Project_BattleGame/assets/arrow_right.png"
# btnSelect_size = (40,80)
# x=80
# y=300 
# btnLeftHuman = pygame.image.load(btnLeft).convert_alpha()
# #btnLeftHuman = pygame.transform.scale(btnLeftHuman,btnSelect_size)
# screen.blit(btnLeftHuman,(x,y))
#pygame.display.flip() 

# btnRightHuman = pygame.image.load(btnRight).convert_alpha()
# btnRightHuman = pygame.transform.scale(btnRightHuman,btnSelect_size)
# screen.blit(btnRightHuman,(450,300))

# btnLeftMachine = pygame.image.load(btnLeft).convert()
# btnLeftMachine = pygame.transform.scale(btnLeftMachine,btnSelect_size)
# screen.blit(btnLeftMachine,(700,300))

# btnRightMachine = pygame.image.load(btnRight).convert()
# btnRightMachine = pygame.transform.scale(btnRightMachine,btnSelect_size)
# screen.blit(btnRightMachine,(1050,300))

#TODO:
#Create 4 Human Players
#Create 4 Machine Players

#Set the Avatars for 3 x 3 Players

#Set the Initial Progress Bars for 3 x 3 Players

#Set the Buttons Attack, Heal and Skip Buttons

#Machine selecting opponent

#
running = True

while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False           
            if event.type == pygame.KEYDOWN:
                print (event.type)
                if event.key == pygame.K_a:
                    print ("a is pressed")
                    if (h>0):
                        h -= 1 
                        myUI.setStage()
                        playerHumanChar.SetCharater("HUMAN",charAvatar[h],charX[h],charY[h],charSize[h])
                        playerHumanChar.SetCharater("MACHINE",charAvatar[m],charX[m],charY[m],charSize[m])
                if event.key == pygame.K_f:
                    if (h<=5):
                        h += 1
                        print ("f is pressed")
                        myUI.setStage()
                        playerHumanChar.SetCharater("HUMAN",charAvatar[h],charX[h],charY[h],charSize[h])
                        playerHumanChar.SetCharater("MACHINE",charAvatar[m],charX[m],charY[m],charSize[m])
                if event.key == pygame.K_h:
                    print ("a is pressed")
                    if (m>0):
                        m -= 1 
                        myUI.setStage()
                        playerHumanChar.SetCharater("HUMAN",charAvatar[h],charX[h],charY[h],charSize[h])
                        playerHumanChar.SetCharater("MACHINE",charAvatar[m],charX[m],charY[m],charSize[m])
                if event.key == pygame.K_l:
                    if (m<=5):
                        m += 1
                        print ("f is pressed")
                        myUI.setStage()
                        playerHumanChar.SetCharater("HUMAN",charAvatar[h],charX[h],charY[h],charSize[h])
                        playerHumanChar.SetCharater("MACHINE",charAvatar[m],charX[m],charY[m],charSize[m])
                               
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     # Set the x, y postions of the mouse click
            #     x, y = event.pos
            #     print (x,y)
            #     print (btnLeftHuman.get_rect())
            #     if btnLeftHuman.get_rect().collidepoint(x, y):
            #         print('clicked on hhimage')
  
          #Update display
        pygame.display.update()

pygame.quit()
