import pygame
import random
import time
import datetime

# global variables
window_width = 1185
window_height = 800
screen = pygame.display.set_mode((window_width,window_height))
pygame.display.set_caption("Game of Champions")       

humans_coins = 0
robots_coins = 0
start_time = datetime.datetime.now()
end_time = datetime.datetime.now()
cur_index_ai = 0
cur_index_human = 0

# constants
WARR_MIN_DEP = 5
WARR_MAX_DEP = 10
TANK_MIN_DEP = 5
TANK_MAX_DEP = 15
WARR_MIN_ATK = 5
WARR_MAX_ATK = 20
TANK_MIN_ATK = 1
TANK_MAX_ATK = 10
EXTRA_MIN_DMG = -15
EXTRA_MAX_DMG = -5
HIGH_EXP = 1.5
LOW_EXP = 1.2
HIGH_DMG = 10
LOW_DMG = 0
MAX_EXP = 100
HEAL_MIN = 3
HEAL_MAX = 10

# initial ATK, DEP
WARR_ATK = random.randint(WARR_MIN_ATK, WARR_MAX_ATK)
TANK_ATK = random.randint(TANK_MIN_ATK, TANK_MAX_ATK)
WARR_DEP = random.randint(WARR_MIN_DEP, WARR_MAX_DEP)
TANK_DEP = random.randint(TANK_MIN_DEP, TANK_MAX_DEP)

# set list for players
human = []
ai = []

class Player:
    def __init__(self, name, profession, index,state,atk,dep,avatar,icon,posx,posy,avatarsize):
        self.index = index
        self.state = state
        self.name = name
        self.profession = profession
        self.healthpoint = 100
        self.atk = atk
        self.dep = dep
        self.experience = 0
        self.rank = 0
        self.avatar = avatar
        self.icon = icon   
        self.posx = posx
        self.posy = posy
        self.avatarsize = avatarsize 
  
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

        self.name = ""
        self.playerType = ["Human","Machine"]
        self.locX = 0
        self.locY = 0
        self.AvatarSize = (100,100)
        self.AvatarHuman = "/Users/joeldizon/Documents/Project_BattleGame/assets/Human_Head.png"
        self.AvatarAI = "/Users/joeldizon/Documents/Project_BattleGame/assets/Machine_Head.png"
        self.Avatar = ""
        self.coinLevel = 0
        self.GOLD = (255, 215, 0)
        self.SCREEN_FLIP_FACTOR = 600

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

        self.locX = 50
        self.locY = 650
        self.Avatar = self.AvatarHuman
        avatar = pygame.image.load(self.Avatar)
        avatar = pygame.transform.scale(avatar,self.AvatarSize)
        screen.blit(avatar,(self.locX,self.locY))

        self.locX = 1010
        self.locY = 650
        self.Avatar = self.AvatarAI
        avatar = pygame.image.load(self.Avatar)
        avatar = pygame.transform.scale(avatar,self.AvatarSize)
        screen.blit(avatar,(self.locX,self.locY))

    def updatecharater(self,objPlayer,playertype,cur_index):
        p = objPlayer[cur_index]
        init_prof = p.profession
        init_posx = p.posx
        init_posy = p.posy
        init_avatarsize = p.avatarsize
        init_avatar = p.avatar

        if (playertype=="ai"):
            init_posx = init_posx + self.SCREEN_FLIP_FACTOR
 
        playChar = pygame.image.load(init_avatar)
        playChar = pygame.transform.scale(playChar,init_avatarsize)
        screen.blit(playChar,(init_posx,init_posy))
    
    def update_uniticons(self,t_player,t_coord):
        i=0
        for p in t_player:
            t_icon = p.icon
            icon = pygame.image.load(t_icon)
            avatar = pygame.transform.scale(icon,(100,200))
            screen.blit(icon,(t_coord[i][0],t_coord[i][1]))
            i += 1

    def update_scoreboard(self,t_player,t_coord,t_fsize,widget):
        # update unit healthpoints 
        t_font = "calibri"   
        t_fontcolor = (10,10,10) # default. dark grey.    
        i=0
        for p in t_player:
            if widget == "EXP":
                t_value = p.experience
            elif widget == "HP":
                t_value = p.healthpoint
            elif widget == "RANK":
                t_value = p.rank
            elif widget == "NAME":
                t_value = p.name
                t_font = None
                t_fontcolor = (255, 215, 0)
            elif widget == "PROF":
                t_value = p.profession
                t_font = None
                t_fontcolor = (25,25,112)

            font = pygame.font.SysFont(t_font, t_fsize)
            img = font.render(str(t_value), True,t_fontcolor)
            screen.blit(img, (t_coord[i][0], t_coord[i][1]))       
            i += 1

    def refresh_score_data(self,human,ai):
        # update human coins
        locX = 115
        locY = 770
        font = pygame.font.SysFont(None, 25)
        img = font.render("0", True, (255, 215, 0))
        screen.blit(img, (locX, locY))       

        # update ai coins
        locX = 1070
        locY = 770
        font = pygame.font.SysFont(None, 25)
        img = font.render("0", True, (255, 215, 0))
        screen.blit(img, (locX, locY))       
        
        # update unit head icons (human player)
        t_coord = [[190,648],[365,648],[190,698],[365,698],[190,752],[365,752]]
        self.update_uniticons(human,t_coord)

        # update unit head icons (ai player)
        t_coord = [[674,648],[849,648],[674,698],[849,698],[674,752],[849,752]]
        self.update_uniticons(ai,t_coord)

        # update unit healthpoints (human player)
        coord = [[265,650],[440,650],[265,702],[440,702],[265,754],[440,754]]
        self.update_scoreboard(human,coord,11,"HP")

        # update unit healthpoints (ai player)
        coord = [[753,650],[925,650],[753,702],[925,702],[753,754],[925,754]]
        self.update_scoreboard(ai,coord,11,"HP")

        # update unit name
        coord = [[192,640],[366,640],[192,693],[366,693],[192,744],[366,744]]
        self.update_scoreboard(human,coord,15,"NAME")

        # update unit name
        coord = [[676,640],[850,640],[676,693],[850,693],[676,744],[850,744]]
        self.update_scoreboard(ai,coord,15,"NAME")

        # update unit profession
        coord = [[290,670],[463,670],[290,722],[463,722],[290,774],[463,774]]
        self.update_scoreboard(human,coord,15,"PROF")

        # update unit profession
        coord = [[774,670],[947,670],[774,722],[947,722],[774,774],[947,774]]
        self.update_scoreboard(ai,coord,15,"PROF")

        # update unit experience
        coord = [[265,669],[440,669],[265,720],[440,720],[265,774],[440,774]]
        self.update_scoreboard(human,coord,11,"EXP")

        # update unit experience
        coord = [[753,669],[925,669],[753,720],[925,720],[753,774],[925,774]]
        self.update_scoreboard(ai,coord,11,"EXP")

        # update unit rank
        coord = [[310,653],[485,653],[310,702],[485,702],[310,755],[485,755]]
        self.update_scoreboard(human,coord,11,"RANK")

        # update unit rank
        coord = [[803,653],[975,653],[803,702],[975,702],[803,755],[975,755]]
        self.update_scoreboard(ai,coord,11,"RANK")

# auto setup players
def setup_player(objplayer,avatar,xunits,prefixname):
     t_index = 0
     for n in range(xunits):
        # select randomly for a unit
        sel = random.randint(0,len(avatar)-1)

        # initialize select unit data
        init_index = n
        # initialize Human name
        init_name = prefixname + "%02d" % (random.randint(1,99),) 

        init_avatar = avatar[sel][0]
        init_icon = avatar[sel][1]
        init_prof = avatar[sel][2]
        init_atk = avatar[sel][3]
        init_dep = avatar[sel][4]
        init_posx = int(avatar[sel][5])
        init_posy = int(avatar[sel][6])
        init_avatarsize = avatar[sel][7]
       
        # instantiate list class for human player
        objplayer.append(Player(init_name, init_prof, init_index, "ALIVE", init_atk, init_dep, init_avatar, init_icon,init_posx,init_posy,init_avatarsize))



#----- MAIN PROGRAM -----#

pygame.init()

# initialize ATK and DEP/DEF
WARR_ATK = random.randint(WARR_MIN_ATK, WARR_MAX_ATK)
TANK_ATK = random.randint(TANK_MIN_ATK, TANK_MAX_ATK)
WARR_DEP = random.randint(WARR_MIN_DEP, WARR_MAX_DEP)
TANK_DEP = random.randint(TANK_MIN_DEP, TANK_MAX_DEP)

# configuration: avatar char, avatar head, prof, atk, dep/def, posx, posy,avatar size
avatar = [
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char1.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char1_head.png","WARRIOR",WARR_ATK,WARR_DEP,80,100,(395,559)],    
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char2.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char2_head.png","TANK",TANK_ATK,TANK_DEP,90,105,(349,505)],
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char3.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char3_head.png","WARRIOR",WARR_ATK,WARR_DEP,100,140,(252,449)],
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char4.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char4_head.png","TANK",TANK_ATK,TANK_DEP,100,150,(388,439)],
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char5.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char5_head.png","WARRIOR",WARR_ATK,WARR_DEP,100,120,(300,500)],
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char6.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char6_head.png","TANK",TANK_ATK,TANK_DEP,180,130,(178,484)],
    ["/Users/joeldizon/Documents/Project_BattleGame/assets/char7.png","/Users/joeldizon/Documents/Project_BattleGame/assets/char7_head.png","WARRIOR",WARR_ATK,WARR_DEP,110,130,(298,459)]]

# setup players automatically
# to be replaced with manual selection for human player
xunits = 6
setup_player(human,avatar,xunits,"HM")
setup_player(ai,avatar,xunits,"AI")

# check list class length
print("human: ", len(human))
print("ai: ", len(ai))
# test list class
print("human name: ", human[0].name)
print("ai name: ", ai[0].name)
# check list class items
print(human)
print(ai)

# setup UI
myUI = BattleUI()
myUI.setHeader()
myUI.setStage()
myUI.setScoreboard()
myUI.refresh_score_data(human,ai)
myUI.updatecharater(human,"human",cur_index_human)
myUI.updatecharater(ai,"ai",cur_index_ai)


# main loop to handle events
running = True
gameover = False
while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  
            if not gameover:         
                if event.type == pygame.KEYDOWN:
                    print (event.type)
                    if event.key == pygame.K_a:
                        print ("a is pressed")
                        myUI.setStage()
                        if cur_index_human == 0:
                            cur_index_human = len(human)-1
                        else:
                            cur_index_human -= 1
                        myUI.updatecharater(human,"human",cur_index_human)
                        myUI.updatecharater(ai,"ai",cur_index_ai)
                        print("Cur Index Human: ", cur_index_human)
                    if event.key == pygame.K_f:
                        print ("f is pressed")
                        myUI.setStage()
                        if cur_index_human == len(human)-1:
                            cur_index_human = 0
                        else:
                            cur_index_human += 1                            
                        myUI.updatecharater(human,"human",cur_index_human)
                        myUI.updatecharater(ai,"ai",cur_index_ai)
                        print("Cur Index Human: ", cur_index_human)

                    if event.key == pygame.K_h:
                        print ("h is pressed")
                        myUI.setStage()
                        if cur_index_ai == 0:
                            cur_index_ai = len(ai)-1
                        else:
                            cur_index_ai -= 1
                        myUI.updatecharater(ai,"ai",cur_index_ai)
                        myUI.updatecharater(human,"human",cur_index_human)
                        print("Cur Index AI: ", cur_index_ai)

                    if event.key == pygame.K_l:
                        print ("l is pressed")
                        myUI.setStage()
                        if cur_index_ai == len(ai)-1:
                            cur_index_ai = 0
                        else:
                            cur_index_ai += 1
                        myUI.updatecharater(ai,"ai",cur_index_ai)
                        myUI.updatecharater(human,"human",cur_index_human)
                        print("Cur Index AI: ", cur_index_ai)
                                
                    if event.key == pygame.K_x:
                        print ("x is pressed")
                        myUI.setStage()
                        
                        myUI.updatecharater(ai,"ai",cur_index_ai)
                        myUI.updatecharater(human,"human",cur_index_human)
                        print("Cur Index AI: ", cur_index_ai)

                        #running = False  
                        gameover = True
                                
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     # Set the x, y postions of the mouse click
            #     x, y = event.pos
            #     print (x,y)
            #     print (btnLeftHuman.get_rect())
            #     if btnLeftHuman.get_rect().collidepoint(x, y):
            #         print('clicked on hhimage')
  
        pygame.display.update()


pygame.quit()
