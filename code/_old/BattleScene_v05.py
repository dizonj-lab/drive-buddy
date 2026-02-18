import pygame
import random
import datetime
import os

# ------- OS Compatibility code here -----------
dirname = os.path.dirname(__file__)
# -------

# global variables
global game_round, human_coins, ai_coins, start_time, end_time, cur_index_ai, cur_index_human, game_over

game_round = 1
human_coins = 0
ai_coins = 0
start_time = datetime.datetime.now()
end_time = datetime.datetime.now()
cur_index_ai = 0
cur_index_human = 0
game_over = False

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

# set list for players
human = []
ai = []

class Player:
    def __init__(self, name, profession, playertype, index,state,atk,dep,avatar,icon,posx,posy,avatarsize):
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
        self.playertype = playertype
    
    def healme(self):
        print(self.name)
        print(self.healthpoint)
    
    def attack(self):
        print(self.name)
        print(self.healthpoint)        

    # heal action orchestration
    def heal(self):
        # calculate heal points
        heal_points = random.randint(HEAL_MIN,HEAL_MAX)

        # heal a unit by increasing healthpoint with additional heal_points
        self.healthpoint += heal_points
        
        # check if unit has exceeded its healthpoint  (100), if so reset to 100 
        if self.healthpoint > 100:
            self.healthpoint = 100
        
        # show results
        print("The unit ",t_unit.name, " is healed.")
        print(heal_points, " health points has been added!")
        
    # attack orchestration
    def attack(self,t_target):
        # calculate damage points. use the initialized atk and dep
        dmgpts = calculateDamagePoints(self.atk,self.dep)
        
        # update coins for human
        # updateCoins("human",dmgpts)
        # update coins for ai
        # updateCoins("ai",target.dep)

        # if self.playertype == "HM":
        #     human_coins += dmgpts
        #     ai_coins += t_target.dep
        # else:
        #     human_coins += dmgpts
        #     ai_coins += t_target.dep
    

        # update target's healthpoints
        updateHealthPoints(t_target,dmgpts)

        # check if target is still alive, update state
        updateUnitState(t_target)
    
        # update the attacker's experience
        updateExperience(self,dmgpts)

        # update the attacker extra experience
        updateExtraExperience(self,dmgpts)

        # update the target's experience
        updateExperience(self,t_target.dep)

        # update the target's extra experience
        updateExtraExperience(t_target,dmgpts)

        # promote attacker
        promoteUnit(self)

        # promote target
        promoteUnit(t_target)


class GUI:
    def __init__(self):

        self.window_width = 1185
        self.window_height = 800

        self.screen = pygame.display.set_mode((self.window_width,self.window_height))
        pygame.display.set_caption("Game of Champions")             
 
        self.header = dirname + "/assets/header.png"
        self.screen_header_size = (1185,124)
        self.screen_headerX = 0
        self.screen_headerY = 0

        self.stage = dirname + "/assets/stage.png"
        self.screen_stage_size = (1185,500)
        self.screen_stageX = 0
        self.screen_stageY = 125

        self.scoreboard = dirname + "/assets/scoreboard.png"
        self.screen_scoreboard_size = (1185,174)
        self.screen_scoreboardX = 0
        self.screen_scoreboardY = 626

        self.name = ""
        self.playerType = ["Human","Machine"]
        self.locX = 0
        self.locY = 0
        self.AvatarSize = (100,100)
        self.AvatarHuman = dirname + "/assets/Human_Head.png"
        self.AvatarAI = dirname + "/assets/Machine_Head.png"
        self.Avatar = ""
        self.coinLevel = 0
        self.SCREEN_FLIP_FACTOR = 600

        self.dead_mark = dirname + "/assets/dead.png"
        self.progressbar_gre = dirname + "/assets/progress_green.png"
        self.progressbar_red = dirname + "/assets/progress_red.png"

        # color RGP constants
        self.GOLD = (255, 215, 0)
        self.WHITE = (255,255,255)
        self.GREY = (51, 51, 38)
        self.YELLOW = (250, 253, 15)
        self.FONTSTYLE = "calibri"

    def setHeader(self):
        screen_header = pygame.image.load(self.header)
        screen_header = pygame.transform.scale(screen_header,self.screen_header_size)
        self.screen.blit(screen_header,(self.screen_headerX,self.screen_headerY))

    def setStage(self):
        screen_stage = pygame.image.load(self.stage)
        screen_stage = pygame.transform.scale(screen_stage,self.screen_stage_size)
        self.screen.blit(screen_stage,(self.screen_stageX,self.screen_stageY))
        
    def setScoreboard(self):
        screen_scoreboard = pygame.image.load(self.scoreboard)
        screen_scoreboard = pygame.transform.scale(screen_scoreboard,self.screen_scoreboard_size)
        self.screen.blit(screen_scoreboard,(self.screen_scoreboardX,self.screen_scoreboardY))

        self.locX = 50
        self.locY = 650
        self.Avatar = self.AvatarHuman
        avatar = pygame.image.load(self.Avatar)
        avatar = pygame.transform.scale(avatar,self.AvatarSize)
        self.screen.blit(avatar,(self.locX,self.locY))

        self.locX = 1010
        self.locY = 650
        self.Avatar = self.AvatarAI
        avatar = pygame.image.load(self.Avatar)
        avatar = pygame.transform.scale(avatar,self.AvatarSize)
        self.screen.blit(avatar,(self.locX,self.locY))

    def updateCharacter(self,t_unit):
        t_posx = t_unit.posx
        if (t_unit.playertype=="AI"):
            t_posx = t_unit.posx + self.SCREEN_FLIP_FACTOR
 
        t_playChar = pygame.image.load(t_unit.avatar)
        t_playChar = pygame.transform.scale(t_playChar,t_unit.avatarsize)
        self.screen.blit(t_playChar,(t_posx,t_unit.posy))
        
        self.showLabelOnStage("[ S ]/[ F ] Select a Human unit  [ J ]/[ L ] Select an AI unit [ A ] Attack an opponent [ H ] Heal a unit [ P ] Pass a move",(300,610),self.FONTSTYLE,12,self.WHITE)
        self.showLabelOnStage("Round",(576,175),self.FONTSTYLE,15,self.GREY)
        self.showLabelOnStage(str(game_round),(585,195),self.FONTSTYLE,40,self.GREY)
        if game_round % 2 > 0:
            self.showLabelOnStage("Human's turn to attack!",(490,570),self.FONTSTYLE,20,self.WHITE)
        else:
            self.showLabelOnStage("AI's turn to attack!",(510,570),self.FONTSTYLE,20,self.WHITE)
       
    def update_uniticons(self,t_player,t_coord):
        i=0
        for p in t_player:
            t_icon = p.icon
            icon = pygame.image.load(t_icon)
            avatar = pygame.transform.scale(icon,(100,200))
            self.screen.blit(icon,(t_coord[i][0],t_coord[i][1]))
            i += 1

    def update_deadmark(self,t_player,t_coord):
        i=0
        for p in t_player:
            print("Name: ", p.name, " State: ",p.state)
            if p.state == "DEAD":
                t_icon = self.dead_mark
                icon = pygame.image.load(t_icon)
                avatar = pygame.transform.scale(icon,(70,70))
                self.screen.blit(icon,(t_coord[i][0],t_coord[i][1]))
            i += 1

    def update_scoreboard(self,t_player,t_coord,t_fsize,widget):
        # update unit healthpoints 
        t_font = self.FONTSTYLE   
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
                t_fontcolor = self.GOLD
            elif widget == "PROF":
                t_value = p.profession
                t_font = None

            font = pygame.font.SysFont(t_font, t_fsize)
            img = font.render(str(t_value), True,t_fontcolor)
            self.screen.blit(img, (t_coord[i][0], t_coord[i][1]))       
            i += 1

    def refresh_score_data(self,human,ai):
        # update unit head icons (human player)
        t_coord = [[190,648],[365,648],[190,698],[365,698],[190,752],[365,752]]
        self.update_uniticons(human,t_coord)

        # update unit head icons (ai player)
        t_coord = [[674,648],[849,648],[674,698],[849,698],[674,752],[849,752]]
        self.update_uniticons(ai,t_coord)

        # update dead mark for human player
        t_coord = [[180,640],[350,640],[180,690],[350,690],[180,743],[350,743]]
        self.update_deadmark(human,t_coord)

        # update dead mark for ai player
        t_coord = [[657,640],[835,640],[657,690],[835,690],[657,743],[835,743]]
        self.update_deadmark(ai,t_coord)

        # update human coins
        t_coord = (115,770)
        font = pygame.font.SysFont(None, 25)
        img = font.render(str(human_coins), True, self.GOLD)
        self.screen.blit(img, t_coord)       

        # update ai coins
        t_coord = (1050,770)
        font = pygame.font.SysFont(None, 25)
        img = font.render(str(ai_coins), True, self.GOLD)
        self.screen.blit(img, t_coord)       
      
        # update unit healthpoints (human player)
        t_coord = [[265,650],[440,650],[265,702],[440,702],[265,754],[440,754]]
        self.update_scoreboard(human,t_coord,11,"HP")

        # update unit healthpoints (ai player)
        t_coord = [[753,650],[925,650],[753,702],[925,702],[753,754],[925,754]]
        self.update_scoreboard(ai,t_coord,11,"HP")

        # update unit name
        t_coord = [[192,640],[366,640],[192,693],[366,693],[192,744],[366,744]]
        self.update_scoreboard(human,t_coord,15,"NAME")

        # update unit name
        t_coord = [[676,640],[850,640],[676,693],[850,693],[676,744],[850,744]]
        self.update_scoreboard(ai,t_coord,15,"NAME")

        # update unit profession
        t_coord = [[290,670],[463,670],[290,722],[463,722],[290,774],[463,774]]
        self.update_scoreboard(human,t_coord,15,"PROF")

        # update unit profession
        t_coord = [[774,670],[947,670],[774,722],[947,722],[774,774],[947,774]]
        self.update_scoreboard(ai,t_coord,15,"PROF")

        # update unit experience
        t_coord = [[265,669],[440,669],[265,720],[440,720],[265,774],[440,774]]
        self.update_scoreboard(human,t_coord,11,"EXP")

        # update unit experience
        t_coord = [[753,669],[925,669],[753,720],[925,720],[753,774],[925,774]]
        self.update_scoreboard(ai,t_coord,11,"EXP")

        # update unit rank
        t_coord = [[310,653],[485,653],[310,702],[485,702],[310,755],[485,755]]
        self.update_scoreboard(human,t_coord,11,"RANK")

        # update unit rank
        t_coord = [[803,653],[975,653],[803,702],[975,702],[803,755],[975,755]]
        self.update_scoreboard(ai,t_coord,11,"RANK")

    def show_progressbar(self,t_progress,t_pbar,t_loc):
        # get the 1% width of the pbar icon
        # get the width of the pbar depending on the current healthpoint
        p = 95/100 * t_progress
        # check if p value is less than zero. less than width for pbar is not accepted.
        if p < 0:
            p = 0
        # show pbar icon on stage
        icon = pygame.image.load(t_pbar)
        icon = pygame.transform.scale(icon,(p,10))
        self.screen.blit(icon,t_loc)

    # write text anywhere on stage
    def showLabelOnStage(self,t_msg,t_loc,t_font,t_size,t_color):
        font = pygame.font.SysFont(t_font, t_size)
        img = font.render(t_msg, True, t_color)
        self.screen.blit(img, t_loc) 

    def showUnitStatistics(self,t_unit):  
        # put current unit name in stage
        if t_unit.playertype == "HM":
            t_loc = (80,162)
            t_hploc = (60,188)
            t_profloc = (60,250)
            t_exploc = (60,200)
        else:
            t_loc = (1070,162)
            t_hploc = (1053,188)
            t_profloc = (1053,250)
            t_exploc = (60,200)

        # show name on stage
        font = pygame.font.SysFont(self.FONTSTYLE, 15)
        img = font.render(t_unit.name, True, self.WHITE)
        self.screen.blit(img, t_loc) 
        
        # show healthpoints progress bar on stage
        if t_unit.healthpoint > 40:
            t_pbar = self.progressbar_gre
        else:
            t_pbar = self.progressbar_red

        self.show_progressbar(t_unit.healthpoint,t_pbar,t_hploc)

        # show unit's profession on stage
        self.showLabelOnStage(t_unit.profession,t_profloc,self.FONTSTYLE,10,(0,0,0))

# ALL FUNCTIONS

# auto setup players
def setupPlayer(t_player,avatar,xunits,prefixname):
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
        init_state = "ALIVE" # random.choice(["ALIVE","DEAD"]) # for testing
        init_playertype = prefixname
       
        # instantiate list class for human player
        t_player.append(Player(init_name, init_prof, init_playertype, init_index, init_state, init_atk, init_dep, init_avatar, init_icon,init_posx,init_posy,init_avatarsize))

# calculate the damage points. reset the damage points if less than zero
def calculateDamagePoints(atp,dfp):
    result = atp - dfp + random.randint(EXTRA_MIN_DMG, EXTRA_MAX_DMG)
    # convert damage points to positive (to make it more intuitive)
    result = int(result * -1)
    # reset result to 0 if dmg points is negative
    if result < 0:
        result = 0
    return result

# update unit's state to Dead from Alive
def updateUnitState(t_unit):
    if t_unit.healthpoint <= 0:
        # promote
        t_unit.state = "DEAD"

# update unit's rank and ressetting experience to zero
def promoteUnit(t_unit):
    if t_unit.experience >= MAX_EXP:
        # promote
        t_unit.rank += 1
        # reset experience
        t_unit.experience = 0 

# update unit's experience
def updateExperience(t_unit,exp):
    # calculate experience
    t_unit.experience += exp

# update unit's experience by adding extra experience
def updateExtraExperience(t_unit,dmgpts):
    # calculate target's experence
    if dmgpts > HIGH_DMG:
        # increase experience by 20% if damage points is more than 10
        t_unit.experience = int(t_unit.experience * LOW_EXP)
    
    if dmgpts <= LOW_DMG:
        # increase experience by 50% if damage points is below 10
        t_unit.experience = int(t_unit.experience * HIGH_EXP)

# update the unit's health points
def updateHealthPoints(t_unit,dmgpts):
    t_unit.healthpoint = int(t_unit.healthpoint - dmgpts)


# randomly select any alive unit
def selectAliveUnit(t_player):
    selected = False
    while selected == False:
        player_index = random.randint(0,len(t_player)-1)
        if t_player[player_index].state == "ALIVE":
            result = player_index
            selected = True
        else:
            selected = False
    return result

# auto select unit to heal. select unit with the 'least' healtpoints
def selectUnitToHeal(t_player):
    alive_units = []

    # get all alive units
    for t_unit in t_player:
        if t_unit.state == "ALIVE":
            alive_units.append([t_unit.index,t_unit.healthpoint])

    # sort list by its healthpoints in ascending. the one on top will be the unit with the least hp which require immediate healing
    alive_units = sorted(alive_units,key=lambda x: x[1])
    
    # get the index of unit to heal
    result = alive_units[0][0] - 1
    
    return result



# check if player is still alive
def isPlayerAlive(t_player):
    result = False
    alive = 0
    for t_unit in t_player:
        if t_unit.state == "ALIVE":
            alive += 1
    if alive > 0:
        result = True   
    return result

# check if unit is still alive
def isUnitAlive(t_unit):
    result = False
    if t_unit.state == "ALIVE":
        result = True
    return result

# implement coin updates for both players
def updateCoins(t_playertype, t_coins):
    global human_coins, ai_coins

    if t_playertype == "human":
        human_coins += t_coins
    if t_playertype == "ai":
        ai_coins += t_coins



# write game status on console
def showResults(t_human, t_ai):
    print("Player Group: HUMANS") 
    print("Total Coins :", human_coins) 
    for t_unit in t_human:
        print("Index: ", t_unit.index, " | State: ", t_unit.state, " | Name: ", t_unit.name, "| Prof: ", t_unit.profession, " | HP: ", t_unit.healthpoint, " | EXP: ", t_unit.experience, " | ATK: ", t_unit.atk, " | DEF: ", t_unit.dep, " | Rank: ", t_unit.rank)

    print("Player Group: ROBOTS") 
    print("Total Coins :", ai_coins) 
    for t_unit in t_ai:
        print("Index: ", t_unit.index, " | State: ", t_unit.state, " | Name: ", t_unit.name, "| Prof: ", t_unit.profession, " | HP: ", t_unit.healthpoint, " | EXP: ", t_unit.experience, " | ATK: ", t_unit.atk, " | DEF: ", t_unit.dep, " | Rank: ", t_unit.rank)


#----- MAIN PROGRAM -----#


pygame.init()

# initialize ATK and DEP/DEF
warr_atk = random.randint(WARR_MIN_ATK, WARR_MAX_ATK)
tank_atk = random.randint(TANK_MIN_ATK, TANK_MAX_ATK)
warr_dep = random.randint(WARR_MIN_DEP, WARR_MAX_DEP)
tank_dep = random.randint(TANK_MIN_DEP, TANK_MAX_DEP)

# configuration: avatar char, avatar head, prof, atk, dep/def, posx, posy,avatar size
avatar = [
    [dirname + "/assets/char1.png",dirname + "/assets/char1_head.png","WARRIOR",warr_atk,warr_dep,80,100,(395,559)],    
    [dirname + "/assets/char2.png",dirname + "/assets/char2_head.png","TANK",tank_atk,tank_dep,90,105,(349,505)],
    [dirname + "/assets/char3.png",dirname + "/assets/char3_head.png","WARRIOR",warr_atk,warr_dep,100,140,(252,449)],
    [dirname + "/assets/char4.png",dirname + "/assets/char4_head.png","TANK",tank_atk,tank_dep,100,150,(388,439)],
    [dirname + "/assets/char5.png",dirname + "/assets/char5_head.png","WARRIOR",warr_atk,warr_dep,100,120,(300,500)],
    [dirname + "/assets/char6.png",dirname + "/assets/char6_head.png","TANK",tank_atk,tank_dep,180,130,(178,484)],
    [dirname + "/assets/char7.png",dirname + "/assets/char7_head.png","WARRIOR",warr_atk,warr_dep,110,130,(298,459)]]

# setup players automatically
# to be replaced with manual selection for human player
xunits = 6
setupPlayer(human,avatar,xunits,"HM")
setupPlayer(ai,avatar,xunits,"AI")

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
myUI = GUI()
myUI.setHeader()
myUI.setStage()
myUI.setScoreboard()
myUI.refresh_score_data(human,ai)
myUI.updateCharacter(human[cur_index_human])
myUI.updateCharacter(ai[cur_index_ai])
myUI.showUnitStatistics(human[cur_index_human])
myUI.showUnitStatistics(ai[cur_index_ai])

showResults(human,ai)

# main loop to handle events
game_over = False
game_round = 1
running = True
while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 

            if not game_over:          
                if event.type == pygame.KEYDOWN:
                    human_unit = human[cur_index_human]
                    ai_unit = ai[cur_index_ai]

                    if event.key == pygame.K_s:
                        print ("s is pressed")
                        myUI.setStage()
                        if cur_index_human == 0:
                            cur_index_human = len(human)-1
                        else:
                            cur_index_human -= 1
                        myUI.updateCharacter(human_unit)
                        myUI.updateCharacter(ai_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)

                        print("Cur Index Human: ", cur_index_human)
                    if event.key == pygame.K_f:
                        print ("f is pressed")
                        myUI.setStage()
                        if cur_index_human == len(human)-1:
                            cur_index_human = 0
                        else:
                            cur_index_human += 1                            
                        myUI.updateCharacter(human_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        print("Cur Index Human: ", cur_index_human)

                    if event.key == pygame.K_j:
                        print ("h is pressed")
                        myUI.setStage()
                        if cur_index_ai == 0:
                            cur_index_ai = len(ai)-1
                        else:
                            cur_index_ai -= 1
                        myUI.updateCharacter(ai_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        print("Cur Index AI: ", cur_index_ai)

                    if event.key == pygame.K_l:
                        print ("l is pressed")
                        myUI.setStage()
                        if cur_index_ai == len(ai)-1:
                            cur_index_ai = 0
                        else:
                            cur_index_ai += 1
                        myUI.updateCharacter(ai_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        print("Cur Index AI: ", cur_index_ai)
                        
                    if event.key == pygame.K_a:
                        print ("a is pressed. attacking")
                        isInvalidAttack = False
                        
                        if human_unit.state == "DEAD" or ai_unit.state == "DEAD":
                            isInvalidAttack = True

                        if game_round % 2 > 0:                            
                            if not isInvalidAttack:
                               human_unit.attack(ai_unit)
                        else:
                            if not isInvalidAttack:
                               ai_unit.attack(human_unit)

                        if isInvalidAttack:
                            myUI.setStage()
                            #myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)

                            #myUI.refresh_score_data(human,ai)
                            myUI.showLabelOnStage("Invalid attack! Both attacker and target must be alive.",(420,300),myUI.FONTSTYLE,15,myUI.YELLOW)

                        else:
                            game_round +=1
                            # update coins for human
                            # updateCoins("human",dmgpts)
                            # update coins for ai
                            # updateCoins("ai",target.dep)

                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            
                            myUI.refresh_score_data(human,ai)

                        # print("Round", game_round)
                        # print("Attacker: ",attacker.name)
                        # print("Target: ",target.name)

                        # print("Cur Index AI: ", cur_index_ai)

                    if event.key == pygame.K_h:
                        print ("h is pressed. healing")
                        isInvalidHealAction = False

                        if human_unit.state == "DEAD" or ai_unit.state == "DEAD":
                            isInvalidHealAction = True

                        if game_round % 2 > 0:                            
                            if not isInvalidAttack:
                               human_unit.heal()
                        else:
                            if not isInvalidAttack:
                               ai_unit.heal()
                        
                        game_round +=1

                        myUI.setStage()
                        myUI.setScoreboard()
                        myUI.updateCharacter(ai_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        myUI.refresh_score_data(human,ai)
                        # print("healing ", attacker.name)
                        # print("Cur Index AI: ", cur_index_ai)
    
                    if event.key == pygame.K_p:
                        myUI.setStage()
    
                        if game_round % 2 > 0:
                            myUI.showLabelOnStage("Human player passed the previous round! ",(450,300),myUI.FONTSTYLE,15,myUI.YELLOW)

                        else:
                             myUI.showLabelOnStage("AI player passed the previous round!",(470,300),myUI.FONTSTYLE,15,myUI.YELLOW)
                        game_round +=1
                        myUI.setScoreboard()
                        myUI.updateCharacter(ai_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        myUI.refresh_score_data(human,ai)

                        print("Cur Index AI: ", cur_index_ai)
                        print ("p is pressed. pass")
            
            # check if HUMAN wins
            if isPlayerAlive(human) and not isPlayerAlive(ai):
                end_time = datetime.datetime.now()
                myUI.showLabelOnStage("Human player wins!",(470,300),myUI.FONTSTYLE,15,myUI.YELLOW)
                game_over = True

                showResults(human,ai)
                repeat = False

            # check if HUMAN wins
            if not isPlayerAlive(human) and isPlayerAlive(ai):
                end_time = datetime.datetime.now()
                myUI.showLabelOnStage("AI player wins!",(470,300),myUI.FONTSTYLE,15,myUI.YELLOW)
                game_over = True

                showResults(human,ai)
                repeat = False
        
            pygame.display.update()

pygame.quit()
