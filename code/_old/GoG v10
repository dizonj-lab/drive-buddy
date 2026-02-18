import pygame
import random
import datetime
import os
import time
import logging

# ------- OS Compatibility code here -----------
dirname = os.path.dirname(__file__)
# ----------------------------------------------

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
APP_VER = "0.10"
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
HEAL_MIN = 2
HEAL_MAX = 5
MAX_UNITS = 6

# default lists
# default unit's character on screen
selected_hmunits_coord = [[(10,660),(30,765),(30,750)],
                         [(95,660),(115,765),(115,750)],
                         [(180,660),(200,765),(200,750)],
                        [(265,660),(285,765),(285,750)],
                        [(350,660),(370,765),(370,750)],
                        [(435,660),(455,765),(455,750)]]

selected_aiunits_coord = [[(1075,660),(1095,765),(1095,750)],
                        [(990,660),(1010,765),(1010,750)],
                        [(905,660),(925,765),(925,750)],
                        [(820,660),(840,765),(840,750)],
                        [(735,660),(755,765),(755,750)],
                        [(650,660),(670,765),(670,750)]]


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
        print("The unit ",self.name, " is healed.")
        print(heal_points, " health points has been added!")
        
    # attack orchestration
    def attack(self,t_target):
        # calculate damage points. use the initialized atk and dep
        dmgpts = calculateDamagePoints(self.atk,self.dep)

        # check if attacker is an ai
        if "AI" in self.name:         
            # update coins for ai
            updateCoins("ai",dmgpts)
        else:
            # update coins for human (target)
            updateCoins("human",t_target.dep)

        # check if attacker is a human
        if "HM" in self.name:         
            # update coins for human
            updateCoins("human",dmgpts)
        else:
            # update coins for ai (target)
            updateCoins("ai",t_target.dep)

    
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

        # game setup attributes
        self.gamesetup_dashboard = dirname + "/assets/game_setup.png"
        self.gamesetup_dashboard_size = (1185,800)
        self.gamesetup_dashboardX = 0
        self.gamesetup_dashboardY = 0

        # color RGP constants
        self.GOLD = (255, 215, 0)
        self.WHITE = (255,255,255)
        self.GREY = (51, 51, 38)
        self.YELLOW = (250, 253, 15)
        self.FONTSTYLE = "calibri"

    def setGameSetupDashboard(self):
        screen_header = pygame.image.load(self.gamesetup_dashboard)
        screen_header = pygame.transform.scale(screen_header,self.gamesetup_dashboard_size)
        self.screen.blit(screen_header,(self.gamesetup_dashboardX,self.gamesetup_dashboardY))
 
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
        
        #self.showLabelOnStage("[ S ]/[ F ] Select a Human unit  [ J ]/[ L ] Select an AI unit [ A ] Attack an opponent [ H ] Heal a unit [ P ] Pass a move",(300,610),self.FONTSTYLE,12,self.WHITE)
        #self.showLabelOnStage("Round",(576,180),self.FONTSTYLE,13,self.GREY)
        self.writeCenterScreen(" Round", 576, 185, self.GREY,self.FONTSTYLE,13)
        self.writeCenterScreen(" " + str(game_round), 585, 215, self.GREY,self.FONTSTYLE,30)
        #self.showLabelOnStage(str(game_round),(585,195),self.FONTSTYLE,40,self.GREY)
        if game_round % 2 > 0:
            self.writeCenterScreen(" Human's turn to attack!", 490, 450, self.WHITE,self.FONTSTYLE,20)

        else:
            self.writeCenterScreen(" AI's turn to attack!", 490, 450, self.WHITE,self.FONTSTYLE,20)
      
    def showSelectedUnit(self, t_icon, t_prof, t_name, t_coord, t_iconsize):
        icon = pygame.image.load(t_icon)
        avatar = pygame.transform.scale(icon,t_iconsize)
        self.screen.blit(icon,t_coord)
    
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

    def show_progressbar(self,t_progress,t_pbar,t_loc,t_maxval):
        # get the 1% width of the pbar icon
        # get the width of the pbar depending on the current healthpoint
        p = 90/t_maxval * t_progress
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
        
    def writeCenterScreen(self,t_msg, t_locx, t_locy, t_color,t_font,t_size):
        font = pygame.font.SysFont(t_font, t_size)
        text = font.render(t_msg, 1, pygame.Color(t_color))
        text_rect = text.get_rect(center=(self.window_width//2, t_locy))
        self.screen.blit(text, text_rect)

    def showUnitStatistics(self,t_unit):  
        # put current unit name in stage
        if t_unit.playertype == "HM":
            t_loc = (80,162)
            t_hploc = (60,188)
            t_profloc = (60,250)
            t_exploc = (60,210)
            t_rank = (60,232)
            t_dep = (60,273)
            t_atk = (60,294)
        else:
            t_loc = (1070,162)
            t_hploc = (1053,188)
            t_profloc = (1053,250)
            t_exploc = (1053,210)
            t_rank = (1053,232)
            t_dep = (1053,273)
            t_atk = (1053,294)

        # show name on stage
        font = pygame.font.SysFont(self.FONTSTYLE, 15)
        img = font.render(t_unit.name, True, self.WHITE)
        self.screen.blit(img, t_loc) 
        
        # show healthpoints progress bar on stage
        if t_unit.healthpoint > 40:
            t_pbar = self.progressbar_gre
        else:
            t_pbar = self.progressbar_red

        self.show_progressbar(t_unit.healthpoint,t_pbar,t_hploc,95)
        self.show_progressbar(t_unit.experience,self.progressbar_gre,t_exploc,95)
        self.show_progressbar(t_unit.rank,self.progressbar_gre,t_rank,20)
        self.show_progressbar(t_unit.dep,self.progressbar_gre,t_dep,15)
        self.show_progressbar(t_unit.atk,self.progressbar_gre,t_atk,20)
        
        # show unit's profession on stage
        self.showLabelOnStage(t_unit.profession,t_profloc,self.FONTSTYLE,10,(0,0,0))

# COMMON FUNCTIONS

# auto setup players
def setupPlayer(t_player,avatar,prefixname):
    t_msg = "Setting up units for " + prefixname + " player."
    log_obj.info(t_msg)

    for sel in range(len(avatar)):

        # initialize select unit data
        init_index = sel
        # initialize Human name
        init_name = avatar[sel][8] #prefixname + "%02d" % (random.randint(1,99),) 

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
        
        # instantiate list class for player
        t_player.append(Player(init_name, init_prof, init_playertype, init_index, init_state, init_atk, init_dep, init_avatar, init_icon,init_posx,init_posy,init_avatarsize))
        t_msg = "Setting up unit " + init_name + " player."
        log_obj.info(t_msg)

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
        t_unit.healthpoint = 0

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
            alive_units.append([t_unit.index,t_unit.name,t_unit.healthpoint])

    # sort list by its healthpoints in ascending. the one on top will be the unit with the least hp which require immediate healing
    alive_units = sorted(alive_units,key=lambda x: x[2])
    print(alive_units)
    
    # get the index of unit to heal
    result = alive_units[0][0] #- 1
    print("heal result: ",result)

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

    t_msg = "Player Group: HUMANS"
    log_obj.info(t_msg)
    print(t_msg) 

    t_msg = "Total Coins :" + str(human_coins)
    log_obj.info(t_msg)
    print(t_msg) 

    for t_unit in t_human:
        t_msg = "Index: " + str(t_unit.index) + " | State: " + t_unit.state + " | Name: " + t_unit.name + "| Prof: " + t_unit.profession + " | HP: " + str(t_unit.healthpoint) + " | EXP: " + str(t_unit.experience) + " | ATK: " + str(t_unit.atk) + " | DEF: " + str(t_unit.dep) + " | Rank: " + str(t_unit.rank)
        log_obj.info(t_msg)
        print(t_msg) 

    t_msg = "Player Group: AIs"
    log_obj.info(t_msg)
    print(t_msg) 

    t_msg = "Total Coins :" + str(ai_coins)
    log_obj.info(t_msg)
    print(t_msg) 

    for t_unit in t_ai:
        t_msg = "Index: " + str(t_unit.index) + " | State: " + t_unit.state + " | Name: " + t_unit.name + "| Prof: " + t_unit.profession + " | HP: " + str(t_unit.healthpoint) + " | EXP: " + str(t_unit.experience) + " | ATK: " + str(t_unit.atk) + " | DEF: " + str(t_unit.dep) + " | Rank: " + str(t_unit.rank)
        log_obj.info(t_msg)
        print(t_msg) 


# AI auto move. randomly selects action/move then execute
def ai_move(t_UI,human,ai,ai_unit,human_unit):
    global cur_index_ai, cur_index_human
    
    # attack move
    actionType = ["A","H","P"]

    # check if ai is still alive (subsequent to human attack)
    # solution to the "hanging" problem once human final attack and eliminated ai
    if isPlayerAlive(ai):
        aiAction = random.choice(actionType)
    else:
        # do nothing
        aiAction = "X"

    if aiAction == "A":
        # ai's turn
        # random select attacker and target
        rand_alivehuman_index = selectAliveUnit(human)
        rand_aliveai_index = selectAliveUnit(ai)
        human_unit = human[rand_alivehuman_index]
        ai_unit = ai[rand_aliveai_index]
        print("Attacker: ",ai_unit.name)
        print("Target: ", human_unit.name)
        print("Target Befor HP: ", human_unit.healthpoint)
        print("Target HP: ", human_unit.healthpoint)
        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)

        t_UI.refresh_score_data(human,ai)
        t_UI.writeCenterScreen(" AI is selecting an attacker and a target...", 400, 400, t_UI.WHITE,t_UI.FONTSTYLE,20)

        pygame.display.update()
        time.sleep(3)
        
        # reposition the current character
        #cur_index_ai = rand_aliveai_index
        #cur_index_human = rand_alivehuman_index

        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)                         
        pygame.display.update()

        t_UI.writeCenterScreen(" AI is attacking target...", 500, 400, t_UI.WHITE,t_UI.FONTSTYLE,20)
        pygame.display.update()

        ai_unit.attack(human_unit)   
        #pygame.time.wait
        time.sleep(1)

        # update screen after AI attack
        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)
        pygame.display.update()
   
    elif aiAction == "H":
        # random select attacker and target
        aiUnit_index = selectUnitToHeal(ai)
        ai_unit = ai[aiUnit_index]
        print("Healing: ",ai_unit.name)
        print("Before HP: ",ai_unit.healthpoint)
        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)

        t_UI.refresh_score_data(human,ai)
        t_UI.writeCenterScreen(" AI is selecting a unit to heal...", 400, 400, t_UI.WHITE,t_UI.FONTSTYLE,20)

        pygame.display.update()
        time.sleep(3)

        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)                         
        pygame.display.update()

        t_UI.writeCenterScreen(" AI is healing its unit...", 500, 400, t_UI.WHITE,t_UI.FONTSTYLE,20)
        pygame.display.update()
        time.sleep(1)

        ai_unit.heal()   
        print("After HP: ",ai_unit.healthpoint)

        # update screen after AI attack
        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)
        pygame.display.update()

    elif aiAction == "P":
        t_UI.writeCenterScreen(" AI player passed a round!", 470, 300, myUI.WHITE,myUI.FONTSTYLE,20)

        pygame.display.update()
        time.sleep(1)

        # update screen 
        t_UI.setStage()
        t_UI.setScoreboard()
        t_UI.updateCharacter(ai_unit)
        t_UI.updateCharacter(human_unit)
        t_UI.showUnitStatistics(ai_unit)
        t_UI.showUnitStatistics(human_unit)                         
        pygame.display.update()
    else:
        print("AI loses")

# common function to write to a log file
def logger(): 
        now = datetime.datetime.now()
        t_filename = dirname + "/logs/" + now.strftime("%Y%m%d") + ".log"

        log_obj = logging.getLogger(t_filename)
        log_obj.setLevel(logging.INFO)

        formatter = logging.Formatter('%(levelname)s%(asctime)s | %(message)s')

        file_handler = logging.FileHandler(t_filename)
        file_handler.setFormatter(formatter)

        log_obj.addHandler(file_handler) 
        return log_obj

#----- MAIN PROGRAM -----#

# initialize log file
log_obj = logger()
t_msg = "Game of Champion " + APP_VER + " has started."
log_obj.info(t_msg)

t_msg = "Initializing Pygame object."
log_obj.info(t_msg)

# initialize pygame object
pygame.init()
myUI = GUI()

# set game setup screen
myUI.setGameSetupDashboard()

# initiate selected unit's lists
sel_HMunits = []
sel_AIunits = []

# initialize ATK and DEP/DEF
warr_atk = random.randint(WARR_MIN_ATK, WARR_MAX_ATK)
tank_atk = random.randint(TANK_MIN_ATK, TANK_MAX_ATK)
warr_dep = random.randint(WARR_MIN_DEP, WARR_MAX_DEP)
tank_dep = random.randint(TANK_MIN_DEP, TANK_MAX_DEP)

# configuration: avatar char, avatar head, prof, atk, dep/def, posx, posy,avatar size
avatar = [
    [dirname + "/assets/char1.png",dirname + "/assets/char1_head.png","WARRIOR",warr_atk,warr_dep,80,100,(395,559),dirname + "/assets/char1_profile.png",(90,80),(20,660)],  
    [dirname + "/assets/char2.png",dirname + "/assets/char2_head.png","TANK",tank_atk,tank_dep,90,105,(349,505),dirname + "/assets/char2_profile.png",(90,80),(200,600)],
    [dirname + "/assets/char3.png",dirname + "/assets/char3_head.png","WARRIOR",warr_atk,warr_dep,100,140,(252,449),dirname + "/assets/char3_profile.png",(90,80),(200,600)],
    [dirname + "/assets/char4.png",dirname + "/assets/char4_head.png","TANK",tank_atk,tank_dep,150,150,(388,439),dirname + "/assets/char4_profile.png",(90,80),(200,600)],
    [dirname + "/assets/char5.png",dirname + "/assets/char5_head.png","WARRIOR",warr_atk,warr_dep,100,120,(300,500),dirname + "/assets/char5_profile.png",(90,80),(200,600)],
    [dirname + "/assets/char6.png",dirname + "/assets/char6_head.png","TANK",tank_atk,tank_dep,180,130,(178,484),dirname + "/assets/char6_profile.png",(90,80),(200,600)],
    [dirname + "/assets/char7.png",dirname + "/assets/char7_head.png","WARRIOR",warr_atk,warr_dep,110,130,(298,459),dirname + "/assets/char7_profile.png",(90,80),(200,600)]]


t_msg = "Loading Game Setup module."
log_obj.info(t_msg)

# game setup
sel_ctr = 0
running = True
# game setup main loop
while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 

            if not game_over:          
                if event.type == pygame.KEYDOWN:                   
                    # for START
                    if event.key == pygame.K_s:
                        t_msg = "User has pressed S - Start."
                        log_obj.info(t_msg)

                        if len(sel_AIunits)!=0 and len(sel_HMunits)!=0:
                            running = False
                        
                    # for DONE
                    # D = Done, will finalized Human selections and will trigger randomized selections for AI
                    if event.key == pygame.K_c:
                        t_msg = "User has press C - Complete, to complete the human units selection."
                        log_obj.info(t_msg)

                        # only if AI units is zero (has not selected yet)
                        if len(sel_AIunits)==0:
                            print ("d is pressed")
                            # randomized selection based on the number human units count
                            sel_ctr = 0
                            i = 0
                            for t_unit in sel_HMunits:
                                i = random.randint(0,len(sel_HMunits)-1)

                                t_name = "AI" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_aiunits_coord[sel_ctr][0]
                                t_namecoord = selected_aiunits_coord[sel_ctr][1]
                                t_profcoord = selected_aiunits_coord[sel_ctr][2]

                                # show the unit on stage
                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_AIunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected AI Units: ",sel_AIunits)
                                # increase the selection counter
                                sel_ctr += 1
                                #time.sleep(0.5)
                        else:
                            print("AI units are already set!")
                        

                    # for RESET
                    # R = Reset, reset the selection of units. Reinitilized selection list data for both players
                    if event.key == pygame.K_r:
                        print ("r is pressed")
                        sel_ctr = 0
                        sel_HMunits = []
                        sel_AIunits = []
                        myUI.setGameSetupDashboard()
                        t_msg = "User has pressed R - Reset to reset the selection."
                        log_obj.info(t_msg)

                    if event.key == pygame.K_1:
                        print ("1 is pressed")
                        #showSelectedUnit(self, t_icon, t_prof, t_name, t_coord, t_iconsize):
                        if len(sel_AIunits)==0:
                            if sel_ctr < MAX_UNITS: 
                                i = 0
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)
                                                     
                    if event.key == pygame.K_2:
                        print ("2 is pressed. ")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 1
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)

                    if event.key == pygame.K_3:
                        print ("3 is pressed. ")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 2
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)

                    if event.key == pygame.K_4:
                        print ("4 is pressed. ")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 3
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)

                    if event.key == pygame.K_5:
                        print ("5 is pressed. healing")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 4
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                 myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)


                    if event.key == pygame.K_6:
                        print ("6 is pressed. healing")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 5
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)
                        
                    if event.key == pygame.K_7:
                        print ("7 is pressed. ")
                        if len(sel_AIunits)==0:       
                            if sel_ctr < MAX_UNITS: 
                                i = 6
                                t_name = "HM" + "%02d" % (random.randint(1,99),) 
                                t_avatar = avatar[i][0]
                                t_icon = avatar[i][1]
                                t_prof = avatar[i][2]
                                t_atk = avatar[i][3]
                                t_dep = avatar[i][4]
                                t_locx = avatar[i][5]
                                t_locy = avatar[i][6]
                                t_avatarsize = avatar[i][7]
                                t_avatarprof = avatar[i][8]
                                t_iconsize = avatar[i][9]
                                t_coord = selected_hmunits_coord[sel_ctr][0]
                                t_namecoord = selected_hmunits_coord[sel_ctr][1]
                                t_profcoord = selected_hmunits_coord[sel_ctr][2]

                                myUI.showSelectedUnit(t_avatarprof,t_prof,t_name,t_coord,t_iconsize)
                                myUI.showLabelOnStage(t_name,t_namecoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                myUI.showLabelOnStage(t_prof,t_profcoord,myUI.FONTSTYLE,12,myUI.WHITE)
                                sel_HMunits.append([t_avatar,t_icon,t_prof,t_atk,t_dep,t_locx,t_locy,t_avatarsize,t_name])
                                print("selected HM Units: ",sel_HMunits)
                                sel_ctr += 1
                            else:
                                myUI.writeCenterScreen(" Maximum unit selection has been reached.", 500, 540, myUI.YELLOW,myUI.FONTSTYLE,14)
            pygame.display.update()

setupPlayer(human,sel_HMunits,"HM")
setupPlayer(ai,sel_AIunits,"AI")

# setup UI
myUI.setHeader()
myUI.setStage()
myUI.setScoreboard()
myUI.refresh_score_data(human,ai)
myUI.updateCharacter(human[cur_index_human])
myUI.updateCharacter(ai[cur_index_ai])
myUI.showUnitStatistics(human[cur_index_human])
myUI.showUnitStatistics(ai[cur_index_ai])

showResults(human,ai)

t_msg = "Battle has started."
log_obj.info(t_msg)

# game play loop to handle events
game_over = False
game_round = 1
running = True
isVirtualMove = False
virtualKeypress = ""
while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 

            if not game_over:          
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        print ("s is pressed")
                        myUI.setStage()
                        if cur_index_human == 0:
                            cur_index_human = len(human)-1
                        else:
                            cur_index_human -= 1
                        
                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]

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
                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]
 
                        myUI.updateCharacter(human_unit)
                        myUI.updateCharacter(ai_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        print("Cur Index Human: ", cur_index_human)

                    if event.key == pygame.K_j:
                        print ("j is pressed")
                        myUI.setStage()
                        if cur_index_ai == 0:
                            cur_index_ai = len(ai)-1
                        else:
                            cur_index_ai -= 1

                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]

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

                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]
 
                        myUI.updateCharacter(ai_unit)
                        myUI.updateCharacter(human_unit)
                        myUI.showUnitStatistics(ai_unit)
                        myUI.showUnitStatistics(human_unit)
                        print("Cur Index AI: ", cur_index_ai)
                        
                    if event.key == pygame.K_a:
                        print ("a is pressed. attacking")
                        isInvalidAttack = False
                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]
                       
                        if human_unit.state == "DEAD" or ai_unit.state == "DEAD":
                            myUI.writeCenterScreen(" Invalid attack! Both attacker and target must be alive.", 420, 400, myUI.YELLOW,myUI.FONTSTYLE,15)
                            # exit this loop
                            break

                        if game_round % 2 > 0:                            
                            human_unit.attack(ai_unit)
                            showResults(human, ai)
                            # increment round for ai
                            game_round += 1

                            # update screen 
                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            pygame.display.update()

                            # random move for AI
                            ai_move(myUI,human,ai,ai_unit,human_unit)

                            game_round +=1

                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            
                            myUI.refresh_score_data(human,ai)
                           
                    if event.key == pygame.K_h:
                        print ("h is pressed. healing")
                        isInvalidHealAction = False
                        human_unit = human[cur_index_human]
                        ai_unit = ai[cur_index_ai]
                       
                        if human_unit.state == "DEAD":
                            myUI.writeCenterScreen(" Invalid heal action! Human target must be alive.", 420, 400, myUI.YELLOW,myUI.FONTSTYLE,15)
                            # exit this loop
                            break

                        if game_round % 2 > 0:                            
                            human_unit.heal()
                            # increment round for ai
                            game_round += 1

                            # update screen 
                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            pygame.display.update()

                            # ai's turn
                            # random move for AI
                            ai_move(myUI,human,ai,ai_unit,human_unit)
                       
                            game_round +=1

                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            
                            myUI.refresh_score_data(human,ai)
                               
                    if event.key == pygame.K_p:
                        myUI.setStage()
    
                        if game_round % 2 > 0:
                            myUI.writeCenterScreen(" Human player skipped a round!", 450, 300, myUI.WHITE,myUI.FONTSTYLE,20)
                            # increment round for ai
                            game_round += 1

                            # update screen 
                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            pygame.display.update()

                             # ai's turn
                            # random move for AI
                            ai_move(myUI,human,ai,ai_unit,human_unit)

                            game_round +=1

                            myUI.setStage()
                            myUI.setScoreboard()
                            myUI.updateCharacter(ai_unit)
                            myUI.updateCharacter(human_unit)
                            myUI.showUnitStatistics(ai_unit)
                            myUI.showUnitStatistics(human_unit)
                            
                            myUI.refresh_score_data(human,ai)
 
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
                #myUI.showLabelOnStage("Human player wins!",(560,300),myUI.FONTSTYLE,20,myUI.YELLOW)
                myUI.writeCenterScreen(" Human player wins!", 560, 300, myUI.WHITE,myUI.FONTSTYLE,20)
                game_over = True

                showResults(human,ai)
                repeat = False

            # check if HUMAN wins
            if not isPlayerAlive(human) and isPlayerAlive(ai):
                end_time = datetime.datetime.now()
                #myUI.showLabelOnStage("AI player wins!",(575,300),myUI.FONTSTYLE,20,myUI.YELLOW)
                myUI.writeCenterScreen(" AI player wins!", 560, 300, myUI.WHITE,myUI.FONTSTYLE,20)
                game_over = True

                showResults(human,ai)
                repeat = False
        
            pygame.display.update()

pygame.quit()
