import pygame
import random
import time
import datetime
import os


# ------- OS Compatibility code here -----------
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

print(os.name)
print(dirname)

# global variables
global game_round
game_round = 1
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

        self.dead_mark = "/Users/joeldizon/Documents/Project_BattleGame/assets/dead.png"

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

    def updatecharacter(self,objPlayer,playertype,cur_index):
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

        #show_navmessage("[ S ]/[ F ] Select a Human unit  [ J ]/[ L ] Select an AI unit ",380,590,22)
        show_navigation("[ S ]/[ F ] Select a Human unit  [ J ]/[ L ] Select an AI unit [ A ] Attack an opponent [ H ] Heal a unit [ P ] Pass a move",300,610,12)
        show_navmessage("Round ",520,130,40)
        show_navmessage(str(game_round),630,130,40)
        if game_round % 2 > 0:
            show_navmessage("Human's turn to attack!",470,165,30)
        else:
            show_navmessage("AI's turn to attack!",490,165,30)
       
        self.show_playerstats(human[cur_index_human],"human")
        self.show_playerstats(ai[cur_index_ai],"ai")

    def update_uniticons(self,t_player,t_coord):
        i=0
        for p in t_player:
            t_icon = p.icon
            icon = pygame.image.load(t_icon)
            avatar = pygame.transform.scale(icon,(100,200))
            screen.blit(icon,(t_coord[i][0],t_coord[i][1]))
            i += 1

    def update_deadmark(self,t_player,t_coord):
        i=0
        for p in t_player:
            print("Name: ", p.name, " State: ",p.state)
            if p.state == "DEAD":
                t_icon = self.dead_mark
                icon = pygame.image.load(t_icon)
                avatar = pygame.transform.scale(icon,(70,70))
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
                #t_fontcolor = (25,25,112)

            font = pygame.font.SysFont(t_font, t_fsize)
            img = font.render(str(t_value), True,t_fontcolor)
            screen.blit(img, (t_coord[i][0], t_coord[i][1]))       
            i += 1

    def refresh_score_data(self,human,ai):
        # update unit head icons (human player)
        t_coord = [[190,648],[365,648],[190,698],[365,698],[190,752],[365,752]]
        self.update_uniticons(human,t_coord)

        # update unit head icons (ai player)
        t_coord = [[674,648],[849,648],[674,698],[849,698],[674,752],[849,752]]
        self.update_uniticons(ai,t_coord)

        # update dead mark for human player
        coord = [[180,640],[350,640],[180,690],[350,690],[180,743],[350,743]]
        self.update_deadmark(human,coord)

        # update dead mark for ai player
        coord = [[657,640],[835,640],[657,690],[835,690],[657,743],[835,743]]
        self.update_deadmark(ai,coord)

        # update human coins
        locX = 115
        locY = 770
        font = pygame.font.SysFont(None, 25)
        img = font.render(str(humans_coins), True, (255, 215, 0))
        screen.blit(img, (locX, locY))       

        # update ai coins
        locX = 1050
        locY = 770
        font = pygame.font.SysFont(None, 25)
        img = font.render(str(robots_coins), True, (255, 215, 0))
        screen.blit(img, (locX, locY))       
      
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

    def show_playerstats(self,t_unit,t_playertype):
        if t_playertype == "human":
            t_locx = 20
            t_locy = 135
        else:
            t_locx = 1105
            t_locy = 135

        font = pygame.font.SysFont("calibri", 28)
        img = font.render(t_unit.name, True, (127, 127, 115))
        screen.blit(img, (t_locx, t_locy))  


# ALL FUNCTIONS


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
        init_state = "ALIVE" # random.choice(["ALIVE","DEAD"])
       
        # instantiate list class for human player
        objplayer.append(Player(init_name, init_prof, init_index, init_state, init_atk, init_dep, init_avatar, init_icon,init_posx,init_posy,init_avatarsize))

def show_navmessage(t_msg,t_locx, t_locy,t_fontsize):
    font = pygame.font.SysFont(None, t_fontsize)
    img = font.render(t_msg, True, (255, 255, 255))
    screen.blit(img, (t_locx, t_locy)) 

def show_navigation(t_msg,t_locx, t_locy,t_fontsize):
    font = pygame.font.SysFont("Calibri", t_fontsize)
    img = font.render(t_msg, True, (255, 255, 255))
    screen.blit(img, (t_locx, t_locy)) 

def show_message(t_msg,t_locx, t_locy,t_fontsize):
    font = pygame.font.SysFont(None, t_fontsize)
    img = font.render(t_msg, True, (244, 243, 239))
    screen.blit(img, (t_locx, t_locy))  

def calculate_damagepoints(atp,dfp):
    result = atp - dfp + random.randint(EXTRA_MIN_DMG, EXTRA_MAX_DMG)
    # convert damage points to positive (to make it more intuitive)
    result = int(result * -1)
    # reset result to 0 if dmg points is negative
    if result < 0:
        result = 0
    return result

def update_playerstate(objplayer):
    if objplayer.healthpoint <= 0:
        # promote
        objplayer.state = "DEAD"

def promote_player(objplayer):
    if objplayer.experience >= MAX_EXP:
        # promote
        objplayer.rank += 1
        # reset experience
        objplayer.experience = 0 

def update_experience(objplayer,exp):
    # calculate experence
    objplayer.experience += exp

def update_extra_experience(objplayer,dmgpts):
    # calculate target's experence
    if dmgpts > HIGH_DMG:
        # increase experience by 20% if damage points is more than 10
        objplayer.experience = int(objplayer.experience * LOW_EXP)
    
    if dmgpts <= LOW_DMG:
        # increase experience by 50% if damage points is below 10
        objplayer.experience = int(objplayer.experience * HIGH_EXP)

def update_healthpoints(objplayer,dmgpts):
    objplayer.healthpoint = int(objplayer.healthpoint - dmgpts)

def generate_points(min,max):
    result = random.randint(min,max)
    return result

def select_alive_player(objplayer):
    selected = False
    while selected == False:
        player_index = random.randint(0,len(objplayer)-1)
        if objplayer[player_index].state == "ALIVE":
            result = player_index
            selected = True
        else:
            selected = False
    return result

def get_unit_to_heal(objplayer):
    alive_units = []

    # get all alive units
    for p in objplayer:
        if p.state == "ALIVE":
            alive_units.append([p.index,p.healthpoint])

    # sort list by its healthpoints in ascending. the one on top will be the unit with the least hp which require immediate healing
    alive_units = sorted(alive_units,key=lambda x: x[1])
    
    # get the index of unit to heal
    result = alive_units[0][0] - 1
    
    return result

def heal(target):
 
    # calculate heal points
    heal_points = random.randint(HEAL_MIN,HEAL_MAX)

    # heal a unit by increasing healthpoint with additional heal_points
    target.healthpoint += heal_points
    
    # check if unit has exceeded its healthpoint  (100), if so reset to 100 
    if target.healthpoint > 100:
        target.healthpoint = 100
    
    # show results
    print("The unit ",target.name, " is healed.")
    print(heal_points, " health points has been added!")
    return

def isalive(objplayer):
    result = False
    alive = 0
    for p in objplayer:
        if p.state == "ALIVE":
            alive += 1
    if alive > 0:
        result = True   
    return result

def isUnitAlive(t_unit):
    result = False
    if t_unit.state == "ALIVE":
        result = True
    return result

def update_coins(player_type, coins):
    global humans_coins, robots_coins

    if player_type == "humans":
        humans_coins += coins
    if player_type == "robots":
        robots_coins += coins


def attack(attacker,target):
    # calculate damage points. use the initialized atk and dep
    dmgpts = calculate_damagepoints(attacker.atk,attacker.dep)
 
    print(dmgpts)
     
    # update target's healthpoints
    update_healthpoints(target,dmgpts)

    # check if target is still alive, update state
    update_playerstate(target)
 
    # update the attacker's experience
    update_experience(attacker,dmgpts)
    update_extra_experience(attacker,dmgpts)

    # update the target's experience
    update_experience(attacker,target.dep)
    update_extra_experience(target,dmgpts)

    # promote attacker
    promote_player(attacker)

    #return
    # promote target
    promote_player(target)

    update_coins("humans",dmgpts)

    update_coins("robots",target.dep)

def show_results(objplayer1, objplayer2):
    print("Player Group: HUMANS") 
    print("Total Coins :", humans_coins) 
    for player in objplayer1:
        print("Index: ", player.index, " | State: ", player.state, " | Name: ", player.name, "| Prof: ", player.profession, " | HP: ", player.healthpoint, " | EXP: ", player.experience, " | ATK: ", player.atk, " | DEF: ", player.dep, " | Rank: ", player.rank)

    print("Player Group: ROBOTS") 
    print("Total Coins :", robots_coins) 
    for player in objplayer2:
        print("Index: ", player.index, " | State: ", player.state, " | Name: ", player.name, "| Prof: ", player.profession, " | HP: ", player.healthpoint, " | EXP: ", player.experience, " | ATK: ", player.atk, " | DEF: ", player.dep, " | Rank: ", player.rank)
 

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
xunits = 3
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
myUI.updatecharacter(human,"human",cur_index_human)
myUI.updatecharacter(ai,"ai",cur_index_ai)
myUI.show_playerstats(human[cur_index_human],"human")
myUI.show_playerstats(ai[cur_index_ai],"ai")

show_results(human,ai)

# main loop to handle events
game_round = 1
running = True
while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False           
            if event.type == pygame.KEYDOWN:
                print (event.type)
                if event.key == pygame.K_s:
                    print ("s is pressed")
                    myUI.setStage()
                    if cur_index_human == 0:
                        cur_index_human = len(human)-1
                    else:
                        cur_index_human -= 1
                    myUI.updatecharacter(human,"human",cur_index_human)
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    print("Cur Index Human: ", cur_index_human)
                if event.key == pygame.K_f:
                    print ("f is pressed")
                    myUI.setStage()
                    if cur_index_human == len(human)-1:
                        cur_index_human = 0
                    else:
                        cur_index_human += 1                            
                    myUI.updatecharacter(human,"human",cur_index_human)
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    print("Cur Index Human: ", cur_index_human)

                if event.key == pygame.K_j:
                    print ("h is pressed")
                    myUI.setStage()
                    if cur_index_ai == 0:
                        cur_index_ai = len(ai)-1
                    else:
                        cur_index_ai -= 1
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    myUI.updatecharacter(human,"human",cur_index_human)
                    print("Cur Index AI: ", cur_index_ai)

                if event.key == pygame.K_l:
                    print ("l is pressed")
                    myUI.setStage()
                    if cur_index_ai == len(ai)-1:
                        cur_index_ai = 0
                    else:
                        cur_index_ai += 1
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    myUI.updatecharacter(human,"human",cur_index_human)
                    print("Cur Index AI: ", cur_index_ai)
                    
                if event.key == pygame.K_a:
                    print ("a is pressed. attacking")
                    isInvalidAttack = False
                    if game_round % 2 > 0:
                        attacker = human[cur_index_human]
                        target = ai[cur_index_ai]
                        print("attacker",attacker.state)
                        print("target",target.state)
                        
                        if attacker.state == "DEAD" or target.state == "DEAD":
                            isInvalidAttack = True
                            print("invalid attack")
                    else:
                        attacker = ai[cur_index_ai]
                        target = human[cur_index_human]
                        print("attacker",attacker.state)
                        print("target",target.state)

                        if attacker.state == "DEA" or target.state == "DEAD":
                            isInvalidAttack = True
                            print("invalid attack")

                    if isInvalidAttack:
                        myUI.setStage()
                        myUI.setScoreboard()
                        myUI.updatecharacter(ai,"ai",cur_index_ai)
                        myUI.updatecharacter(human,"human",cur_index_human)
                        myUI.refresh_score_data(human,ai)
                        show_message("Invalid attack! Both attacker and target must be alive. ",380,210,22)

                    else:
                        attack(attacker,target)
                        game_round +=1
                        myUI.setStage()
                        myUI.setScoreboard()
                        myUI.updatecharacter(ai,"ai",cur_index_ai)
                        myUI.updatecharacter(human,"human",cur_index_human)
                        myUI.refresh_score_data(human,ai)

                    print("Round", game_round)
                    print("Attacker: ",attacker.name)
                    print("Target: ",target.name)

                    print("Cur Index AI: ", cur_index_ai)
                if event.key == pygame.K_h:
                    print ("h is pressed. healing")

                    if game_round % 2 > 0:
                        attacker = human[cur_index_human] 
                    else:
                        attacker = ai[cur_index_ai] 

                    heal(attacker)
                    game_round +=1

                    myUI.setStage()
                    myUI.setScoreboard()
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    myUI.updatecharacter(human,"human",cur_index_human)
                    myUI.refresh_score_data(human,ai)
                    print("healing ", attacker.name)
                    print("Cur Index AI: ", cur_index_ai)
 
                if event.key == pygame.K_p:
                    myUI.setStage()
 
                    if game_round % 2 > 0:
                        show_message("Human player passed the previous round! ",440,210,22)
                    else:
                        show_message("AI player passed the previous round! ",450,210,22)
                    game_round +=1
                    myUI.setScoreboard()
                    myUI.updatecharacter(ai,"ai",cur_index_ai)
                    myUI.updatecharacter(human,"human",cur_index_human)
                    myUI.refresh_score_data(human,ai)

                    print("Cur Index AI: ", cur_index_ai)
                    print ("p is pressed. pass")
        
        # check if HUMAN wins
        if isalive(human) and not isalive(ai):
            end_time = datetime.datetime.now()
            show_message("Human player wins! ",440,210,46)

            print("--------------")
            print("HUMAN WINS!!!!")
            print("--------------")
            print("Total Rounds: ", round)
            print("--------------")
            print("Start Time: ", start_time)
            print("End Time:   ", end_time)
            print("--------------")
            # display status
            show_results(human,ai)
            repeat = False

        # check if HUMAN wins
        if not isalive(human) and isalive(ai):
            end_time = datetime.datetime.now()
            show_message("AI player wins! ",440,210,46)
            print("--------------")
            print("ROBOTS WINS!!!!")
            print("--------------")
            print("Total Rounds: ", round)
            print("--------------")
            print("Start Time: ", start_time)
            print("End Time:   ", end_time)
            print("--------------")
        # display status
            show_results(human,ai)
            repeat = False
    
        pygame.display.update()


pygame.quit()
