import random
import time
import datetime
import logging
import os

# ------- OS Compatibility code here -----------
dirname = os.path.dirname(__file__)
# -------

class Player:
    def __init__(self, name, profession, index,state,atk,dep):
        self.index = index
        self.state = state
        self.name = name
        self.profession = profession
        self.healthpoint = 100
        self.atk = atk
        self.dep = dep
        self.experience = 0
        self.rank = 0    

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

def update_coins(player_type, coins):
    global humans_coins, robots_coins

    if player_type == "humans":
        humans_coins += coins
    if player_type == "robots":
        robots_coins += coins


def attack(attacker,target):
    print(attacker.profession)
    print(target.profession)

    # ----- start ------
    # ----- use this to randomly generate atk and dep for attacker
    # if attacker.profession == "WARRIOR":
    #     # calculate attack points
    #     atp = generate_points(WARR_MIN_ATK, WARR_MAX_ATK)
    #     print("atp = ",atp)

    #     # calculate defense points
    #     dfp = generate_points(WARR_MIN_DEP, WARR_MAX_DEP)
    #     print("dfp = ",dfp)

    # #return
    # if attacker.profession == "TANK":
    #     # calculate attack points
    #     atp = generate_points(TANK_MIN_ATK, TANK_MAX_ATK)

    #     # calculate defense points
    #     dfp = generate_points(TANK_MIN_DEP, TANK_MIN_DEP)
    # ----- end ------

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
   
def write_to_logfile(t_msg):
    print("logging...")
    now = datetime.datetime.now()
    t_filename = dirname + "/logs/" + now.strftime("%Y%m%d") + "-1.log"
    logging.basicConfig(filename=t_filename, encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(t_msg)


# initialize logging
write_to_logfile("-" + "Started the programe.")

# set list for players

humans = []
robots = []

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

# global variables
humans_coins = 0
robots_coins = 0
start_time = datetime.datetime.now()
end_time = datetime.datetime.now()

# initial ATK, DEP
WARR_ATK = random.randint(WARR_MIN_ATK, WARR_MAX_ATK)
TANK_ATK = random.randint(TANK_MIN_ATK, TANK_MAX_ATK)
WARR_DEP = random.randint(WARR_MIN_DEP, WARR_MAX_DEP)
TANK_DEP = random.randint(TANK_MIN_DEP, TANK_MAX_DEP)

# set human players manually
humans.append(Player("PlayerA", "WARRIOR", 1,"ALIVE",WARR_ATK,WARR_DEP))
humans.append(Player("PlayerB", "TANK", 2,"ALIVE",TANK_ATK,TANK_DEP))
humans.append(Player("PlayerC", "WARRIOR", 3,"ALIVE",WARR_ATK,WARR_DEP))

# auto selection for AI players
prof = ["TANK","WARRIOR"]
for i in range(len(humans)):
    # initialize AI name
    n = "AI" + "%02d" % (random.randint(1,99),) 
    # initialize AI profession
    p = random.choice(prof)
    if p == "WARRIOR":
        # initialize for WARRIOR
        atk = WARR_ATK
        dep = WARR_DEP
    else:
        # initialize for TANK
        atk = TANK_ATK
        dep = TANK_DEP

    robots.append(Player(n, p, i+1, "ALIVE", atk, dep))

repeat = "Y"
round = 1
while repeat == "Y":
    print("--------------")
    print("Round: ",round)
    print("--------------")
    # display status
    show_results(humans,robots)

    # main loop for fight sequence
    action = ""
    #choices = ["A","a","P","p","H","h"]   
    choices = ["A","a","H","h"]
    while action not in choices:
        # alternating turns between humans and robots players
        if round % 2 == 0:
            # sequences for robot player actions
            print("--------------")
            print("Robot's turn...")
            print("--------------")
            print("Select action [A]ttack, [P]ass, [H]eal :")
            action = random.choice(choices)
            print("Robot's action choice: ",action)
            # attack sequences
            if action.upper() == "A":
                attacker = select_alive_player(robots)
                print("Attacker: ",robots[attacker].name)
                target = select_alive_player(humans)
                print("Target: ",humans[target].name)

                # attack!!!!
                attack(robots[attacker],humans[target])

            # heal sequences
            if action.upper() == "H":
                print("Enter index no. of attacker to heal: ")
                attacker = get_unit_to_heal(robots)
                print("Attacker: ",robots[attacker].name)
                print("Attacker ",robots[attacker].name, " is healed!")

                # heal a unit
                heal(robots[attacker])


            # pass/skip sequences 
            if action.upper() == "P":
                print("Robot player skipped a move! ")
        else:  
            # # MANUAL sequences for human player actions
            # print("--------------")
            # print("Human's turn...") 
            # print("--------------")
            # action = input("Select action [A]ttack, [P]ass, [H]eal :")
            # # attack sequences
            # if action.upper() == "A":
            #     attacker = int(input("Enter index no. of attacker: "))-1
            #     print("Attacker: ",humans[attacker].name)
            #     target = int(input("Enter index no. of target: "))-1
            #     print("Target: ",robots[target].name)

            #     # attack!!!!
            #     attack(humans[attacker],robots[target])

            # # heal sequences
            # if action.upper() == "H":
            #     attacker = int(input("Enter index no. of attacker to heal: "))-1
            #     print("Attacker: ",humans[attacker].name)
            #     print("Attacker ",humans[attacker].name, " is healed!")

            # # pass/skip sequences 
            # if action.upper() == "P":
            #      print("Human player skipped a move! ")

            # sequences for robot player actions
            
            # AUTO SEQUENCE FOR HUMAN. FOR TESTING PURPOSES
            print("--------------")
            print("Human's turn...")
            print("--------------")
            print("Select action [A]ttack, [P]ass, [H]eal :")
            action = random.choice(choices)
            print("Human's action choice: ",action)
            # attack sequences
            if action.upper() == "A":
                attacker = select_alive_player(humans)
                print("Attacker: ",humans[attacker].name)
                target = select_alive_player(robots)
                print("Target: ",robots[target].name)

                # attack!!!!
                attack(humans[attacker],robots[target])

            # heal sequences
            if action.upper() == "H":
                print("Enter index no. of attacker to heal: ")
                attacker = get_unit_to_heal(humans)
                print("Attacker: ",humans[attacker].name)
                print("Attacker ",humans[attacker].name, " is healed!")

                # heal a unit
                heal(humans[attacker])

            # pass/skip sequences 
            if action.upper() == "P":
                print("Robot player skipped a move! ")

    # check if HUMAN wins
    if isalive(humans) and not isalive(robots):
        end_time = datetime.datetime.now()
        print("--------------")
        print("HUMAN WINS!!!!")
        print("--------------")
        print("Total Rounds: ", round)
        print("--------------")
        print("Start Time: ", start_time)
        print("End Time:   ", end_time)
        print("--------------")
        # display status
        show_results(humans,robots)
        repeat = False

    # check if HUMAN wins
    if not isalive(humans) and isalive(robots):
        end_time = datetime.datetime.now()
        print("--------------")
        print("ROBOTS WINS!!!!")
        print("--------------")
        print("Total Rounds: ", round)
        print("--------------")
        print("Start Time: ", start_time)
        print("End Time:   ", end_time)
        print("--------------")
       # display status
        show_results(humans,robots)
        repeat = True

    # readying for the next round (if needed)
    round += 1

    # slow down the game
    time.sleep(0.05)

    







