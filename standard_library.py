# standard_library.py
"""Python Essentials: The Standard Library.
Oscar Escobar
Sec 3 Vol 1
25 June 2023
"""

import box
import calculator
import itertools
import random
import sys
import time

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    return min(L), max(L), (sum(L) / len(L))

# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """

    #Creating objects to check for mutability
    int = 4
    str = 'Hello there'
    list = ['General', 'Kenobi', 3, 4]
    tup = (1, 2, 3, 'Hi')
    set = {'Meowdy', 4.5, 8}

    #Assigning the a new name
    int2 = int
    str2 = str
    list2 = list 
    tup2 = tup
    set2 = set

    #Altering
    int2 = 7
    str2 += 'ay'
    list2.append(3.14)
    tup2 += (1,)
    set2.discard(4.5)
    
    #Displaying result using fstring
    print(f'int is mutable: {int == int2}\nstr is mutable: {str == str2}' +
          f'\nlist is mutable: {list==list2}\ntup is mutable: {tup==tup2}\nset is mutable: {set==set2}')


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """

    return calculator.sqrt(calculator.sum(calculator.product(a, a), calculator.product(b, b)))

# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """

    l = []

    for i in range(len(A)+1):
        
        #Adding the empty set
        if i == 0:
            l.append(set())
            continue
        
        #Adding the rest of the combinations of the set
        for Set in list(itertools.combinations(A, i)):
            l.append(set(Set))
    
    return l

# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""

    #Initiating the game
    print('Welcome to shut the box game. Shall we play a game? Let us begin!\n')
    name = player
    nums = [i for i in range(1,10)]
    roll = random.randint(1,12)
    timelim = float(timelimit)
    game_timer = timelim   #Using a copy in order to be able to find the total time played later

    while (box.isvalid(roll, nums)) and (game_timer > 0):

        print('Numbers left: ', nums)
        print("Roll: ", roll)
        print('Seconds left: ', round(game_timer, 2))

        start_time = time.time()
        num_choice = input('Numbers to eliminate: ')
        eliminate  = box.parse_input(num_choice, nums)

        #Checking to see if the given input can be obtained from the given numbers list (nums)
        #If not, code continues to ask for input
        while (len(eliminate) == 0) or (box.isvalid(roll, eliminate) is False): 

            print('Invalid input\n')
            end_time = time.time()
            game_timer-= (end_time - start_time)

            if game_timer <=0 :
                break

            print('Seconds left: ', round(game_timer, 2))
            start_time = time.time()
            num_choice = input('Numbers to eliminate: ')
            eliminate  = box.parse_input(num_choice, nums)

        #Updating time and stopping clock
        end_time = time.time()
        game_timer -= (end_time - start_time)

        #Checking to see if we only need to roll one die or not
        if sum(nums) > 6: roll = random.randint(1,12)
        else: roll = random.randint(1,6)

        #Exitting if timer ends
        if game_timer <= 0:

            print('Game over!\n')
            break
        
        for num in eliminate:
            nums.remove(num)

        print()
        
    if len(nums) == 0:
        print('Score for {}: {} points\nTime played: {} seconds\nCongratulations!! You shut the box!'.format(name, 
                sum(nums), round(timelim - game_timer, 2)))

    else:
        print('Score for {}: {} points\nTime played: {} seconds\nBetter luck next time, LOSER!'.format(name, 
                sum(nums), round(timelim - game_timer, 2)))

if __name__ == '__main__':
   
   if len(sys.argv) == 3:     #Allows the user to play Shut the Box if the number of given arguments are 3
       shut_the_box(player = sys.argv[1], timelimit = sys.argv[2])
