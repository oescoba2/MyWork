# HOMEWORK 4.1 - CODING PROBLEMS 

from collections import deque
import numpy as np

#  -------------------- Problem 4.1 --------------------  # 
def naive_fib(n):
    """Return the n'th Fibonacci number (zero-indexed) 
    using naive recursion.
    
    Examples:
        naive_fib(0) should return 1.
        naive_fib(1) should return 1.
        naive_fib(2) should return 2
        .... etc etc. ....

    Parameters:
        n (int): The index of the fibonnaci number to return 
            (starting at zero)
    
    Returns:
        int: The value of the n'th fibonnaci number.
    """

    #Base case of Fibonnaci Sequence
    if (n==1) or (n==0):
        return n
    
    #Recurse back to find the n-1 and n-2 Fibonnaci number, recursing as far as back as to the base case
    return naive_fib(n-1) + naive_fib(n-2)

def memoized_fib(n):
    """Return the n'th Fibonacci number (zero-indexed) 
    using memoized recursion
    
    Examples:
        memoized_fib(0) should return 1.
        memoized_fib(1) should return 1.
        memoized_fib(2) should return 2
        .... etc etc. ....

    Parameters:
        n (int): The index of the fibonnaci number to return 
            (starting at zero)
    
    Returns:
        int: The value of the n'th fibonnaci number.
    """

    fib_dict = {0:0, 1:1} #base cases for fibonacci sequence

    def recurse(num):
        """Function that performs that actual recursive step. Function updates the dictionary 
        after checking to see if desired n-th Fibonacci number is in the dictionary.

        Parameters:
            - num (int): the actual number of the Fibonacci Sequence we want

        Returns:
            - fib_dict[num]: the value of the nth Fibonacci number stored within the updated dictionary
        """
        
        #Return dictionary lookup value if we have it
        if num in fib_dict.keys():
            return fib_dict[num]
        
        #Else, calculate the n-th value and store each individual n-1, n-2 along the way 
        else:
            fib_dict[num] = recurse(num-1) + recurse(num-2)
            return fib_dict[num]

    return recurse(n)

def bottom_up_fib(n):
    """Return the n'th Fibonacci number (zero-indexed) 
    using bottom-up dynamic programming.
    
    Examples:
        bottom_up_fib(0) should return 1.
        bottom_up_fib(1) should return 1.
        bottom_up_fib(2) should return 2
        .... etc etc. ....

    Parameters:
        n (int): The index of the fibonnaci number to return (starting 
            at zero)
    
    Returns:
        int: The value of the n'th fibonnaci number.
    """
    fib_deq = deque([0,1])

    #Deleting the left most value we do not need for calculating n-th Fibonacci number (i.e. deleting n-3)
    for i in range(2, n):
            fib_deq.append(fib_deq[0] + fib_deq[1])
            fib_deq.popleft()
    
    return fib_deq[0]+fib_deq[1]

#  -------------------- Problem 4.2 --------------------  # 
def naive_coins(v, C):
    """Return the optimal number of coins in the change-making 
    problem for v cents and the coinage system given by C.

    Use a naive recursive approach.

    Examples:
        naive_coins(5, [1,5,10,25,50,100]) should return 1
        naive_coins(7, [1,5,10,25,50,100]) should return 3
    
    Parameters:
        v (int): The amount of change to make (in cents)
        C (list(int)): The coinage system (an ordered list of
            integers from least to greatest coin value)
    
    Returns:
        int: The optimal number of coins to use to make change
            in the amount of v cents
        list: The combination of coin values used
    """

    if v == 0:
        return 0
    
    if v in C:
        return 1
    
    min_coins = 1 + min([naive_coins(v - coin, C) for coin in C if v-coin >= 0])

    return min_coins

def bottom_up_coins(v, C):
    """Return the optimal number of coins in the change-making 
    problem for v cents and the coinage system given by C.

    Use a bottom-up dynamic programming approach.

    Examples:
        bottom_up_coins(5, [1,5,10,25,50,100]) should return 1
        bottom_up_coins(7, [1,5,10,25,50,100]) should return 3
    
    Parameters:
        v (int): The amount of change to make (in cents)
        C (list(int)): The coinage system (an ordered list of
            integers from least to greatest coin value)
    
    Returns:
        int: The optimal number of coins to use to make change
            in the amount of v cents
        list: The combination of coin values used
    """
    
    coins_dict = {0:0}      #Base Case of no change (initializing dictionary); Change Amount Needed (key): Minimal number of coins needed to make change (value)
    for c in C:             #Base Cases of all coins present in the coinage system
        coins_dict[c] = 1   

    def recurse(num):
        if (num < min(coins_dict.keys())) and (num != 0):
            raise ValueError(f'Cannot have a value smaller than smallest coin {C[0]} (but it can be 0)')
        
        #Case for when the given number is an actual value already in the dictionary
        if num in coins_dict.keys():
            return coins_dict[num]
        
        #Calculating when value is not dictionary
        else:
            for i in range(1, num+1):
                coins_dict[i] = 1+ min([recurse(i - coin) for coin in C if i-coin>=0])

        return coins_dict[num]
    
    return recurse(v)

#  -------------------- Problem 4.3 --------------------  # 
def greedy_coins(v, C):
    """Return the optimal number of coins in the change-making 
    problem for v cents and the coinage system given by C.

    Use a greedy approach.

    Examples:
        greedy_coins(5, [1,5,10,25,50,100]) should return 1
        greedy_coins(7, [1,5,10,25,50,100]) should return 3
    
    Parameters:
        v (int): The amount of change to make (in cents)
        C (list(int)): The coinage system (an ordered list of
            integers from least to greatest coin value)
    
    Returns:
        int: The optimal number of coins to use to make change
            in the amount of v cents
        list: The combination of coin values used
    """

    num_coins = 0

    def recurse(num):
        nonlocal num_coins

        if num == 0:
            return 0

        else:
        
            start_coin = max([coin for coin in C if num-coin>=0])
            num_coins += 1
            recurse(num - start_coin)

            return num_coins

    return recurse(v)

if __name__ == "__main__":
    # Write any code used to test your algorithms here 
    coin_system = [1,5,10,25,50,100]
    change_needed = 68
    a = bottom_up_coins(change_needed, coin_system)
    #b = naive_coins(change_needed, coin_system)
    c = greedy_coins(change_needed, coin_system)
    print("Bottom Up:", a)
    print("Greedy:", c)