# website_projects
This repository is to show my current projects that I am doing as part of my coursework for my degree in Applied and Computational Mathematics as well as to showcase my coding 
knowledge and style. This is repository is also meant to help me show that while I may not have full work experience in applying the theory I have thus far learned, I am working
in projects that test my ability to be able to make sense of the theory and make an application of it in a useful manner. For example, in my algorithms & approximation class, 
we have finished covering the graph theory and data structures chapter and are now working in implementing a linked-list program and a binary search tree. In Mathematical 
analysis, we are covering linear analysis and getting into matrix properties, and so the concurrent lab projects are at the moment linear transformations and complexity
analysis of matrix operations and linear system solving using LU decomposition. 

Labs will only be given a short description and the title as full problem explanations can be found at https://foundations-of-applied-mathematics.github.io. Written homework will be labeled appropriately as well as given an in-depth description of the problem. The written homework comes from the following textbooks: 
* Foundations of Applied Mathematics, Volume 1: Mathematical Analysis
* Foundations of Applied Mathematics, Volume 2: Algorithms, Approximation, Optimization

Feel free to contact me for any question using form found at the bottom of my webpage https://oescoba2.github.io. 


### HW_4_1Probs (Dynamic Programming Homework, Vol 2)
This was part of written homework when we first discussed dynammic programming (top-down/memoization, bottom-up/tabular, and greedy algorithms). I had a particular hard time with this aspect as I was not understanding how to recurse but I was able to meet with Dr. Boyd and further learn and implement these solutions to the homework. 

The homework problems were to create a memoized, tabular, and naive method for calculating the n-th Fibonacci term and then test our code for values [1, 40] and compare calculation times. I was able to get to the 2-millionth Fibonnaci number using my bottom-down approach where a deque was implemented. 

The last homework problem was to create a bottom-up and greedy approach to the minimal number of coins, in a given coin system, needed to complete a given amount of change (i.e. how many coins would be needed for a change of x-cents). I was able to produce a greedy algorithm as well as bottom up algorithm. We were not asked on memoization since that was a given example in the textbook. I did not work on returning the list of coins used since this was the problem I had troubles with that and cared more for understanding the concept of how to dynamically program theses and work on that and then if I had time afterwards comeback and implement the second half. 

### hw_4_5_coding (Vol 2)
This algorithm was intended to solve a {0,1} (that is either you take one or zero of an item) knapsack problem of n-items with given weights (w1,...,wn) and values (v1,...,vn). The code returns the maximum number of items that can be placed into the knapsack given a weight limit W. I had fun working on this logic. It was rather difficult, but I was able to help others learn from my example since I coded the problem with the guidance of the TA in just ensuring I was incorporating all cases. The code does assume that the given items will be unique, so there are no repeats. I was going to account for possible repeats of an item (with a different weight) but the autograder only needed the case for unique items. 

## Labs

### Exceptions File Input/Output
This lab was intended to help me understand the syntax of working with files and of writing to them. We were also tested on our understanding and subsequent implementation of raising exceptions. I received a 49/50 for this lab. 

### Standard Library

The lab was inteded in helping me understand the built-in functions of the Standard Library that Python comes with. The best part was to create a similuation of the popular British pub game Shut the Box. I was able to only beat it once but loved the part we got to program a game. I received a full 50/50 for this lab. 

### Linked Lists

This lab was intended in helping me practice more of my Python Object-Oriented Programming as well as introduce me on how to code a doubly-linked list and then implement a deque. We were also asked to implement unit-tests for our code
on a separate file. But I added some further just to ensure it worked fine. I received a 45/50 on this one as it failed one test case. 

### Binary Search Trees

I very much enjoyed this lab. The lab was intended to help me learn to code a BST tree using some recursive techniques for searching and inserting nodes. It also helped me learn the difference in time complexities of BST vs. AVL vs. Singly Linked Lists. This one tested more of my knowledge about programming thus far as well as comprehension of recursion. I received a 50/50