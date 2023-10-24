####  Homework 4.5 - Coding Problems 4.26-4.27  #### 

def knapsack(w, items):
    """Function for Problems 4.26-4.27.
    
    Implements a dynamic programming approach to the {0,1} 
    knapsack problem (where no multiples are allowed).

    Returns the maximum value that can be carried in the 
    knapsack for the given items and maximum weight (w).

    Also returns a list of how many of each item should be 
    included (either 1 or 0) to achieve the maximum value. This 
    list is in the same order of the original items list.

    Examples:
        knapsack(100, [(20, 0.5), (100, 1.0)]) 
            should return (1.0, [0,1])
        knapsack(10, [(6, 30), (3, 14), (4, 16), (2, 9)])
            should return (46, [1,0,1,0])

    Parameters:
        w (int): The maximum weight allowed in the knapsack 
        items(list[tuple]): A list of potential items to include 
            in the knapsack. Each item in the list is a tuple of
            the form (item weight, item value)
    
    Returns:
        float - The maximum value that can be carried in the knapsack 
        list - A list of 1s and 0s representing which items to include 
            in the knapsack to achieve this maximum value. 
    """
    #items = [(weight1, value1), ... (weight-n, value-n)]
    n = len(items)
    items_dict = {0: (0 , [0]*n)} #Weight : tuple of (max_value of the linear combinations of the each item's value,
                                                      #combination that got to that value(some list))

    #Iterate starting from base case up to weight limit W
    for weight in range(1, w+1):
        max_val = 0 #the 
        max_items = [0]*n

        #Loop through each item within the given set Items
        for index, tup_item in enumerate(items):
            item_weight = tup_item[0]
            item_val = tup_item[1]

            if weight - item_weight >= 0:
                curr_max_val, combination = items_dict[weight - item_weight]

                if (combination[index] == 0) and (curr_max_val + item_val > max_val):
                    max_val = curr_max_val + item_val
                    max_items = combination.copy()
                    max_items[index] = 1
        
        items_dict[weight] = (max_val, max_items)

    return items_dict[w]

# -------- USE THIS AREA FOR TESTING YOUR CODE ------- # 
if __name__ == "__main__":
    Items = [(20, 0.5), (100, 1.0)]
    max_weight = 100

    val, combi = knapsack(w=max_weight, items=Items)
    print(val)
    print(combi)