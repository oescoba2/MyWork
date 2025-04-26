# binary_trees.py
"""Volume 2: Binary Trees.
Oscar J. Escobar
Sec 2 - Vol II Labs
10 Oct 2023
"""

# These imports are used in BST.draw().
from matplotlib import pyplot as plt
from random import sample
import networkx as nx
import networkx.drawing.nx_agraph 
import time

class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """

        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """

        def _step(node):
            """Recursively stepping through list to find node with specified information.

            Parameters:
                - node: a SinglyLinkedListNode obj
            
            Returns:
                - node: the node with the specified data (if it is in the list)
                - step(node.next): recalls the function to recurse using the next node coming after given node

            Raise:
                - ValueError
            """

            if node is None:
                raise ValueError(f"{data} is not in the list")
            
            elif node.value == data:
                return node
            
            else:
                return _step(node.next)

        return _step(self.head)

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """

    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """

        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value

class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """

    def __init__(self):
        """Initialize the root attribute."""

        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """

        #Searching for duplicatte
        duplicate = False
        try:
            self.find(data)
            duplicate = True          #executes only if there is a duplicate since find returns ValueError when data is not in BST
        
        except ValueError:            #Handling ValueError when data is not to be found within BST
            pass

        if duplicate:                 #Raising a ValueError for duplicate
            raise ValueError('Cannot have duplicate value within BST')

        #Creating new node
        new_node = BSTNode(data=data)

        #Base case for empty BST
        if self.root is None:
            self.root = new_node

        #Nonempty BST
        else: 
            def _step(node):
                """ This function walks through the tree and finds the node to 
                attach the given data key value as a new child

                Parameters:
                    - node (BSTNode obj): the starting point of the tree (root)

                Returns:
                    - node (BSTNode obj): the parent node to attach the given 
                    key data value. 
                """
                
                #Walk right if given key is bigger 
                if data > node.value:
                    if node.right is None:     #Return the node is there is no right child
                        return node
                    else:                      #Continue walking right
                        return _step(node.right)
                
                #Walk left if given key is smaller
                else:                          #Return the node if there is no left child
                    if node.left is None:
                        return node
                    else:                      #Continue walking left
                        return _step(node.left)
                
            parent_node = _step(self.root)

            #Assigning child to parent node depending on key comparison to parent 
            if data > parent_node.value:
                parent_node.right = new_node
                new_node.prev = parent_node
            else:
                parent_node.left = new_node
                new_node.prev = parent_node

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        
        to_delete = self.find(data)
        parent = to_delete.prev

        #Case when node is a leaf
        if (to_delete.right is None) and (to_delete.left is None):

            #Node is root
            if to_delete is self.root:
                self.root = None

            #Delete right child
            elif to_delete.value > parent.value:
                parent.right = None

            #Delete left child
            else:
                parent.left = None

        #Case when node has only a left child
        elif (to_delete.right is None) and (to_delete.left is not None):
            child = to_delete.left

            #Root deletion
            if to_delete is self.root:
                self.root = child
                child.prev = None

            #Desired node is right child
            elif to_delete.value > parent.value:
                parent.right = child
                child.prev = parent

            #Desired node is left child
            else:
                parent.left = child
                child.prev = parent
            
        #Case when node has only a right child
        elif (to_delete.right is not None) and (to_delete.left is None):
            child = to_delete.right

            if to_delete is self.root:
                self.root = child
                child.prev = None

            elif to_delete.value > parent.value:
                parent.right = child
                child.prev = parent
            else:
                parent.left = child
                child.prev = parent

        else:
            left_child = to_delete.left

            #left child of desired deletion has a right child
            if left_child.right is not None:
                right = left_child.right
                imm_predecessor = right.value

                #Stepping right to find the immediate predecessor
                while right.right is not None:
                    if (right.right.value > imm_predecessor) and (right.right.value < to_delete.value):
                        imm_predecessor = right.right.value
                    right = right.right

                #Removing targeted node 
                self.remove(right.value)
                to_delete.value = imm_predecessor

            #Left child has not right child so it becomes the immediate predecessor
            else:
                imm_predecessor = left_child.value
                self.remove(left_child.value)
                to_delete.value = imm_predecessor

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=nx.nx_agraph.graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)

# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """

    N = [2**i for i in range(3,11)]
    single_build_times = []
    single_search_times = []
    avl_build_times = []
    avl_search_times = []
    bst_build_times = []
    bst_search_times = []

    #Reading file
    with open(file='english.txt', mode='r')  as file:
        contents_list = file.readlines()

    for n in N:
        single = SinglyLinkedList()
        avl = AVL()
        bst = BST()
        items = sample(contents_list, k=n)  #Retrieving a subset of n-items
        searches = sample(items, k=5)       #Retrieving 5 items from subset

        #build Singly Linked List
        start = time.time()
        for item in items:
            single.append(item)
        end = time.time()
        single_build_times.append(end-start)

        #Seach Singly Linked List
        start = time.time()
        for search in searches:
            single.iterative_find(search)
        end = time.time()
        single_search_times.append(end-start)

        #Build AVL Tree
        start = time.time()
        for item in items:
            avl.insert(item)
        end = time.time()
        avl_build_times.append(end-start)

        #Seach AVL Tree
        start = time.time()
        for search in searches:
            avl.find(search)
        end = time.time()
        avl_search_times.append(end-start)

        #Build BST
        start = time.time()
        for item in items:
            bst.insert(item)
        end = time.time()
        bst_build_times.append(end-start)

        #Seach BST
        start = time.time()
        for search in searches:
            bst.find(search)
        end = time.time()
        bst_search_times.append(end-start)
        
    plt.subplot(121)
    plt.loglog(N, single_build_times, color = 'blue', label = 'Singly Linked List')
    plt.loglog(N, avl_build_times, color = 'red', label = 'AVL Tree')
    plt.loglog(N, bst_build_times, color = 'green', label = 'BST')
    plt.legend(loc = 'upper left')
    plt.ylabel('Built Time(s)')
    plt.xlabel('Values of N=[2^3, 2^10] (log scale)')

    plt.subplot(122)
    plt.loglog(N, single_search_times, color = 'blue', label = 'Singly Linked List')
    plt.loglog(N, avl_search_times, color = 'red', label = 'AVL Tree')
    plt.loglog(N, bst_search_times, color = 'green', label = 'BST')
    plt.legend(loc = 'upper left')
    plt.ylabel('Search Time(s)')
    plt.xlabel('Values of N=[2^3, 2^10] (log scale)')

    plt.show()

if __name__ == "__main__":

    #--------- Test for Recursion ---------
    single_link = SinglyLinkedList()
    for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
        single_link.append(letter)
    assert single_link.recursive_find('f').value == 'f', 'failed to implement recursive find properly'

    #------- Tests for BST: Insert to an empty BST-------
    BST1 = BST()
    BST1.insert(24)
    assert BST1.root.value == 24, 'failed to insert properly at an empty BST'

    BST1.insert(60)
    BST1.insert(57)
    for val in [53, 17, 89, 76, 55, 99, 23, 85, 59, 8]:
        BST1.insert(val)
    print(BST1)
    BST1.draw()
    BST1.remove(8)
    BST1.remove(53)
    BST1.remove(60)
    BST1.remove(24)
    BST1.draw()

    """
    BST1 = BST()
    BST2 = BST()

    list1 = [4, 1, 10, 3, 5, 11, 6, 14, 9, 12, 16, 7]
    list2 = [6, 1, 7, 2, 3, 4, 5]

    for num1, num2 in zip(list1, list2):
        BST1.insert(num1)
        BST2.insert(num2)

    #BST1.draw()
    BST2.draw()

    data = 6
    BST2.remove(data)

    try:
        BST2.find(data)

    except ValueError:
        print(f'Able to remove {data}')
    """