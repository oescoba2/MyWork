# linked_lists.py
"""Volume 2: Linked Lists.
Óscar J. Escobar
Sec 3 - Vol II Labs
8 Oct 2023
"""

import pytest
from collections import deque as dq

# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        
        #Raising an error if the data is not of type int, float, or str
        if not isinstance(data, (int, float, str)):
            raise TypeError("Data must be an int, float, or str.")
        
        self.value = data

class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.

# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """

    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """

        self.head = None
        self.tail = None
        self._len_= 0     #Hidden attribute to keep track of length of list

    def append(self, data):
        """Append a new node containing the data to the end of the list."""

        # Create a new node to store the input data.
        new_node = LinkedListNode(data)

        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node

        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

        self._len_ += 1 #Incrementing the length of the list

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """

        #Setting a variable equal to head to be able to traverse list
        node = self.head
        
        #Checking to see if the node has address of None (last element will create an address for self.next as None 
        # so any value not in the list will get the list to None)
        while node is not None:
            if node.value == data:
                return node
            node = node.next

        raise ValueError(f"'{data}' is not in the list")

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """

        #Checking for incorrect index
        if (i < 0) or (i >= self._len_):
            raise IndexError(f"index '{i}' out of range")
        
        #Begin at tail and traverse until node is found
        if (i > (self._len_ // 2)):
            node = self.tail
            for j in range((self._len_-1), i, -1):
                node = node.prev

            return node
        
        #Begin at head and traverse until node is found
        else:
            node = self.head
            for j in range(i):
                node = node.next

            return node
        
    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """

        return self._len_

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """

        node = self.head
        list_rep = []

        for i in range(self._len_):
            list_rep.append(node.value)
            node = node.next

        return repr(list_rep)

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        
        delete = self.find(data)                  #Find node

        #deleted node is only element
        if (delete is self.head) and (delete is self.tail):
            self.head = None
            self.tail = None

        #deleted node is head
        elif delete is self.head:
            self.head = delete.next
            delete.next.prev = None

        #deleted node is tail
        elif delete is self.tail:
            self.tail = delete.prev
            delete.prev.next = None

        #All other elements
        else:
            delete.prev.next = delete.next        #delete.prev -/-> delete & delete.prev --> delete.next
            delete.next.prev = delete.prev        #delete <-/- delete.next & delete.prev <-- delete.next

        self._len_ -= 1                           #Update length

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """

        #Base case of being the end of the list
        if index == self._len_-1:
            self.append(data)

        #If not at the end of the list
        else: 
            target = self.get(i=index)
            new_node = LinkedListNode(data)          #Call it C

            #If we are appending to the beginning
            if target is self.head:
                target.prev = new_node
                new_node.next = target
                self.head = new_node 

            
            #All others
            else:
                target.prev.next = new_node         #target.prev --> C
                new_node.prev = target.prev         #target.prev <-- C
                new_node.next = target              #C --> target
                target.prev = new_node              #C <-- target

            self._len_ += 1                         #Update length only here as append does that already

# Problem 6: Deque class.
class Deque(LinkedList):
    
    def __init__(self):
        """Creating empty linked list using inheritance."""

        LinkedList.__init__(self)

    def appendleft(self, data):
        """Inserting new node to beginning of list using inheritance.
        
        Parameters:
            - data: data to be placed at the beginning
        """

        LinkedList.insert(self, index=0, data=data)

    def popleft(self):
        """Removing first node (head) and returning its data.
        
        Returns:
            - head_data: the value/data contained within the deleted head

        Raises:
            - ValueError if list is empty
        """

        head_data = self.head.value
        LinkedList.remove(self, data=head_data)

        return head_data
    
    def pop(self):
        """Inserting new node to the end of the list.

        Returns:
            - tail_data: the value/data contained within the deleted tail

        Raises:
            -ValueError if list is empty
        """

        if self._len_ == 0:
            raise ValueError('Cannot pop from an empty list')
        
        tail_data = self.tail.value
        node = self.tail
        node.prev.next = None
        self.tail = node.prev

        self._len_ -= 1

        return tail_data
    
    def remove(*args, **kwargs):
        "Raises NotImplementedError if user tries to pop from within the list"
        raise NotImplementedError("Use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        "Raises NotImplementedError if user tries to insert within the list"
        raise NotImplementedError("Use append() or appendleft() for insert")

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    
    with open(infile, 'r') as file:
        lines = file.readlines()

    deque = dq(lines)
    with open(outfile, 'x') as file:
        for line in lines[::-1]:
            file.writelines(line)

if __name__ == "__main__":
    list_1 = LinkedList()
    for x in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']:
        list_1.append(x)
    list_2 = LinkedList() #empty list
    list_3 = LinkedList() #list of only one element
    list_3.append(1)
   
    assert list_1.get(i=7).value == 'h', 'failed __get__()'
    assert list_1.get(i=4).value == 'e', 'failed __get__()'

    print("---Unaltered lists---")
    print(list_2)
    print(list_1)
    print(list_3)

    #removing one element
    print("---Remove 1 element---")
    list_1.remove('d')
    print(list_1)

    #removing single element
    print("--Removing element from list with only one element---")
    list_3.remove(1)
    print(list_3)

    #removing head
    print("---Removing head---")
    list_1.remove('a')
    print(list_1)

    #removing tail
    print("---Removing tail---")
    list_1.remove('k')
    print(list_1)

    with pytest.raises(ValueError) as e:
        list_1.find('d')
    assert e.value.args[0] == "'d' is not in the list", 'failed to remove element'

    assert list_1.head.value == 'b', 'failed to remove head properly'
    assert list_1.head.prev is None, 'failed to remove head properly'
    assert list_1.tail.value == 'j', 'failed to remove tail properly'
    assert list_1.tail.next is None, 'failed to remove tail properly'

    with pytest.raises(ValueError) as e:
        list_1.remove('z')
    assert e.value.args[0] == "'z' is not in the list", 'failed to raise ValueError on nonexistent data'

    #inserting new element
    print("---Insert new element---")
    print('within list')
    list_1.insert(index=5, data='x')
    print(list_1)

    assert list_1.get(5).value == 'x', 'failed to insert node properly within list'

    print('at head')
    list_1.insert(index=0, data='s')
    assert list_1.head.value == 's', 'failed to insert at head properly'
    print(list_1)

    print('at tail')
    list_1.insert(index=9, data='t')
    assert list_1.tail.value == 't', 'failed to insert at tail properly'
    assert list_1.tail.next is None, 'failed to insert at tail properly'
    print(list_1)

    deque = Deque()

    elems = ['a', 'b', 'c', 1, 2]

    for elem in elems:
        deque.append(elem)

    deque.appendleft('z')
    deleted = deque.pop()
    delete = deque.popleft()
    assert deleted == 2, 'failed to pop from the right/end properly'

    with pytest.raises(ValueError) as e:
        deque.find('z')
    assert e.value.args[0] == "'z' is not in the list", 'failed to remove element'

    dq2 = Deque()
    for num in [45, 77, 15, 87, 34, 26, 23, 76, 18]:
       dq2.append(num)
    dq2.popleft()
    print(dq2)

    dq3 = Deque()
    dq3.append(59)
    print(dq3)


    prob7(infile='english.txt', outfile='test_deque_write.txt')