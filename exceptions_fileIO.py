# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Oscar Escobar
Sec 3 Vol I
14 Sep 2023
"""

from random import choice

# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last digits differ by 2 or more: ")
    if len(step_1) != 3: raise ValueError('You must enter a 3 digit number') #Checking whether entered number is 3 digits
    int1 = int(step_1)
    if abs(int(step_1[0]) - int(step_1[-1])) < 2:                               #Checking difference of first and last digit
        raise ValueError('Entered number\'s first and last digit must be differ by 2 or more')
    
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    int2 = int(step_2)
    if step_2[::-1] != step_1:                                               #Checking to see if the reverse is inputted
        raise ValueError('The second digit is not the reverse of the first.')

    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int2 - int1):                                    #Checking to see if the positive difference is given
        raise ValueError('You must enter the positive difference')

    step_4 = input("Enter the reverse of the previous result: ")
    if step_4[::-1] != step_3:
        raise ValueError('You must enter the reverse of the difference.')
    
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    iteration = 0
    directions = [1, -1]

    #starting the iterations and completing the iterations unless a KeyBoardInterrupt is given
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
            iteration += 1
        print('Process completed')

    #Catching the keyboard interruption and giving an output of the moment stopped
    except KeyboardInterrupt as e:
        print(f'Process interrupted at iteration {iteration}')

    #Always returning the value of walk regardless of exception or not
    finally: 
        return walk
    
# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
    """

    # Problem 3
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """

        #Entering an infinite loop that only breaks once correct input is given
        while True:

            try:
                #Attempting to read the file; it seems line does not work for 4 as int or any less
                with open(filename, 'r') as file:
                    self.file_contents = file.read()
                    self.filename = filename
                    self._num_lines_ = len(file.readlines())
                
                #breaking while loop if we are able to read the file and no exceptions are raised
                break  

            #Rerunning if any exceptions are given or input is invalid
            except Exception:
                filename = input('Please enter a valid file name: ')
                continue

        """
        Creating 'hidden' attributes that help in calculating statistics of the file. We use
        list comprehension to determine the types of characters in the file by parsing through
        each character in the file and checking whether it is an alpha, a digit, or a whitespace.
        """
        self._tot_chars_ = len(self.file_contents) 
        self._alpha_chars_ = sum([s.isalpha() for s in self.file_contents])
        self._num_chars_ = sum([s.isdigit() for s in self.file_contents])
        self._whitespaces_ = sum([s.isspace() for s in self.file_contents])

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """

        #Checking validity of mode and raising error if it is not a, w, or x
        if (mode == 'a') or (mode == 'w') or (mode == 'x'):
            pass
        
        else:
            raise ValueError("Mode must either be 'a', 'w', or 'x'.")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """

        #Checking validity of mode
        ContentFilter.check_mode(self, mode = mode)

        #Writing new outfile either as upper or lower (or raising an exception)
        if case == 'upper':
            with open(file = outfile, mode = mode) as file:
                file.write(self.file_contents.upper())
        
        elif case == 'lower':
            with open(file=outfile, mode=mode) as file:
                file.write(self.file_contents.lower())

        else:
            raise ValueError("Case is either 'lower' or 'upper'")

    def reverse(self, outfile, mode='w', unit='word'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """

        lines = self.file_contents.strip().split('\n')  #Creating a list of each line & Removing the empy string that comes from the last \n seq

        #Writing the reverse of the given file but reversing line order (not word order) 
        #to a new file
        if unit == 'line':

            #Open file to write
            with open(outfile, mode=mode) as file:

                #Sort through each line in reverse and write to file
                file.writelines(line + '\n' for line in lines[::-1])

        #Writing the reverse word order with line order staying the same
        elif unit == 'word':
            
            #Open file to write
            with open(outfile, mode=mode) as file:

                #Iterate through each line and split each line into words/constituents
                for line in lines:
                    words = line.split(" ")

                    #Iterate through word and write
                    for word in words[::-1]:
                        file.writelines(word + " ")
                    
                    #Separate each line
                    file.write('\n')

        else:
            raise ValueError("Reordering must either be 'word' or 'line'")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """

        with open(outfile, mode=mode) as file:

            """Splitting the test into lines and then removing an empty string that comes from the last
            \n sequence. Then, since we assume each line has an equal number of words, we split the
            line into a list of words and take its cardinality to determine the number of words. We are 
            trying to make a list whose elements are lists of the words found on column i. Thus
            List  = [ [words found in col 1], [words in col 2]...]
            """
            lines = self.file_contents.strip().split('\n')
            num_words = len(lines[0].split(" "))
            lines_list = [[] for i in range(num_words)]

            for line in lines:
                line.strip(' ')                   #Removing the leading and trailing whitespace, if any

                words = line.split(" ")           #Splitting each line into indivdual words (a list)

                for i in range(num_words):        #Appending each word into an appropriate list
                    lines_list[i].append(words[i])

            #Now transposing the text
            for i in range(len(lines_list)):
                for j in range(len(lines_list[0])):
                    file.write(lines_list[i][j] + ' ')
                
                file.write('\n')

    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """

        return (f"Source file:\t\t{self.filename}\nTotal characters:\t{self._tot_chars_}"+
                f"\nAlphabetic characters:\t{self._alpha_chars_}\nNumerical characters:\t{self._num_chars_}"+
                f"\nWhitespace characters:\t{self._whitespaces_}\nNumber of lines:\t{self._num_lines_}")

if __name__ == '__main__':
    def test_func():

        file_name = 'cf_example1.txt'
        uni_test = 'uniform_test.txt'
        rev_test = 'reverse_test.txt'
        trans_test = 'trans_test.txt'


        obj = ContentFilter(file_name)

        if obj.filename != file_name:
            print("object's name attribute instantiated incorrectly")

        print(obj.filename)
        print(obj.file_contents)

        obj.uniform(uni_test, 'w', 'lower')

        #with open('uniform_test.txt', 'r') as uni:
        # print(f"{uni_test} file created successfully. Now reading...")
            #uni.readlines()

        obj.reverse(rev_test, unit='word', mode='x')
        obj.transpose(trans_test, mode = 'x')
        print(obj)

    test_func()