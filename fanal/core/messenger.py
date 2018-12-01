"""
The messenger prints info depending on its
level of verbosity
"""

from sys import exit

class Messenger:

    """
    Messenger

    Provides a simple messanger class. A conditional meesage
    is printed if the level of the log function <= than the level
    of the class.
    """

    def __init__(self, level = 0, name = ""):

        self.level = level
        self.name  = name

        """
        Set the verbosity level. The higher
        number, the more verbose.
        """


    def log(self, level, *args):

        """
        Print args if level is smaller-equal than
        the verbosity level
        """
        
        if level <= self.level:
            if self.name: 
                message = self.name + ': '
            else: message = ''

            #message += '('+str(self.level)+':'+str(level)+'): ' 

            for arg in args:
                message += str(arg) + ' '

            print(message)


    def error(self, msg, code=""):
        self.log(0,"<<< ERROR >>>:",msg,code)


    def fatalError(self, msg, code=""):
        self.error(msg, code)
        self.log(0, "Fatal Error... Abort!")
        exit()


    def warning(self, msg, code=""):
        self.log(1,"<<< WARNING >>>:", msg, code) 


if __name__ == '__main__':

    m = Messenger(1)
    x = 10
    y = [1, 2, 3]
    z = {1:1, 2:2, 3:3}
    t = " text"
    m.log(0, x, y, z, t)
    m.log(1, x, y, z, t)
    m.log(2, x, y, z, t)
    
    m.warning("this is a warning", 99)
    m.error("this is an error", 999)
