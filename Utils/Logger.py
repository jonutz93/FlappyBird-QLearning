import Constants
class Logger(object):
    @staticmethod
    def Log(message):
        if Constants.useFileLogging == True:
            file  = open("Debug/logs.txt", "a")
            file.write(str(message)) 
            file.write("\n")
            file.close()
        if Constants.useConsoleLogging == True:
               print(str(message))



