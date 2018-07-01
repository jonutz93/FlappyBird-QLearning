class Logger(object):
    @staticmethod
    def Log(message):
        file  = open("logs.txt", "a")
        #file.write(message) 
        #file.write("\n")
        file.close()



