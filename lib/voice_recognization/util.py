import os

# check exist model
def isExistModal(filePath):
    if os.path.exists(filePath):
        return True
    return False