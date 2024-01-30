"""
Object holding IP address and pem key.
Used to SSH to the server to download corpus.
"""

class User:

    def __init__(self, key, ip, user, annotators, corPath, annPath, mainAnno):
        self.myKey = key
        self.myIp = ip
        self.myUser = user
        self.myAnno = annotators
        self.corPath = corPath
        self.annPath = annPath
        self.mainAnno = mainAnno

    def get_key(self):
        return self.myKey

    def get_ip(self):
        return self.myIp

    def get_user(self):
        return self.myUser

    def get_anno(self):
        return self.myAnno

    def get_annPath(self):
        return self.annPath

    def get_corPath(self):
        return self.corPath

    def get_main(self):
        return self.mainAnno
