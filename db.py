import sqlite3 as sql

class DB:

    def __init__(self):
        self.db = sql.connect('mlModel.db')
        self.con = self.db.cursor()
       
        try:
            self.con.execute('create table Params (model text)')
        except:
            pass
    
    def addInDb(self, model):
        self.con.execute('delete from Params')
        self.con.execute(f'insert into Params values (?)',(model,))
        self.db.commit()
    
    def getModel(self):
        try:
            self.con.execute('select model from Params')
            return self.con.fetchone()
        except:
            pass