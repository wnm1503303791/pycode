class person:
	num=0
	
	def __init__(self,name,id):
		self.name=name
		self.id=id
		num+=1;

	def setid(self,id):
		self.id=id;
	
	def getnum(self):
		return num

	def showinfo(self):
		print("name:",self.name,"id:",self.id)


p1=person("Jim",2015)
p2=person("Alice",2016)
print(p1.getnum(),p2.getnum())
p2.showinfo()
