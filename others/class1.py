class person:
	num=0
	
	def __init__(self,name,id):
		self.name=name
		self.id=id
		person.num+=1
		print("creating person...")

	def setid(self,id):
		oldid=self.id
		self.id=id;
		print("old id:",oldid,"new id:",self.id)
	
	def getnum(self):
		return person.num

	def showinfo(self):
		print("name:",self.name,"id:",self.id)
		
	def __del__(self):
		person.num-=1
		print("deleleing person:",self.id)
		
class student(person):
	num=0
	
	def __init__(self,name,id):
		person.__init__(self,name,id)
		student.num+=1
		print("creating student...")
		
	def getnum(self):
		return student.num
		
	def __del__(self):
		person.__del__(self)
		student.num-=1
		print("deleleing student:",self.id)


p1=person("Jim",2015)
p2=person("Alice",2016)
print(p1.getnum(),p2.getnum())
p2.showinfo()

print("person.__doc__:", person.__doc__)
print("person.__name__:", person.__name__)
print("person.__module__:", person.__module__)
print("person.__bases__:", person.__bases__)
print("person.__dict__:", person.__dict__)

print(id(p1),id(p2))

del p2

s1=student("Bob",2017)
s1.showinfo()
s1.setid(2018)
print(p1.getnum(),s1.getnum())
del s1
print(person.num,student.num)
#程序结束自动执行del p1