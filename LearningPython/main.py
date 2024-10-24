# name = "Chandu"
# name1 = name[::-1]
# print(name1)
# Name = None
# while not Name:
#     Name =input("Enter the name: ")
# print("Hello",Name)
# import time
# for i in range(10,0,-1):
#     print(i)
#     time.sleep(1)
# print("Happy new year ")
# lists >> we can change and update after declaration of a list
#
# food = ["pizza", "hotdog", "burger", "shawarma"]
#
# food.append("icecream")
# food.pop()
# food.remove("pizza")
# food.insert(2,"ice cream")
# food.sort()
# food.clear()
# for x in food:
#     print(x)
#
# tuples >> ordered and unchangeable

# student =("chandu", 22, "2nd year")
# print(student.count(22))
# print(student.index("chandu"))
#
# if "chandu" in student:
#     print("yes , chandu is here ")

# set >> unordered , unindexed and no duplicates

# utensils={"knife","dishes","plates"}
# kitchen={"sink","baker","fridge","knife"}

# utensils.add("spoon")
# utensils.remove("knife")
# # utensils.clear()
# utensils.difference(kitchen)
# utensils.union(kitchen)
# utensils.interference(kitchen)

# Dictionary >> changeable , unordered and they use hashing so faster to access

# capitals = {"USA": "Washington DC", "India": "new Delhi", "Russia": "Moscow"}
#
# print(capitals.get("USA"))
# print(capitals.pop("USA"))
# print(capitals.values())
# print(capitals.items())
# print(capitals.keys())

# def function(first, middle, last):
#     print("Hello " + first + " " + middle + " " + last)
#
#
# function(last="Chandu", middle="Sai", first="Yarrapothu")#
#
# Scope

# python follows the legb rule
# first it sees the local variables , then enclosed variables,then global variables and then built_in variables if any

# *args >> that pack the arguments into a tuple
# def add(*args):
#     sum = 0
#     for i in args:
#         sum = sum + i
#     return sum
#
#
# print(add(1, 2, 3, 4))
# print(add(1, 2, 3, 4, 5, 6, 7))

# def add(*stuff):
#     stuff=list(stuff)
#     stuff[2]=0
#     sum = 0
#     for i in stuff:
#         sum = sum + i
#     return sum
#
#
# print(add(1, 2, 3, 4))
# print(add(1, 2, 3, 4, 5, 6, 7))

# **kwargs >> parameter that packs the arguments into a dictionary

# def hello(**kwargs):
#     print("hello"+kwargs["first"]+" "+kwargs["last"])
#
#
# hello(first=" chandu", middle="sai", last=" Yarrapothu")
#
# def hello(**kwargs):
#     print("Hello",end=" ")
#     for key , value in kwargs.items():
#         print(value,end=" ")
#
#
# hello(title ="Mr",first="Yarrapothu",middle="Sai",last="Chandu")


# format()>>
# animal ="cow"
# item ="moon"
#
# text ="The {} jumped over the {}"
#
# print(text.format(animal,item))
# print("The {1} jumped over the {1}".format(animal,item))
# print("The {animal} jumped over the {item}".format(animal="cow",item="moon"))

# Padding
# name = "bro"
# print("Hello good morning {}".format(name))
# print("Hello good morning {:10} you are welcome".format(name))
# print("Hello good morning {:<10} you are welcome".format(name))
# print("Hello good morning {:>10} you are welcome".format(name))
# print("Hello good morning {:^10} you are welcome".format(name))


# numbers
# num1 = 1.1414
# num = 120000
# print("square root of 2 is {:.2f}".format(num1))
# print("square root of 2 is {:b}".format(num))
# print("square root of 2 is {:o}".format(num))
# print("square root of 2 is {:,}".format(num))
# print("square root of 2 is {:X}".format(num))
# print("square root of 2 is {:x}".format(num))
# print("square root of 2 is {:E}".format(num))

# Random
# import random
#
# x = random.random()
# y = random.randint(1,10)
# print(x)
# print(y)
# myList = [1, 2, 3, 5, 74]
# print(random.choice(myList))
#
# cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, "j", "q", "k", "a"]
# random.shuffle(cards)
# print(cards)

# Exception handling

# try:
#     a = int(input("Enter the number: "))
#     b = int(input("Enter the number to divide the prev: "))
#     result = a / b
# except ValueError as e:
#     print(e)
#     print("plz Enter the integer value")
# except ZeroDivisionError as e:
#     print(e)
#     print("You cant divide with zero! idiot !!")
# except Exception as e :
#     print(e)
#     print("Something went wrong")
# else:
#     print(result)
# finally:
#     print("This always executed")

# files
# import os
# path ="C:\\Users\\Dell\\OneDrive\\Desktop\\test.txt"
#
# if os.path.isdir(path):
#     print("File existed")
# text = "Hello Chandu !!! Good Morning !!:)"
# with open("text.txt", "w") as file:
#     file.write(text)
#
# import shutil
#
# shutil.copyfile("text.txt", "copy.txt")


# while True :
#     print("who are you? ")
#     name=input()
#     print("Hi "+name)
#     re_attempt=input("what to re_attempt ?yes/no  ").lower()
#     if re_attempt == "no":
#         break
#
# class Car:
#     wheels = 4  # class variable
#
#     def __init__(self, name, model, year):
#         self.name = name  # instance variable
#         self.model = model  # instance variable
#         self.year = year  # instance variable
#
#     def start(self):
#         print("The", self.model, "is started")
#
#     def stop(self):
#         print("The", self.model, "is Stopped")
#
#
# car1 = Car("Ford", "mustang", 2022)
# print(car1.model)
# print(car1.name)
# print(car1.year)
# car1.start()
# car1.stop()
# print(car1.wheels)
# car1.wheels = 1  # you can change the class variable for a specific object that doesn't affect the default value of our class variable
# Car.wheels = 2  # changes the default variable wheels = 4 to 2


# Inheritance
#
#
# class Animal:
#
#     def eat(self):
#         print("Animal is eating now")
#
#     def sleep(self):
#         print("Animal is sleeping now")
#
#
# class fish(Animal):
#     pass
#
#
# fish1 = fish()
# fish1.eat()


# multiple inheritance
# multi level inheritance
# method overriding
# method chaining

# class car:
#     def drive(self):
#         print("you are driving the car ")
#         return self  # you have to return something in order to do method chaining
#
#     def stop(self):
#         print("you stopped the car ")
#         return self
#
#
# car1 = car()
# car1.stop().drive()
# car1.stop()\
#     .drive()
# # \ is the line continuation character

# super() = is used to access the parent class function in a child class
#
# class rectangle:
#     def __init__(self, length, width):
#         self.length = length
#         self.width = width
#
#
# class square(rectangle):
#     def __init__(self, length, width):
#         super().__init__(length, width)
#
#
# class cube(rectangle):
#     def __init__(self, length, width, height):
#         super().__init__(length, width)
#         self.height = height

# Abstract class = it is a class that contains two or more abstract methods
# Abstract method = a method that has a declaration but does not have an implementation
# it compels the user to override the abstract methods in a child class

# from abc import ABC, abstractmethod
#
#
# class animal(ABC):
#     @abstractmethod
#     def wake(self):
#         pass
#
#     @abstractmethod
#     def sleep(self):
#         pass
#
#
# class bird(animal):
#
#     def wake(self):
#         print("The bird is awake")
#
#     def sleep(self):
#         print("The bird is sleep now")
#
# bird1 = bird()
# bird1.wake()





