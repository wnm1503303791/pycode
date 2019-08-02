# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:30:19 2019

@author: Administrator
"""

cache={}

def fib(n):
   if n in cache:
       return cache[n]
   else:
       if n<=1:
           res=n
       else:
           #此处要使用递归
           res=fib(n-1)+fib(n-2)
       cache[n]=res
       return res

# def fib(n):
    # if n in cache:  # 输入已经被 cache， 直接输出
        # return cache[n]
    # else:
        # if n <= 1:
            # res = n
        # else:
            # res = fib(n-1) + fib(n-2)
        # cache[n] = res # 将计算结果存其来
        # return res

print(fib(10),len(cache))