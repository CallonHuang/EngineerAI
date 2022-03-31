## 基本数据类型


```python
a = 123
a = int("456")
b = 123.
b = float("456")
c = "abc"
c = 'def'
tmp = c.find("f")
print(tmp)

array = c.split("e")
print(array)

d = list()
d = [3, 3, 2, 2, 1]
print(d[:3])
print(d[-3:-1])

e = dict()
e = {'a' : 1, 'b' : 2, 'c' : 3}
f = set(d)
g = tuple(d)
g = (1, 2)
print(g)

h = "a is %d" % a
print(h)

h = "a is {}".format(a)
print(h)

h = f"a is {a}, b is {b}, c is {c}, d is {d}, e is {e}, f is {f}, g is {g}"
print(h)

print(r"c://a\b\c//")

t = (1)
print(type(t), t)

t = (1,)
print(type(t), t)
```

    2
    ['d', 'f']
    [3, 3, 2]
    [2, 2]
    (1, 2)
    a is 456
    a is 456
    a is 456, b is 456.0, c is def, d is [3, 3, 2, 2, 1], e is {'a': 1, 'b': 2, 'c': 3}, f is {1, 2, 3}, g is (1, 2)
    c://a\b\c//
    <class 'int'> 1
    <class 'tuple'> (1,)
    

## 基本语法


```python
b = 1
if b == 1 or b == 3:
    print("abc")
elif b == 2:
    print("abcd")
else:
    print("abcdef")

while a:
    print("abc")
    break

#for var in list/tuple/支持iter的类
for i in range(2):
    print(i)
print(list(range(10)))

var = [3, 2, 5]
for index, elem in enumerate(var):
    print(f"{index}, {elem}")
    
#从这里可以看出，上面的for循环就是tuple拆包的过程
for index in enumerate(var):
    print(f"{index}")

a = [item + 2 for item in var if item > 2]
print(a)

# a = isok ? 1 : 2
isok = True
a = 1 if isok else 2
print(a)

```

    abc
    abc
    0
    1
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    0, 3
    1, 2
    2, 5
    (0, 3)
    (1, 2)
    (2, 5)
    [5, 7]
    1
    

## 函数


```python
# 实现在其他地方
def func():
    pass

def func1(a, b, c):
    global d
    print(a)
    print(d)
    d = 456

def func2(a, b, c):
    print(f"a = {a}, b = {b}, c = {c}")

def final(w):
    print(f"w = {w}")
    
def func3(a, b, *args0, **args1):
    print("args0", args0)
    print("args1", args1)
    final(**args1)
    
def func4():
    return 1, 2
    
d = 123
func1(1, 2, 3)
print(d)
func2(4, 5, 6)
func2(*[4, 5, 6])
func2(**{'a' : 4, 'b' : 5, 'c' : 6})
func3(1, 2, 33, 55, w=333)
a, b = func4()
print(f"a is {a}, b is {b}")
```

    1
    123
    456
    a = 4, b = 5, c = 6
    a = 4, b = 5, c = 6
    a = 4, b = 5, c = 6
    args0 (33, 55)
    args1 {'w': 333}
    w = 333
    a is 1, b is 2
    

## 类


```python
import os
class Person:
    def __init__(self, name):
        print(f"construct: name {name}")
        self.name = name
    
    def call(self):
        print("person {} call".format(self.name))
        
    def __getitem__(self, item):
        return "abc" + item

person = Person("hkl")
person.age = 30
print(person.age)

person.call()

print(person["  ccc"])

print(__name__)

if __name__ == "__main__":
    print("hello")
    print(os.listdir("."))
    #print(os.listdir.__doc__)

```

    construct: name hkl
    30
    person hkl call
    abc  ccc
    __main__
    hello
    ['.ipynb_checkpoints', 'basic_python.ipynb']
    Return a list containing the names of the files in the directory.
    
    path can be specified as either str, bytes, or a path-like object.  If path is bytes,
      the filenames returned will also be bytes; in all other circumstances
      the filenames returned will be str.
    If path is None, uses the path='.'.
    On some platforms, path may also be specified as an open file descriptor;\
      the file descriptor must refer to a directory.
      If this functionality is unavailable, using it raises NotImplementedError.
    
    The list is in arbitrary order.  It does not include the special
    entries '.' and '..' even if they are present in the directory.
    

## 冷门操作


```python
class MyWith:
    def __enter__(self):
        print("enter MyWith")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit MyWith")

with MyWith() as my:
    print("in with")
    print(my)

#with open("a.txt", "w") as f:
#    f.write("abc")

a = [1, 1, 2, 3, 5, 5]
b = [3, 5, 9, 10]
print(set(a))
print(2 in set(a))
print(set(a) | set(b))
print(set(a) & set(b))
print(set(a) ^ set(b))

a = [1, 2, 3]
b = ["a", "b", "c"]
print(list(zip(a,b)))

a = [1, 3, 2, 5]
a.sort()
print(a)
```

    enter MyWith
    in with
    None
    exit MyWith
    {1, 2, 3, 5}
    True
    {1, 2, 3, 5, 9, 10}
    {3, 5}
    {1, 2, 9, 10}
    [(1, 'a'), (2, 'b'), (3, 'c')]
    [1, 2, 3, 5]
    

## numpy 简单操作


```python
import numpy as np
import cv2

a = np.zeros((100, 100, 3), np.uint8)
a[:, :, 1] = 255

cv2.imshow("image", a)
cv2.waitKey()


```




    -1


