#%%
import math
import matplotlib.pyplot as plt
import numpy as np

# %%
def f(x):
    return 3*x**2 - 4*x +5
# %%
print("when x is 30. function value is :",f(3.0))
# %%
xs = np.arange(-5,5,0.25)
ys = f(xs)
ys
plt.plot(xs, ys)

# %%
#value object
class Value:
    def __init__(self,data , _children=() , _op='' ,label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data ,(self, other) ,'+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad +=  1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data ,(self,other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad +=  self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other,(self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self,other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t , (self,), "tanh")

        def _backward():
            self.grad  +=  ( 1- t**2)  * out.grad

        out._backward = _backward
        return out
    
    
    def exp(self):
        x = self.data
        
        out = Value(math.exp(x) , (self,), "tanh")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        

        self.grad = 1.0
        
        for node in reversed(topo):
            node._backward()

#%%
a = Value(4.0)
2 * a
        

#%%
a = Value(2.0, label= 'a')
b = Value(-3.0 , label='b')
c = Value(10.0 , label = 'c')
e = a * b ; e.label = 'e'
d = e + c ;d.label = 'd'
f = Value(-2.0, label='f')
L = d * f ; L.label = 'L'
L

# %%
print(d._prev)
print(d._op)
# %%
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
        for child in v._prev:
            edges.add((child, v))
            build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{%s | data %.4f |grad %.4f}" % (n.label, n.data ,n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
# %%
draw_dot(L)

# %%
L.grad = 1



# %%
def lol():
        h = 0.01
        a = Value(2.0, label= 'a')
        b = Value(-3.0 , label='b')
        c = Value(10.0 , label = 'c')
        e = a * b ; e.label = 'e'
        d = e + c ;d.label = 'd'
        f = Value(-2.0, label='f')
        L = d * f ; L.label = 'L'
        L1 = L.data
        print(L1)


        a = Value(2.0  , label= 'a')
        b = Value(-3.0 , label='b')
        c = Value(10.0 , label = 'c')
        e = a * b ; e.label = 'e'
        d = e + c ;d.label = 'd'
        f = Value(-2.0, label='f')
        L = d * f ; L.label = 'L'
        L2 = L.data + h
        print(L2)

        print((L2 - L1)/h)

lol()


# %%
#weights
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label = "x2" )

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label = "w2" )

b = Value(6.8813735870195432, label ="b")
x1w1 = x1 * w1 ; x1w1.label='x1w1'
x2w2 = x2 * w2 ; x2w2.label='x2w2'

x1w1x2w2 = x1w1 + x2w2  ; x1w1x2w2.label = "x1w1 + x2w2"
n  = x1w1x2w2 + b ; n.label = 'n'  

e = (2*n).exp()
o = (e-1)/(e+1)

o.label = 'o'
o.backward()

#%%
draw_dot(o)

#%%
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)
topo

for node in reversed(topo):
    node._backward()

#%%
o.backward()

#%%
draw_dot(o)


# %%
o.grad = 1
draw_dot(o)

#%%

o.grad = 1.0
#%%
n._backward()
#%%
o._backward()
#%%
x1w1x2w2._backward()

#%%
x2w2._backward()
x1w1._backward()

# %%
o.grad = 1.0
n.grad = 0.5



# %%
x1w1x2w2.grad = 0.5
b.grad = 0.5

# %%
x1w1.grad = 0.5
x2w2.grad = 0.5

#%%
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad
#%%
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad





# %%
1- (o.data)**2

# %%


#bugss
a = Value(3.0, label="a")
b = a + a ; b.label = "b"
b.backward()
draw_dot(b)

#%%

import torch

x1 = torch.Tensor([2.0]).double()                    ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()                    ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()                   ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()                    ; w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double()      ; b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())

#%%
import random
class Neuron:

    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range (nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w,x)) ,self.b) 
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:

    def __init__(self, nin , nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
    
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
    


#%%
len(n.parameters())
#%%

n = MLP(3, [4,4,1])

#%%
draw_dot(n(x))
# %%


xs = [
    [2.0, 3.0, -1,0],
    [3.0, -1.0 , 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0 , -1.0],]

#desired outputs
ys = [1.0, -1.0 ,-1.0, 1.0]#desired outputs
ypred = [n(x) for x in xs]
ypred

# %%
# mean squared error loss
loss = sum([(yout - ygt)**2 for ygt , yout in zip(ys, ypred)])
loss
# %%
loss.backward()
#%%
for p in n.parameters():
    p.data += -0.01 * p.grad
# %%
draw_dot(loss)
# %%
import math
for k in range(17):
    #forward pass
    ypred = [n(x) for x in xs]  
    loss = sum((yout - ygt)**2 for ygt , yout in zip(ys, ypred))

    #backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()


    #gradient decent
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
#%%
ypred




