r = BinaryTree('a')
print(r.getRootVal())  #a
print(r.getLeftChild())#None
r.insertLeft('b')
print(r.getLeftChild())#<__main__.BinaryTree object at 0x0000000005FFC780>
print(r.getLeftChild().getRootVal())#b
r.insertRight('c')
print(r.getRightChild())#<__main__.BinaryTree object at 0x0000000005DA2898>
print(r.getRightChild().getRootVal())#c
r.getRightChild().setRootVal('hello')
print(r.getRightChild().getRootVal())

from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree

def buildParseTree(formula):
    formulalist = formula.split()  #['('，'3'，'+'，'('，'4'，'*'，'5'，')'，')']
    rootStack = Stack()  #用于跟踪父节点
    Tree = BinaryTree('') #创建一个二叉树
    rootStack.push(Tree)
    currentTree = Tree
    for i in formulalist:
        if i == '(':
            currentTree.insertLeft('')
            rootStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in ['+','-','*','/',')']:
            currentTree.setRootVal(int(i))
            parent = rootStack.pop()
            cerrentTree = parent
        elif i in ['+','-','*','/']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            rootStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = rootStack.pop()
        else:
            raise ValueError  #raise的参数必须是一个异常实例或者一个异常类：ValueError无效参数
    return Tree 
   
pt = buildParseTree("( ( 10 + 5 ) * 3 )")
pt.postorder() #5+3*
   
