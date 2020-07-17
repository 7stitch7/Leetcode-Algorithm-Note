## 高质量的代码

- 需要完整的解决问题，考虑所有的corner case

## 面试题3 数组中重复的数字

#### 题目一描述

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

1. 排序 O(nlogn)

2. 哈希表 时间复杂度O(n)，空间复杂度O(n)

   ```python
   # -*- coding:utf-8 -*-
   class Solution:
       # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
       # 函数返回True/False
       def duplicate(self, numbers, duplication):
           # write code here
           num = {}
           for i in numbers:
               try:
                   num[i]=num[i]+1
                   duplication[0] = i
                   print(duplication[0])
                   return True
               except:
                   num[i]=1
           return False
   ```

   

3. 由于所有数字都在0到n-1的范围内，如果数组排序后，数值应与下标相等。第一个重复的数，一定会遭遇，第一个重复数排序后与下标相同，剩下的与下标均不同。所以，可以依次扫描数组，若下标 i 与数值 a 不同，则交换a 与 下标 a 上的数，直到找到重复数。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        if len(numbers)<=0:
            return False
        for i in range(len(numbers)):
            if numbers[i]<0 or numbers[i]> len(numbers)-1:
                return False
            while(numbers[i]!=i):
                if numbers[numbers[i]]== numbers[i]:
                    duplication[0]=numbers[i]
                    return True
                a = numbers[i]
                numbers[i]=numbers[a]
                numbers[a] = a
        return False
```



#### 题目二描述

在一个长度为n+1的数组里的所有数字都在1～n的范围内。 数组中至少有一个数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

**解法：**

和前面的方法相似，但不能修改数组。于是我们可以考虑用二分法的思想解决这个问题，重复的数字一定出现在前半段，或后半段，那么只要持续将含有重复数字的区间二分，就可以找到重复数。但时间复杂度会上升到O(nlogn)，该算法不能找到全部的重复数字，可能会出现两边都含有重复数字，但在算法运行中只会选择一边。 



### 面试题4 二维数组中的查找

#### 题目描述

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**解法：**

首先选取数组中右上角或者左下角的数进行比较，如果需要比较的数比选取的数大，则排除行，反之则排除列。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        length = len(array)
        if length == 0:
            return False
        width = len(array[0])
        row = 0
        col = width-1
        while row<length and col>=0:
            if array[row][col]==target:
                return True
            if array[row][col]>target:
                col-=1
            if array[row][col]<target:
                row+=1
```



### 面试题5 替换空格

#### 题目描述

请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

**解法：**

如果从前往后扫描，当我们替换空格时，需要将后面的字符串向后移。这样时间复杂度会升至O(n^2)。那么如果先知道一共有几个空格，从后往前扫描就不需要反复计算使用空间了。准备两个指针P1和P2, P1指向原始字符串结尾，P2指向替换后字符串结尾。向前移动P1,逐个将它指向的字符复制到P2指向的位置, 直到P1,P2重合。

<img src="/Users/fuqinwei/Library/Application Support/typora-user-images/image-20200714013516848.png" alt="image-20200714013516848" style="zoom:50%;" />

```python
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        string = s.split(' ')
        return "%20".join(string)
```



(Python 莫得灵魂)



### 面试题6 从尾到头打印链表

#### 题目描述

输入一个链表，按链表从尾到头的顺序返回一个ArrayList。

解法：

1. 从头到尾遍历链表入栈，然后依次遍历栈，打印元素

```python
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        stack = []
        if listNode is None:
            return stack
        while listNode:
            stack.append(listNode.val)
            print(listNode.val)
            listNode = listNode.next
        return stack[::-1]
```



2. 递归：每访问一个节点的时候，先递归输出后面的节点，再输出它本身

```python
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        stack = []
        if listNode is None:
            return stack
        def printlist(listNode):
            if listNode.next:
                return printlist(listNode.next)
```

（待debug）



### 面试题7 重建二叉树

#### 数据结构：树

前序遍历：root-left-right

中序遍历：left-root-right

后序遍历：left-right-root

**二叉树特例：堆，红黑树**

红黑树：

节点定义成红，黑两种颜色，root-external node的最长路径不能超过最短的两倍

#### 题目描述

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

解法：

前序的第一个元素是根结点的值，在中序中找到该值，中序中该值的左边的元素是根结点的左子树，右边是右子树，然后递归的处理左边和右边

<img src="/Users/fuqinwei/Library/Application Support/typora-user-images/image-20200716221908490.png" alt="image-20200716221908490" style="zoom:50%;" />

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        if len(pre) == 1:
            return TreeNode(pre[0])
        else:
            flag = TreeNode(pre[0])
            flag.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],tin[:tin.index(pre[0])])
            flag.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],tin[tin.index(pre[0])+1:] )
        return flag
```



### 面试题8 二叉树的下一个节点

#### 题目描述

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

**解法：**

如果节点有右子树，那么右子树的最坐节点就是下一个节点。

如果节点没有右子树

- 情况1: 它是父节点的左子节点，那么它的父节点就是下一个节点
- 情况2: 它是父节点的右节点，那么需要一直遍历父节点直到遇到作为左节点的父节点，那么这个节点的父节点就是下一个节点

```python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if pNode==None:
            return None
        if pNode.right!=None:
            left1 = pNode.right
            while(left1.left!=None):
                left1 = left1.left
            return left1
        elif pNode.next!=None:
            current = pNode
            parent = pNode.next
            while(parent!=None and current == current.next.right ):
                current = parent
                parent = parent.next
            return parent



```



### 面试题9：用两个栈实现队列

#### 题目描述

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

**解法：**

栈1实现push, 栈2实现pop. 当栈2为空时，将栈1的元素全部压入栈2

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1=[]
        self.stack2=[]
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        # return xx
        if self.stack2==[]:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()
        return self.stack2.pop()
```



问题二 两个队列实现栈： 方法相同



___

### 递归和循环

### 面试题10: 斐波那契数列

#### 题目描述

大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。

n<=39

gg解法

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n<=0:
            return 0
        elif n==1:
            return 1
        return self.Fibonacci(n-1)+self.Fibonacci(n-2)
```

dp 解法：

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n==0:
            return 0
        elif n==1:
            return 1
        bc0 = 0
        bc1 = 1
        i = 2
        while(i<=n):
            fn = bc0+bc1
            i+=1
            bc0=bc1
            bc1=fn
        return fn

```

题目二： 青蛙跳台阶问题

#### 题目描述

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number==0:
            return 0
        if number==1:
            return 1
        if number==2:
            return 2
        bc0 = 0
        bc1 = 1
        bc2 = 2
        i = 3
        while(i<=number):
            fn = bc2+bc1
            i+=1
            bc1=bc2
            bc2=fn
        return fn
            
```



题目三： 放瓷砖



___

### 查找和排序

### 面试题11：旋转数组的最小数字

