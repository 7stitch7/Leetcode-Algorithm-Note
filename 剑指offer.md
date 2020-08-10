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

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number==0:
            return 0
        if number==1:
            return 1
        return 2**(number-1)
```



题目三： 放瓷砖



___

### 查找和排序

### 面试题11：旋转数组的最小数字

#### 题目描述

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

解法：

寻找最小数字最简单的方法自然是以O(n)的复杂度遍历，但没有利用好旋转数组的特点。可以观察到，旋转数组的第一个元素总是比最后一个大，从中间选择一个元素，如果比第一个元素小，那么一定在正常排序的数组里，反之则在旋转的数组里，那么就可以利用二分法，用O(logn)找到最小值

corner case：输入未旋转的数组将不符合判断条件,考虑[1, 0, 0, 1]这种数据，只能顺序查找

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        start = 0
        end = len(rotateArray)-1
        if end == -1:
            return False
        if rotateArray[start]<rotateArray[end]:
            return rotateArray[0]
        while(start!=end):
            if end-start==1:
                return rotateArray[end]
            mid = (start+end)/2
            if rotateArray[start]==rotateArray[mid]==rotateArray[end]:
              return min(rotateArray)
            if rotateArray[start]<=rotateArray[mid]:
                start = mid
            if rotateArray[end]>=rotateArray[mid]:
                end = mid
        return rotateArray[mid]
```



### 面试题12: 矩阵中的路径

#### 题目描述

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如 \begin{bmatrix} a & b & c &e \\ s & f & c & s \\ a & d & e& e\\ \end{bmatrix}\quad⎣⎡*a**s**a**b**f**d**c**c**e**e**s**e*⎦⎤ 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

**回溯法：**

回溯法（探索与回溯法）是一种选优搜索法，又称为试探法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法，而满足回溯[条件](https://baike.baidu.com/item/条件/1783021)的某个[状态](https://baike.baidu.com/item/状态/33204)的点称为“回溯点”。在探寻下一步是否可以走通时，可以使用递归。

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if matrix==None and rows<=0 and cols<=0 and path == None:
            return False
        visited = [0]*(rows*cols)
        pathlength = 0
        for row in range(rows):
            for col in range(cols):
                if self.hasPathCore(matrix, rows, cols, row, col, path,visited, pathlength):
                    return True
        return False
    def hasPathCore(self,matrix, rows, cols, row, col, path ,visited, pathlength):
        if len(path) == pathlength:
            return True
        hasPath = False
        if row>=0 and row<rows and col>=0 and col<cols and matrix[row*cols+col]==path[pathlength] and visited[row*cols+col]==0:
            pathlength +=1
            visited[row*cols+col]=1
            hasPath = self.hasPathCore(matrix, rows, cols, row+1, col, path,visited, pathlength) or \
                        self.hasPathCore(matrix, rows, cols, row, col+1, path,visited, pathlength) or \
                        self.hasPathCore(matrix, rows, cols, row-1, col, path,visited, pathlength) or \
                        self.hasPathCore(matrix, rows, cols, row, col-1, path,visited, pathlength)
            if not hasPath:
                pathlength-=1
                visited[row*cols+col]=0
        return hasPath


```



### 面试题13: 机器人的运动范围

#### 题目描述

地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

```python
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        markmatrix = [False] * (rows * cols)
        count = self.GetNum(threshold, rows, cols, 0, 0, markmatrix)
        return count

    def GetNum(self, threshold, rows, cols, row, col, markmatrix):
        count = 0

        if self.GetSum(threshold, rows, cols, row, col, markmatrix):
            markmatrix[row * cols + col] = True
            count = 1 + self.GetNum(threshold, rows, cols, row - 1, col, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row, col - 1, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row + 1, col, markmatrix) + \
                    self.GetNum(threshold, rows, cols, row, col + 1, markmatrix)
        return count

    def GetSum(self, threshold, rows, cols, row, col, markmatrix):
        if row >= 0 and row < rows and col >= 0 and col < cols and self.getDigit(row) + self.getDigit(
                col) <= threshold and not markmatrix[row * cols + col]:
            return True
        return False

    def getDigit(self, number):
        sumNum = 0
        while number > 0:
            sumNum += number % 10
            number = number // 10
        return sumNum
```





___

### 动态规划与贪心算法

从上到下分析问题，从下到上求解问题

### 面试题14: 剪绳子

#### 题目描述

给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1，m<=n），每段绳子的长度记为k[1],...,k[m]。请问k[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**解法一：** 动态规划 f(n) = max(f(i)*f(n-i))

```python
# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        # write code here
        if number<2:
            return False
        if number==2:
            return 1
        if number==3:
            return 2
        product = [0]*(number+1)
        product[0]=0
        product[1]=1
        product[2]=2
        product[3]=3
        max_product = 0
        for i in range(4,number+1):
            max_projuct = 0
            for j in range(1,i//2+1):
                pro = product[j]*product[i-j]
                if pro > max_product:
                    max_product = pro
                product[i]=max_product
        return product[number]
```

PS: 当剪完后的数字小于4时，它们继续剪下去的结果会比它本身小，比如3剪完后的最大乘积是2，但3本身比2大，所以max_product 只保留3

**解法二：** 贪心算法

当n>= 5时，尽可能剪长度为3； 当剩下长度为4时，剪成2

```python
# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        # write code here
        if number<2:
            return False
        if number==2:
            return 1
        if number==3:
            return 2
        number3 = number//3
        if number-3*number3==1:
            number3-=1
        number2=(number-3*number3)//2
        return 3**number3*2**number2


```

___

### 位运算

位运算：除法的效率远低于位运算

### 面试题15: 二进制中1的个数

#### 题目描述

输入一个整数，输出该数32位二进制表示中1的个数。其中负数用补码表示。

**解法：**

**把一个整数减1再与原来的数做位运算，得到的结果相当于把该整数二进制中最右边的1变成了0**

如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，原来在1

后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。

举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，

而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们

再把原来的整数和减去1之后的结果做与运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是

说，把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少

次这样的操作。

但是负数使用补码表示的，对于负数，最高位为1，而负数在计算机是以补码存在的，往右移，符号位不变，符号位1往右移，

最终可能会出现全1的情况，导致死循环。与0xffffffff相与，就可以消除负数的影响



```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        count = 0
        if n<0:
            n = n & 0xffffffff
        while n:
            count += 1
            n = n & (n-1)
        return count
```



**相关题目**

1. 判断一个整数是不是2的整数次方
   - 2的整数次方意味着二进制中只有一个1
2. 输入整数m和n，需要改变m的二进制中的几位才能变为n
   - 1. 求两个数的异或 2. 统计异或结果中1的个数



___

## 高质量的代码

### 代码的完整性

___

### 面试题16: 数值的整数次方

#### 题目描述

给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

保证base和exponent不同时为0

**注意**: 要考虑exponent为0和为负数的情况

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        invalid = False
        if base == 0 and exponent <0:
            invalid = True
            return False
        if exponent<0:
            result = 1/self.PowerwithUnsignedExponent(base,-exponent)
            return result
        if exponent>=0:
            return self.PowerwithUnsignedExponent(base,exponent)
    
    def PowerwithUnsignedExponent(self, base, exponent):
        if exponent==0:
            return 1
        if exponent==1:
            return base
        result=self.PowerwithUnsignedExponent(base, exponent//2)
        result*=result
        if exponent&0x1==1:
            result*=base
        return result
```

最优解：考虑到base^(n/2)*base^(n/2) = base^2, 所以我们可以递归地求base^n, 而不是使用循环



### 面试题17: 打印从1到最大的n位数

陷阱：必须考虑n的数量级

**解法：** 在字符串上模拟数字加法的解法

缺点：代码很长

**优化：** 用递归实现全排列，来替代字符串加法



### 面试题18: 删除链表中的节点

**题目一：**

在O(1)时间内删除一个已知的节点

**解法：**复制后一个节点的值给要删除的节点，然后删除后一个节点

PS: 当需要删除的节点是尾节点时，没有下一个节点，只能遍历找到前序节点

```python
def delete_node(link, node):
    if node == link:  # 只有一个结点
        del node
    if node.next is None:  # node是尾结点
        while link:
            if link.next == node:
                link.next = None
            link = link.next
    else:
        node.val = node.next.val
        n_node = node.next
        node.next = n_node.next
        del n_node
```



**题目二：**

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

解法：从头遍历，寻找重复的节点，当遇到重复节点时，可以看作头节点重复的子链表，可以递归地删除子链表中的重复节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        head1 = pHead.next
        if head1.val != pHead.val:
            pHead.next = self.deleteDuplication(pHead.next)
        else:
            while pHead.val == head1.val and head1.next is not None:
                head1 = head1.next
            if head1.val != pHead.val:
                pHead = self.deleteDuplication(head1)
            else:
                return None
        return pHead
```



### 面试题19: 正则表达式匹配

#### 题目描述

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

解法：我们需要分析匹配将会有几个状态：

 - 当第二个字符不是‘*’时，
   	- 状态1: 如果第一个字符匹配（包括与"."匹配），那么都向后移动一个字符
 - 当第二个字符是'*'时，
   	- 状态2: 如果第一个字符不匹配，那么‘*’可以使pattern中第一个字符出现0次，pattern向后移动两个字符
      	- 状态3: 如果第一个字符匹配，那么‘*’可以使pattern中第一个字符只出现1次，pattern向后移动两个字符，字符串向后移动一个字符
   	- 状态4: 如果第一个字符匹配，那么‘*’可以使pattern中第一个字符只出现n次，字符串向后移动一个字符

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        if (len(s) == 0 and len(pattern) == 0):
            return True
        if (len(s) > 0 and len(pattern) == 0):
            return False
        if (len(pattern) > 1 and pattern[1] == '*'):
            if (len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.')):
                return (self.match(s, pattern[2:]) or self.match(s[1:], pattern[2:]) or self.match(s[1:], pattern))
            else:
                return self.match(s, pattern[2:])
        if (len(s) > 0 and (pattern[0] == '.' or pattern[0] == s[0])):
            return self.match(s[1:], pattern[1:])
        return False
```



### 面试题20: 表示数值的字符串

#### 题目描述

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

**解法：** 分开讨论整数部分A, 小数部分B，指数部分C. 

- A, C都是可能以‘+’‘-’开头的0～9的数位串
- B只可以是以'.'开头的0～9的数位串



### 面试题21: 调整数组顺序使奇数位于偶数前

#### 题目描述

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分。

**解法：** 两个指针从首尾开始扫描，如果发现第一个指针指向偶数，第二个指向奇数，则交换它们的位置

PS: 为了更好的扩展性，可以把判断奇偶部分抽取出来

```python
def reorder(nums, func):
    left, right = 0, len(nums) - 1
    while left < right:
        while not func(nums[left]):
            left += 1
        while func(nums[right]):
            right -= 1
        if left < right:
            nums[left], nums[right] = nums[right], nums[left]


def is_even(num):
    return (num & 1) == 0
```



### 3.4 代码的鲁棒性

___

### 面试题22: 链表中倒数第k个节点

#### 题目描述

输入一个链表，输出该链表中倒数第k个结点。

**解法：** 快慢指针，中间间隔k-1

Corner case: 

1. 空链表
2. k<=0
3. k>链表长度



```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self,head,k):
        # write code here
        FirstNode = head
        BindNode = head
        if not head:
            return None
        if k<=0:
            return None
        for i in range(k-1):
            if BindNode.next:
                BindNode = BindNode.next
            else:
                return None
        while(BindNode.next!=None):
            FirstNode = FirstNode.next
            BindNode = BindNode.next
        return FirstNode
```





### 面试题23: 链表中环的入口节点

#### 题目描述

给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

**解法：**

 1. 确定是否有环： 一个指针一次走一步，另一个一次走两步，如果快指针追得上慢指针，则有环

			2. 确定入环口：设置距离为n的快慢指针，当n为入口节点距离时，慢指针会与快指针在入口节点正好相遇
      			3. 利用步骤一确定环中节点的个数：快慢指针每次相差1，当它们相遇时，相差的间隔n，就相当于慢指针走的路程

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        if pHead==None or pHead.next==None or pHead.next.next==None:
            return None
        if pHead==pHead.next:
            return pHead
        slow = pHead
        fast = pHead
        while slow!=None and fast!=None:
            slow = slow.next
            fast = fast.next.next
            if slow==fast:
                break
        fast=pHead #当找到第一个相遇点时，间隔为k，只要慢指针往前走k步，每次一步，就会和从表头开始的新指针相遇
        while slow!=fast:
            slow=slow.next
            fast=fast.next
        return fast
```





### 面试题24: 反转链表

#### 题目描述

输入一个链表，反转链表后，输出新链表的表头。

思路：需要考虑空链表，只有一个结点的链表，把前一个节点存下来

```python
def reverse_link(head):
    if not head or not head.next:
        return head
    then = head.next
    head.next = None
    last = then.next
    while then:
        then.next = head
        head = then
        then = last
        if then:
            last = then.next
    return head
```



### 面试题25: 合并两个排序的链表

#### 题目描述

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

注意：空链表的输入



```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if pHead1==None:
            return pHead2
        if pHead2==None:
            return pHead1
        mergenode = None
        if pHead1.val<pHead2.val:
            mergenode = pHead1
            mergenode.next = self.Merge(pHead1.next,pHead2)
        else:
            mergenode = pHead2
            mergenode.next = self.Merge(pHead1,pHead2.next)
        return mergenode

```



### 面试题26: 树的子结构

#### 题目描述

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

解法：用递归先找到根节点相同的点，如果找到，检查子节点是否一致，子节点以递归的方式，检查左右节点是否一致

注意：需要考虑指针为空带来的结果



```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        result = False
        if pRoot1!=None and pRoot2!=None:
            if pRoot1.val==pRoot2.val:
                result= self.DoseTree1HasTree2(pRoot1,pRoot2)
            if result==False:
                result= self.HasSubtree(pRoot1.left,pRoot2)
            if result==False:
                result= self.HasSubtree(pRoot1.right,pRoot2)
        return result
    def DoseTree1HasTree2(self,pRoot1,pRoot2):
        if pRoot2==None:
            return True
        if pRoot1==None:
            return False
        if pRoot1.val!=pRoot2.val:
            return False
        return self.DoseTree1HasTree2(pRoot1.left,pRoot2.left) and self.DoseTree1HasTree2(pRoot1.right,pRoot2.right)

```



## 第4章 解决面试题的思路

### 面试题27: 二叉树的镜像

#### 题目描述

操作给定的二叉树，将其变换为源二叉树的镜像。

解法：前序遍历，并交换左右子节点

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if root != None:
            root.left,root.right = root.right,root.left
            self.Mirror(root.left)
            self.Mirror(root.right)
```



### 面试题28: 对称的二叉树

#### 题目描述

请实现一个函数，用来判断一棵二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

解法：设计一种root-right-left的遍历方法，如果是对称的，则与前序遍历的结果一样

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        if pRoot.left and not pRoot.right:
            return False
        if not pRoot.left and pRoot.right:
            return False
        return self.is_same(pRoot.left,pRoot.right)
    def is_same(self,p1,p2):
        if not p1 and not p2:
            return True
        if (p1 and p2) and p1.val==p2.val:
            return self.is_same(p1.left,p2.right) and self.is_same(p1.right,p2.left)
        return False
```



### 面试题29: 顺时针打印矩阵

#### 题目描述

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

解法：每一圈的开始位置总是坐上角元素[0, 0], [1, 1]...，圈最后结束的位置总是在中间 cols>start*2 and rows>start*2



```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        if len(matrix)==0 or len(matrix[0])==0:
            return None
        rows = len(matrix)
        cols = len(matrix[0])
        start = 0
        ret = []
        while(cols>start*2 and rows>start*2):
            self.PrintMatrixInCircle(matrix,rows,cols,start,ret)
            start+=1
        return ret
    def PrintMatrixInCircle(self,matrix,rows,cols,start,ret):
        endX=cols-1-start
        endY=rows-1-start
        i=start
        # left->right
        while(i<=endX):
            ret.append(matrix[start][i])
            i+=1
        # top->bottom
        if start<endY:
            i = start+1
            while(i<=endY):
                ret.append(matrix[i][endX])
                i+=1
        # right->left
        if start<endX and start<endY:
            i = endX-1
            while(i>=start):
                ret.append(matrix[endY][i])
                i-=1
        # bottom->top
        if start<endX and start<endY-1:
            i=endY-1
            while(i>=start+1):
                ret.append(matrix[i][start])
                i-=1
                
```





### 面试题30: 包含min函数的栈

#### 题目描述

定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    def push(self, node):
        # write code here
        self.stack.append(node)
        if not self.min_stack or node <= self.min_stack[-1]:
            self.min_stack.append(node)
    def pop(self):
        # write code here
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()
    def top(self):
        # write code here
        return self.stack[-1]
    def min(self):
        # write code here
        return self.min_stack[-1]
```





### 面试题31: 栈道压入、弹出序列

#### 题目描述

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

解法：使用一个辅助栈，逐一压入pushV,并与popV依次比较，当下一个弹出的数字在栈顶时，直接弹出，不在栈顶时，继续压入直到栈顶数字与popV相等，如果全部压入后仍然没有相等的栈顶数字，则return False

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if not pushV or len(pushV) != len(popV):
            return False
        stack = []
        for i in pushV:
            stack.append(i)
            while len(stack) and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
        if len(stack):
            return False
        return True
```



### 面试题32: 从上到下打印二叉树

#### 题目描述

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

解法：每打印一个节点，将它的子节点放到队列的最后面

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None                              
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        l=[]
        if not root:
            return []
        q=[root]
        while len(q):
            t=q.pop(0)
            l.append(t.val)
            if t.left:
                q.append(t.left)
            if t.right:
                q.append(t.right)
        return l       
```





### 问题二: 之字形打印二叉树

#### 题目描述

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

解法：使用两个栈存放子节点

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        result = []
        stack = [pRoot]
        level = 1
         
        while len(stack) > 0:
            ns = []
            rs = []
            for i in stack:
                if i != None:
                    rs.append(i.val)
                    if i.left != None:
                        ns.append(i.left)
                    if i.right != None:
                        ns.append(i.right)
                         
            if level % 2 == 0:
                rs.reverse()
            if len(rs) > 0:
                result.append(rs)
                 
            stack = ns
            level += 1
        return result
```





### 面试题33: 二叉搜索树的后序遍历序列

#### 题目描述

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

解法：用递归解决，后序遍历，最后一个元素是根节点，前一部分一定比根节点小，后一部分一定比根节点大

```python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if len(sequence)==0:
            return False
        root = sequence[-1]
        i = 0
        while sequence[i]<root:
            i+=1
        j=i
        while j<len(sequence)-1:
            if sequence[j]<root:
                return False
            j+=1
        leftseq = sequence[:i]
        rightseq = sequence[i:len(sequence)-1]
        left=True
        if len(leftseq)>0:
            left = self.VerifySquenceOfBST(leftseq)
        right=True
        if len(rightseq)>0:
            right = self.VerifySquenceOfBST(rightseq)
        return left and right
```





### 面试题34: 二叉树中和为某一值的路径

#### 题目描述

输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

解法：深度优先搜索变形

```python
def find_path(tree, num):
    ret = []
    if not tree:
        return ret
    path = [tree]
    sums = [tree.val]

    def dfs(tree):
        if tree.left:
            path.append(tree.left)
            sums.append(sums[-1]+tree.left.val)
            dfs(tree.left)
        if tree.right:
            path.append(tree.right)
            sums.append(sums[-1] + tree.right.val)
            dfs(tree.right)
        if not tree.left and not tree.right:
            if sums[-1] == num:
                ret.append([p.val for p in path])
        path.pop()
        sums.pop()

    dfs(tree)
    return ret
```





### 面试题35: 复杂链表的复制

#### 题目描述

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针random指向一个随机节点），请对此链表进行深拷贝，并返回拷贝后的头结点。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

解法1: 使用哈希表记录特殊指针节点，空间换时间

解法2: 直接将复制的节点链接到原节点的后面

分为三步完成：

一:复制每个结点，并把新结点放在老结点后面，如1->2,复制为1->1->2->2

二:为每个新结点设置other指针

三:把复制后的结点链表拆开

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        pClone = self.CloneNode(pHead)
        pClone = self.ConnectSiblingNodes(pClone)
        return self.ReconnectNodes(pClone)
    
    def CloneNode(self,pHead):
        pNode = pHead
        while(pNode):
            pClone = RandomListNode(pNode.label)
            pClone.next = pNode.next
            pClone.random = None
            pNode.next = pClone
            pNode = pClone.next
        return pHead
    def ConnectSiblingNodes(self,pHead):
        pNode = pHead
        while(pNode):
            pClone = pNode.next
            if pNode.random:
                pClone.random = pNode.random.next
            pNode = pClone.next
        return pHead
    def ReconnectNodes(self,pHead):
        pNode = pHead
        pCloneHead = None
        pCloneNode = None
        if pNode:
            pCloneHead = pCloneNode = pNode.next
            pNode.next = pCloneHead.next
            pNode = pNode.next
        while pNode:
            pCloneNode.next = pNode.next 
            pCloneNode = pCloneNode.next
            pNode.next = pCloneNode.next
            pNode = pNode.next
        return pCloneHead

```



### 面试题36: 二叉搜索树与双向链表

#### 题目描述

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

解法：中序遍历是从小到大遍历，父节点将链接左子树最大的节点和右子树最小的节点。自然地想到左右子树递归生成已经排序的双向链表，再链接上该父节点。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        if not pRootOfTree:
            return pRootOfTree
        if not pRootOfTree.left and not pRootOfTree.right:
            return pRootOfTree
        # 处理左子树
        self.Convert(pRootOfTree.left)
        left=pRootOfTree.left

        # 连接根与左子树最大结点
        if left:
            while(left.right):
                left=left.right
            pRootOfTree.left,left.right=left,pRootOfTree

        # 处理右子树
        self.Convert(pRootOfTree.right)
        right=pRootOfTree.right

        # 连接根与右子树最小结点
        if right:
            while(right.left):
                right=right.left
            pRootOfTree.right,right.left=right,pRootOfTree
            
        while(pRootOfTree.left):
            pRootOfTree=pRootOfTree.left
        return pRootOfTree
```



### 面试题37: 序列化二叉树

#### 题目描述

请实现两个函数，分别用来序列化和反序列化二叉树

二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。

二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。



例如，我们可以把一个只有根节点为1的二叉树序列化为"1,"，然后通过自己的函数来解析回这个二叉树



解法： 前序+中序构建二叉树只能适用于不重复数字的二叉树中，并且需要等序列全部读出时才能反序列化。把二叉树分为根节点，左子树，右子树，3个部分，递归地解决

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    flag = -1
    def Serialize(self, root):
        # write code here
        if not root:
            return '#'
        return str(root.val) + ',' + self.Serialize(root.left) + ',' + self.Serialize(root.right)
    
    def Deserialize(self, s):
        # write code here
        self.flag += 1
        
        l = s.split(',')
        if self.flag >= len(s):
            return None
        
        root = None
        if l[self.flag] != '#':
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root
```





### 面试题38: 字符串的排列

#### 题目描述

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则按字典序打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

解法：动态规划，原组合+新字符

```python 
# -*- coding:utf-8 -*-
import itertools
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        return sorted(list(set(map(''.join, itertools.permutations(ss)))))
```





### 面试题39: 数组中出现次数超过一半的数字

#### 题目描述

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

解法：这个数字一定是中位数

1. 快排：随机选取一个数字，扫描数组，比它大的放右边，小的放左边，如果下标n/2在左边，则在左边用递归找中位数。
2. 扫描数组，记录每个数出现的次数

```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if len(numbers)==0:
            return 0
        if len(numbers)==1:
            return numbers[0]
        if len(numbers)%2==1:
            n = (len(numbers)+1)//2
        else:
            n = len(numbers)//2+1
        times = {}
        for i in range(len(numbers)):
            try:
                times[numbers[i]]+=1
                if times[numbers[i]]>=n:
                    return numbers[i]
            except:
                times[numbers[i]]=1
        return 0


```



### 面试题40: 最小的k个数

#### 题目描述

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

解法：

1. 同上一题一样，用快排 O(n), 可以修改输入数组时
2. O(nlogk)，最大堆/红黑树， 使用heapq，该模块是一个最小堆，需要转化成最大堆，只要在入堆的时候把值取反就可以转化成最大堆(仅适用于数字)

```python
# -*- coding:utf-8 -*-
import heapq

class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or not k or k > len(tinput):
            return []
        heapq.heapify(tinput)
        return [heapq.heappop(tinput) for _ in xrange(k)]
```





### 面试题41: 数据流中的中位数

#### 题目描述

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

解法：AVL树，插入O(logn)，找到中位数O(1), 但AVL难以实现，用堆替代左右子树，右边最小堆的任意元素大于左边的最大堆

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.st=[]
    def Insert(self, num):
        # write code here
        self.st.append(num)
    def GetMedian(self,st):
        # write code here
        self.st.sort()
        n=len(self.st)
        if n%2==1:
            return self.st[n/2]
        else:
            return (self.st[n/2-1]+self.st[n/2])/2.0
```



### 面试题42: 连续子数组的最大和

#### 题目描述

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

解法: 动态规划 max(pData[i],f(i-1)+pData[i])

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if len(array)==0:
            return 0
        if len(array)==1:
            return array[0]
        cumax=-10000
        fx = [array[0]]*len(array)
        for i in range(1,len(array)):
            fx[i]= max(array[i],fx[i-1]+array[i])
            if fx[i]>cumax:
                cumax = fx[i]
        return cumax
```



### 面试题43: 1～n整数中1出现的次数

#### 题目描述

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

解法： 获取每个位数区间上所有数中包含1的个数，然后分别对高位分析，然后递归的处理低位数

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        if n < 10:
            return 1 if n >= 1 else 0
        digit = self.get_digits(n)  # 位数
        low_nums = self.get_1_digits(digit-1)  # 最高位之前的1的个数
        high = int(str(n)[0])  # 最高位
        low = n - high * 10 ** (digit-1)  # 低位

        if high == 1:
            high_nums = low + 1  # 最高位上1的个数
            all_nums = high_nums
        else:
            high_nums = 10 ** (digit - 1)
            all_nums = high_nums + low_nums * (high - 1)  # 最高位大于1的话，统计每个多位数后面包含的1
        return low_nums + all_nums + self.NumberOf1Between1AndN_Solution(low)
    def get_digits(self,n):
        # 求整数n的位数
        ret = 0
        while n:
            ret += 1
            n /= 10
        return ret


    def get_1_digits(self,n):
        """
        获取每个位数之间1的总数
        :param n: 位数
        """
        if n <= 0:
            return 0
        if n == 1:
            return 1
        current = 9 * self.get_1_digits(n-1) + 10 ** (n-1)
        return self.get_1_digits(n-1) + current


```



### 面试题44: 数字序列中某一位的数字

#### 题目描述



### 面试题45: 把数组排成最小的数

#### 题目描述

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

解法： 比较nm与mn的大小，为了避免内存溢出，先转化为string，拼接后进行比较

```python
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ""
        lmb = lambda n1, n2:int(str(n1)+str(n2))-int(str(n2)+str(n1))
        array = sorted(numbers, cmp=lmb)
        return ''.join([str(i) for i in array])
        
```



### 面试题46: 把数字翻译成字符串

#### 题目描述

给定一个数字，把它翻译成字符串，0->a, 1->b...25->z, 输入一个数字，输出有多少种翻译的结果

解法： f(i) = f(i+1)+g(i,i+1)*f(i+2), i表示第i位，g(i,i+1)表示当i,i+1位上的数字拼成的数字在10~25内，为1，否则为0。为了避免重复，用bottom-up，从数字的末尾开始



### 面试题47: 礼物的最大价值

#### 题目描述

在m*n的棋盘上，每一格都有一个不同价值的礼物，从左上角出发，每次只能向下或向右走一格，走到右下角是停止

解法：动态规划，f(i,j)=max(f(i-1,j)+v(i,j),f(i,j-1)+v(i,j))



### 面试题48: 最长不含重复字符的子字符串

#### 题目描述

从一串字符串中找出最长不重复子字符串

解法：动态规划



### 面试题49: 丑数

#### 题目描述

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

解法： 按顺序保存已知的丑数，下一个是已知丑数中某三个数乘以2，3，5中的最小值。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        if (index <= 0):
            return 0
        uglyList = [1]
        indexTwo = 0
        indexThree = 0
        indexFive = 0
        for i in range(index-1):
            newUgly = min(uglyList[indexTwo]*2, uglyList[indexThree]*3, uglyList[indexFive]*5)
            uglyList.append(newUgly)
            if (newUgly % 2 == 0):
                indexTwo += 1
            if (newUgly % 3 == 0):
                indexThree += 1
            if (newUgly % 5 == 0):
                indexFive += 1
        return uglyList[-1]
 
```



### 面试题50: 第一个只出现一次的字符

#### 题目描述

在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.（从0开始计数）

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        dic = {}
        for ch in s:
            try:
                if dic[ch]!=0:
                    dic[ch]+=1
            except:
                dic[ch]=1
        for i in range(len(s)):
            ch = s[i]
            if dic[ch]==1:
                return i
        return -1
```





### 面试题51: 数组中的逆序对

#### 题目描述

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        global count
        count = 0
        
        def Mergefind(nums):
            global count
            Len=len(nums)
            if Len<=1:
                return nums
            newnum=Len//2
            left=Mergefind(nums[:newnum])
            right=Mergefind(nums[newnum:])
            leftindex=len(left)-1
            rightindex=len(right)-1
            result=[]
            while leftindex>=0 and rightindex>=0:   #每组数组内部的逆序数已经在对应的递归中计数
                if left[leftindex]>right[rightindex]:
                    count+=(rightindex+1)
                    result.insert(0,left[leftindex])
                    leftindex-=1
                else:
                    result.insert(0,right[rightindex])
                    rightindex-=1
            if leftindex<0:
                result=right[:rightindex+1]+result
            else:
                result=left[:leftindex+1]+result
            return result  #返回数组用于下一层递归，count已累计
        
        Mergefind(data)
        return count%1000000007
```





### 面试题52: 两个链表的第一个公共节点

#### 题目描述

输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）

解法：

1. 公共节点以后的数相同，将元素压入栈，公共部分相同且位于栈顶

2. 遍历两个链表的长度，长的链表用快指针，短的用慢指针，在公共节点处相遇

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        stack1=[]
        stack2=[]
        node1=pHead1
        node2=pHead2
        while(node1):
            stack1.append(node1.val)
            node1=node1.next
        while node2:
            if node2.val in stack1:
                return node2
            else:
                node2 = node2.next

```





### 面试题53: 在排序数组中查找数字

#### 题目描述

统计一个数字在排序数组中出现的次数。

解法： 二分法分别找开头和结尾

```python
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        start = 0
        end = len(data)-1
        num = 0
        if len(data)<=0:
            return 0
        else:
            first = self.GetFirst(data,start,end,k)
            last = self.GetLast(data,start,end,k)
            if first!=-1 and last!=-1:
                num = last-first+1
        return num
    def GetFirst(self,data,start,end,k):
        if start>end:
            return -1
        mid = (start+end)//2
        if data[mid]==k:
            if mid==0 or data[mid-1]!=k:
                return mid
            else:
                end = mid-1
        if data[mid]>k:
            end = mid-1
        if data[mid]<k:
            start = mid+1
        return self.GetFirst(data,start,end,k)
    def GetLast(self,data,start,end,k):
        if start>end:
            return -1
        mid = (start+end)//2
        if data[mid]==k:
            if mid==len(data)-1 or data[mid+1]!=k:
                return mid
            else:
                start = mid+1
        if data[mid]>k:
            end = mid-1
        if data[mid]<k:
            start = mid+1
        return self.GetLast(data,start,end,k)
```



### 面试题54: 二叉搜索树的第k个节点

#### 题目描述

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）  中，按结点数值大小顺序第三小结点的值为4。

解法：中序遍历

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        global result
        result=[]
        self.midnode(pRoot)
        if  k<=0 or len(result)<k:
            return None
        else:
            return result[k-1]
             
    def midnode(self,root):
        if not root:
            return None
        self.midnode(root.left)
        result.append(root)
        self.midnode(root.right)

```





### 面试题55: 二叉树的深度

#### 题目描述

输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

解法：每走到一个子节点+1

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot==None:
            return 0
        left = self.TreeDepth(pRoot.left)+1
        right = self.TreeDepth(pRoot.right)+1
        x = max(left,right)
        return x
```



问题二：平衡二叉树

#### 题目描述

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

在这里，我们只需要考虑其平衡性，不需要考虑其是不是排序二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if pRoot==None:
            return True
        left = self.TreeDepth(pRoot.left)+1
        right = self.TreeDepth(pRoot.right)+1
        diff = left - right
        if diff>1 or diff<-1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot==None:
            return 0
        left = self.TreeDepth(pRoot.left)+1
        right = self.TreeDepth(pRoot.right)+1
        x = max(left,right)
        return x
```

更优的解法：后序遍历，记录每个节点的深度

### 面试题56: 数组中数字出现的次数

#### 题目描述

数组中只出现过一次的两个数字。

解法：异或





### 面试题57: 和为s的两个数字

#### 题目描述

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

解法：分别从首尾开始扫描

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        if len(array)==0:
            return []
        start = 0
        end = len(array)-1
        while(array[start]+array[end]!=tsum):
            if start>=end:
                return []
            if array[start]+array[end]<tsum:
                start+=1
            if array[start]+array[end]>tsum:
                end-=1
        return [array[start],array[end]]
```





### 面试题58: 翻转字符串

#### 题目描述

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

解法：先翻转句子，再翻转单词

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        l=s.split(' ')
        return ' '.join(l[::-1])
```

(按空格分割，然后逆序join)

### 面试题59: 队列的最大值

#### 题目描述

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

窗口大于数组长度的时候，返回空

解法：

1. 将窗口当成队列，用两个栈来保持栈顶的数字为最大值
2. 删除窗口内不可能成为最大值的数，使得队列头部一直为最大值，注意：存入的是下标

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if len(num) == 0 or size == 0:
            return []
        max_num = num[0]
        dp = []
        index = 0
        for i, x in enumerate(num):
            if x >= max_num:
                max_num = x
                index = i
            else:
                if i - index >= size:
                    max_num = num[i]
                    index = i
                    for m in range(i - size + 1, i):
                        if num[m] > max_num:
                            max_num = num[m]
                            index = m
            dp.append(max_num)
        return dp[size - 1:]
```





### 面试题60: n个骰子的点数

#### 题目描述

把n个骰子扔在地上，所有朝上的面的骰子之和为s，输出所有可能的s的概率

### 面试题61: 扑克牌中的顺子

#### 题目描述

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

解法：先排序，然后补0，有对子存在的话，则不可能是顺子

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # 如果扑克牌为空，返回假
        if not numbers or len(numbers) < 5:
            return False
        # write code here
        numbers.sort()
        # 输出0的个数
        zero_num = 0
        for i in range(0, len(numbers)):
            if numbers[i] != 0:
                zero_num = i
                break
        # 输出gap的个数
        start = zero_num
        next = zero_num + 1
        end = len(numbers)
        gap_num = 0
        while next < end:
            # 如果出现对子的情况，一定不可能成为顺子
            if numbers[next] == numbers[start]:
                return False
            gap_num += numbers[next] - (numbers[start] + 1)
            start = next
            next += 1
        # 判断能够成为顺子
        if zero_num >= gap_num:
            return True
        return False

```



### 面试题62: 圆圈中最后剩下的数字

#### 题目描述

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

如果没有小朋友，请返回-1

解法：解约瑟夫环

1. 环形链表，O(nm)
2. 创新解法，当 n > 1 时： f(n,m) = [f(n-1, m)+m]%n,当 n = 1 时： f(n,m)=0，*关键是推导出关系表达式*

```python
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if not m or not n:
            return -1
        res = range(n)
        i = 0
        while len(res)>1:
            i = (m+i-1)%len(res)
            res.pop(i)
        return res[0]
```





### 面试题63: 股票的最大利润

#### 题目描述

输入股票是一段历史价格，求出买卖一次的最大利润是多少

解法：实际为所有数对的最大差值，通过记录i-1时最低的买入价，来计算当卖出价为i时的最大利润是多少



### 面试题64: 求1+2+...+n

#### 题目描述

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

解法：构造虚函数

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.sum = 0
    def Sum_Solution(self, n):
        # write code here
        def qiusum(n):
        	self.sum += n
        	n -= 1
        	return n>0 and self.Sum_Solution(n)
        
        qiusum(n)
        return self.sum

```





### 面试题65: 不用加减乘除做加法

#### 题目描述

写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

解法：使用位运算

### 面试题66: 构造乘积数组

#### 题目描述

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）

对于A长度为1的情况，B无意义，故而无法构建，因此该情况不会存在。



### 面试题67: 在排序数组中查找数字

#### 题目描述



#### 