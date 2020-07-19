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



