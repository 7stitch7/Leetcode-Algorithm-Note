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

