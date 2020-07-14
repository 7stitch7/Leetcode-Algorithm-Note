class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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

sol = Solution()
L = {67,0,24,58}
print(sol.printListFromTailToHead(L))