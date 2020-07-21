#-*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def FindKthToTail(self,head,k):
        # write code here
        FirstNode = head
        BindNode = head
        if not head.next:
            return False
        if k<=0:
            return False
        for i in range(k-1):
            if not BindNode.next:
                BindNode = BindNode.next
            else:
                return False
        while(BindNode.next!=None):
            FirstNode = FirstNode.next
            BindNode = BindNode.next
        return FirstNode.val

so = Solution()
nodelist = ListNode(1)
for i in range(2,6):
    node = ListNode(i)
    nodelist.next = node


re = so.FindKthToTail(nodelist,2)
print(re)