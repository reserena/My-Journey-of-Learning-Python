class Solution(object):
    def twoSum(self, nums, target):
        i = 0
        lst = []
        for i in range(len(nums)):
            for n in range(i+1,len(nums)):
                if (nums[i] + nums[n]) == target:
                    lst.append(i)
                    lst.append(n)
                    return lst
