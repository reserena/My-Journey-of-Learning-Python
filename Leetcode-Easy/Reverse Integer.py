class Solution(object):
    def reverse(self, x):
        x = str(x)
        l = len(x)
        if x[0] == '-':
            y = x[0]
            for i in range(l):
                if i == (l-1):
                    if abs(int(y)) > 2147483647:
                        return 0
                    return int(y)
                y = y + x[l-i-1]
        else:
            y = ''
            for i in range(l):
                y = y + x[l-i-1]
            if abs(int(y)) > 2147483647:
                return 0
            return int(y)
