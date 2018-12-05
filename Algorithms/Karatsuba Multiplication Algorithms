def karatsuba(a, b):
    if (type(a) != str) and (type(b) != str):
        try:
            x = str(a)
            y = str(b)
        except:
            print("Cannot be converted to string")
            exit
    else:
        x = a
        y = b
    xlen = len(x)
    ylen = len(y)
    
    if xlen < ylen:
        x = "0"*(ylen-xlen)+x
        length = ylen
    elif xlen > ylen:
        y = "0"*(xlen-ylen)+y
        length = xlen
    else:
        length = xlen
        
    if (length == 1):
        return int(x)*int(y)
    elif (length == 2):
        one = int(x[0])*int(y[0])
        two = int(x[1])*int(y[1])
        x1 = int(x[0])+int(x[1])
        y1 = int(y[0])+int(y[1])
        if (len(str(x1))>1) or (len(str(y1))>1):
            three = karatsuba_v2(x1, y1)-one-two
        else:
            three = x1*y1-one-two
        return one*(10**2)+three*10+two
    else:
        if length%2 == 0:
            mid = int(length/2)
        else:
            mid = int((length+1)/2)
            
        one = karatsuba(x[:mid],y[:mid])
        two = karatsuba(x[mid:],y[mid:])
        three = karatsuba(int(x[:mid])+int(x[mid:]), int(y[:mid])+int(y[mid:]))-one-two
        return one*(10**((length-mid)*2))+three*(10**(length-mid))+two
