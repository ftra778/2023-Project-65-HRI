def moveLimiter(new, curr, uppB, lowB):
    if abs(new - curr) > lowB:            #NEW
        if abs(new - curr) > uppB:
            if (new > curr):
                curr = curr + uppB
            else:
                curr = curr - uppB
        else:
            curr = new
    return curr

x = -1.57
print(moveLimiter(-1.565, x, 0.3, 0.1))