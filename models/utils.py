def get_divisors(n):
    '''
    Returns list divisors of a number
    '''
    divisors = []
    for i in range(1,n//2):
        if n%i==0:
            divisors.append(i)
    divisors.append(n)
    return divisors