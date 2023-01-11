def Alignmentmat( s1, s2,m,n):
    
    rows, cols = (16, 16)
    matrix = [[0 for i in range(cols+1)] for j in range(rows+1)]
    #making the first column zero
    for i in range(0,m):
        matrix[i][0] = 0;
    print(matrix[m][n])    
    #making the first row zero
    for j in range(0,n):
        matrix[0][j] = 0;
        
    for i in range(0,m):    
        for j in range(0,n):
            #if its a match score is +5 from the diagonal
            if(s1[i-1] == s2[j-1]):
                matrix[i][j] = matrix[i-1][j-1] +5 
            #if its a mismatch score is -4 from diagonal or top or left    
            else:
                matrix[i][j] = max(matrix[i-1][j],
                                   matrix[i][j-1],
                                   matrix[i-1][j-1]) -4
    
    
    return matrix[rows][cols]




m=16 #length of s1 is 16
n=16 #length of s2 is 16

s1 = "aaaaccccggggtttt"
s2 = "aaaccccggggtttta"

print(Alignmentmat(s1, s2, m, n))
    

   


        
            
        
