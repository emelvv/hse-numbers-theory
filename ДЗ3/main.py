from Mat import Mat

A = Mat([[11, -13, 1], [1, 0, 0], [0, 1, 0]], slu=2)

A[:, 0]+=A[:, 1]
A[:, 1]-=7*A[:, 0]
A[:, 0]+=2*A[:, 1]
A.swap_cols(0, 1)
A[:, 2] -= A[:, 0]

print(A)
