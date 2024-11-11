
path = 'image.txt'

f = open(path, 'r')
L=[]
s = f.read()
f.close()
s = s.replace("\n"," ")
s = s.replace("\t"," ")
L = s.split(" ")
print(L)

f = open("r2.txt", 'w')

for i in L:
    # np.set_printoptions(threshold=sys.maxsize)
    print(i,file = f)
    # print("\n",file = f)
    # print()

f.close()