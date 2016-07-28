'''generate the adjacency matrix for a 60 vertex truncated icosahedron, aka buckyball'''
import math,itertools,os,csv
from numpy import matrix #moving inside function as it is not used elsewhere and is very slow.

def distance(a,b):
    '''calculates the straight line distance between two points a and b'''
    SS = sum(map(lambda x,y:(x-y)**2,a,b))
    return SS**0.5

def makecoords():
    '''generate a list of coordinates for the buckyball'''
    phi = 0.5*(1+math.sqrt(5))
    c1 = (0,1,3*phi)
    c2 = (2,(1+2*phi),phi)
    c3 = (1,2+phi,2*phi)
    combos1 = list(itertools.product((1,-1),repeat=2))
    for i in range(len(combos1)):
        combos1[i] = (1,)+combos1[i]
    combos23 = list(itertools.product((1,-1),repeat=3))
    coords = []
    for i in combos1:
        coords.append(matrix(map(lambda x,y:x*y,c1,i)).transpose()) #column vectors
    for i in combos23:
        coords.append(matrix(map(lambda x,y:x*y,c2,i)).transpose())
        coords.append(matrix(map(lambda x,y:x*y,c3,i)).transpose())
    #permutation matrices
    P1 = matrix([[0,0,1],[1,0,0],[0,1,0]])
    P2 = matrix([[0,1,0],[0,0,1],[1,0,0]])
    for i in coords[:]:
        coords.append((P1*i))
        coords.append((P2*i))
    coords = [tuple(i.transpose().tolist()[0]) for i in coords]
    return coords

def graphcoords(coords):
    '''takes a list of triples and makes a 3-D graph'''
    #moving inside function as it is not used elsewhere and is very slow.
    import pylab as p
    import mpl_toolkits.mplot3d.axes3d as p3
    (x,y,z) = zip(*coords) #unzip the list of tuples.
    fig = p.figure()
    ax=p3.Axes3D(fig)
    ax.plot_wireframe(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    p.show()

def makeadjmat(coords):
    '''make a 60x60 adjacency matrix for the coordinates.'''
    D = [[distance(i,j) for j in coords] for i in coords]
    for i in range(len(D)):
        for j in range(len(D[i])):
            if D[i][j] != 2.0:
                D[i][j] = 0
            else:
                D[i][j] = 1
    return D

def writecsv(matrix,ofile):
    '''write the matrix to a csv file'''
    with open(ofile,'wb') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(tuple(row))
            
if __name__ == "__main__":
    ofile = 'bucky.csv'
    coords = makecoords()
    D = makeadjmat(coords)
    writecsv(D,ofile)


