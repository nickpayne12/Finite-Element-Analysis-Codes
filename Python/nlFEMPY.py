import timeit
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from math import sqrt

def r_theta(x: np.ndarray, y):
    """Returns polar coordinates r and theta for a given x and y pair."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return r, theta


def map_DOF(x):
    return x*2, x*2+1


def gather_index(a, b):
    """return index pairs between index locations a and b, including a, b."""
    a1, a2 = a
    b1, b2 = b
    c = np.max(abs(np.array(a)-np.array(b))) + 1 # +1 to include last pair
    output = [None]*c
    if a1 == b1 and a2 < b2:
        for i in enumerate(output):
            output[i[0]] = [a1, a2 + i[0]]
    elif a1 == b1 and a2 > b2:
        for i in enumerate(output):
            output[i[0]] = [a1, a2 - i[0]]
    elif a2 == b2 and a1 < b1:
        for i in enumerate(output):
            output[i[0]] = [a1 + i[0], a2]
    elif a2 == b2 and a1 > b1:
        for i in enumerate(output):
            output[i[0]] = [a1 - i[0], a2]
    else:
        raise Exception('Input arrays must share 1 common values.')
    return output


def shape4N(xi, eta):
    """N matrix of 4 node quadrilateral element."""
    N = [[(1/4)*(1-xi)*(1-eta), 0,  (1/4)*(1+xi)*(1-eta), 0,  (1/4)*(1+xi)*(1+eta), 0,  (1/4)*(1-xi)*(1+eta), 0], 
         [ 0,  (1/4)*(1-xi)*(1-eta), 0,  (1/4)*(1+xi)*(1-eta), 0,  (1/4)*(1+xi)*(1+eta), 0,  (1/4)*(1-xi)*(1+eta)]]
    
    return np.array(N)


def N1234(xi, eta):
    return np.array([(1/4)*(1-xi)*(1-eta), (1/4)*(1+xi)*(1-eta), (1/4)*(1+xi)*(1+eta), (1/4)*(1-xi)*(1+eta)])


def xy_shape8N(xieta, xy):
    """Takes in the local coordinate [xi,eta] and the node points [x,y]_i
      of a 8 noded isoparametric element.
      Returns the position in [x,y]"""
    xi = xieta[0]
    eta = xieta[1]
    N = np.zeros(8)
    x = xy[0,:]
    y = xy[1,:]
    N[0] = -(1 - xi)*(1 - eta)*(1 + xi + eta)/4
    N[1] = -(1 + xi)*(1 - eta)*(1 - xi + eta)/4
    N[2] = -(1 + xi)*(1 + eta)*(1 - xi - eta)/4
    N[3] = -(1 - xi)*(1 + eta)*(1 + xi - eta)/4
    N[4] = (1 - xi**2)*(1 - eta)/2
    N[5] = (1 + xi)*(1 - eta**2)/2
    N[6] = (1 - xi**2)*(1 + eta)/2
    N[7] = (1 - xi)*(1 - eta**2)/2
    return np.array([np.dot(N,x), np.dot(N,y)])


def BESTFITQ(input_array):
    """Finds the values of a quantity at elemental nodes based on an array of input values defined at the gauss points.
    Uses a least squares fitting technique to find the nodal array values."""
    a = input_array

    N = np.array([N1234(-1/np.sqrt(3), -1/np.sqrt(3)), 
                  N1234(1/np.sqrt(3), -1/np.sqrt(3)), 
                  N1234(1/np.sqrt(3), 1/np.sqrt(3)), 
                  N1234(-1/np.sqrt(3), 1/np.sqrt(3))])
    
    K = N.T@N
    F = N.T@a
    Q = np.linalg.lstsq(K, F, rcond = None)

    return Q[0]


##### Classes #####

class Mesher:
    def __init__(self):
        pass

    def coords_quarterCircle(self, r):
        """Returns the block coordinates for a 90deg circular sector.  Input radius, r.
        Output is a 3D array in the form of [xy, point number, block number]"""
        block0 = np.array([[0, r/2, r/2, 0, r/4, r/2, r/4, 0],
                            [0, 0, r/2, r/2, 0, r/4, r/2, r/4]])
        
        block1 = np.array([[r/2, r, r/np.sqrt(2), r/2, r*3/4, r*np.cos(np.pi/8), r*(1+np.sqrt(2))/4, r/2],
                            [0, 0, r/np.sqrt(2), r/2, 0, r*np.sin(np.pi/8), r*(1+np.sqrt(2))/4, r/4]])

        block2 = np.array([[0, r/2, r/np.sqrt(2), 0, r/4, r*(1+np.sqrt(2))/4, r*np.cos(np.pi*3/8), 0],
                            [r/2, r/2, r/np.sqrt(2), r, r/2, r*(1+np.sqrt(2))/4, r*np.sin(np.pi*3/8), r*3/4]])
        
        output = np.zeros((2,8,3))
        output[:,:,0] = block0
        output[:,:,1] = block1
        output[:,:,2] = block2
        return output
    
    def coords_Quad(self, w, h):
        """Returns block coordinates for a quadrilaterial of width w and height h."""
        output = np.zeros((2,8,1))
        block0 = np.array([[0, w, w, 0, w/2, w, w/2, 0],
                            [0, 0, h, h, 0, h/2, h, h/2]])
        output[:,:,0] = block0
        return output
    
    def set_params(self, blocks_nums, div_nums, blocks_coords, void = [], merge = [[]], surfs = [[]]):
        self.blocks_nums = np.array(blocks_nums)
        self.div_nums = np.array(div_nums)
        self.void = void
        self.__surflist = surfs
        if len(np.shape(merge)) == 2 or merge == None:
            self.merge = merge
        else:
            raise Exception('merge must be array of dimension 2.')
        if len(np.shape(blocks_coords)) == 3:
            self.blocks_coords = np.array(blocks_coords)
        else:
            raise Exception('blocks_coords must be array of dimension 3.')

    def create(self):
        NS = self.blocks_nums[0] #number of blocks in S direction
        NW = self.blocks_nums[1] #number of blocks in W direction
        NSW = NS*NW #total number of blocks

        NSD = self.div_nums[0] #array containing number of divisions in each block in S
        NWD = self.div_nums[1] #array containing number of divisions in each block in W

        #Count total number of elements not including void blocks
        BLOCKID = 0
        self.num_elements = 0
        for KW in range(NW):
            for KS in range(NS):
                if BLOCKID in self.void:
                    pass
                else:
                    self.num_elements += NSD[KS]*NWD[KW]
                BLOCKID += 1
        NNS = 1 + np.sum(NSD) #total nodes along S
        NNW = 1 + np.sum(NWD) #toal nodes along W

        NNT = NNS*NNW #total number of nodes

        # NNAR = np.zeros((NNS,NNW)) #initialize node list with zeros
        NNAR = np.full((NNS,NNW), None)
        NX = np.full((NNS,NNW), None) #initialize nodal x coordinate list with None
        NY = np.full((NNS,NNW), None) # "" y coordinate list with None

        # Loop through and set -1 to all blocks not set to void
        BLOCKID = 0
        WPLACE = 0
        for KW in range(NW):
            SPLACE = 0
            for KS in range(NS):
                if BLOCKID in self.void:
                    pass
                else:
                    NNAR[SPLACE : SPLACE + NSD[KS] + 1, WPLACE : WPLACE + NWD[KW] + 1 ] = -1
                SPLACE += NSD[KS]
                BLOCKID += 1
            WPLACE += NWD[KW]
        # print(NNAR)

        # Create a map of the corner points
        self.MAPPER = np.reshape(list(range((NS + 1)*(NW + 1))), (NS + 1, NW + 1), order='F')

        # Create an array indicating the indices of corner points of each block
        indexW = [0]
        WPLACE = 0
        indexS = [0]
        SPLACE = 0
        for k in NWD:
            WPLACE += k
            indexW.append(WPLACE)
        for p in NSD:
            SPLACE += p
            indexS.append(SPLACE)
        X, Y = np.meshgrid(indexS, indexW)
        self.POS = [None]*len(np.ravel(X))
        for i in range(len(np.ravel(X))):
            self.POS[i] = [np.ravel(X)[i], np.ravel(Y)[i]]
        # print(self.POS)

        # Set merge nodes
        if self.merge != [[]]:
            for pair in self.merge:
                # get node numbers of nodes on merge lists
                line1 = pair[0:2]
                line2 = pair[2:4]
                # print(line1,line2)
                # print(self.POS[line1[0]], self.POS[line1[1]])
                # Convert nodes to lists of index positions to be merged
                line1 = np.array(gather_index(self.POS[line1[0]], self.POS[line1[1]]))
                # print(line1)
                # print(self.POS[line2[0]], self.POS[line2[1]])
                line2 = np.array(gather_index(self.POS[line2[0]], self.POS[line2[1]]))
                # print(line2)
                # print(np.max(line1[:,0])>np.max(line2[:,0]))
                # Find the line with the highest node number and conver that to the lower node numbers
                if np.max(line1[:,0]) < np.max(line2[:,0]):
                    for point in enumerate(line1):
                        # print(point)
                        # print(line2[point[0]])
                        node_num = np.ravel_multi_index(line2[point[0]], NNAR.shape, order="F")
                        NNAR[point[1][0], point[1][1]] = node_num
                else:
                    for point in enumerate(line2):
                        node_num = np.ravel_multi_index(line1[point[0]], NNAR.shape, order="F")
                        NNAR[point[1][0], point[1][1]] = node_num
                # print(NNAR)

        # Assign final node numbers
        DUMMY = np.full(NNT, None)
        NCOUNT = 0
        for node in enumerate(np.ravel(NNAR, order="F")):
            # self.maxnode = NCOUNT
            if node[1] == None:
                pass
            elif node[1] < 0:
                DUMMY[node[0]] = NCOUNT # Assign the node number according to sequence
                NCOUNT += 1
            elif node[1] > 0:
                DUMMY[node[0]] = node[1] # Assign the merged node number
                if node[1] == NCOUNT:
                    NCOUNT += 1
            else:
                pass
            self.maxnode = NCOUNT
        NNAR = np.reshape(DUMMY, NNAR.shape, order="F")
        # print('final',NNAR)

        # Move through NNAR and create elements
        self.elements = np.zeros((self.num_elements,4), int)
        WPLACE = 0
        EPLACE = 0
        BLOCKID = 0
        for KW in range(NW):
            SPLACE = 0
            for KS in range(NS):
                if BLOCKID in self.void:
                    pass
                else:
                    BLOCKNODES = NNAR[SPLACE : SPLACE + NSD[KS] + 1, WPLACE : WPLACE + NWD[KW] + 1 ]
                    # print(BLOCKNODES.shape)
                    for y in range(BLOCKNODES.shape[1]-1):
                        for x in range(BLOCKNODES.shape[0]-1):
                            nodelist = [BLOCKNODES[x,y],
                                     BLOCKNODES[x+1,y],
                                     BLOCKNODES[x+1,y+1],
                                     BLOCKNODES[x,y+1]]
                            self.elements[EPLACE,:] = nodelist
                            EPLACE += 1
                SPLACE += NSD[KS]
                BLOCKID += 1
            WPLACE += NWD[KW]

        #First make [xi,eta] coordinates for every node then assign
        # [x,y] cordinates to each node of each block.
        BLOCKID = 0
        WPLACE = 0
        for KW in range(NW):
            SPLACE = 0
            for KS in range(NS):
                if BLOCKID in self.void:
                    pass
                else:
                    # print('NSD[KS]', NSD[KS])
                    # print('NWD[KW]', NWD[KW])
                    xi, eta = np.meshgrid(np.linspace(-1, 1, NSD[KS] + 1), np.linspace(-1, 1, NWD[KW] + 1)) #create mesh grid of xi and eta points
                    # print('xi shape:', xi.shape)
                    # print('eta shape:', eta.shape)
                    xi_and_eta = np.vstack((np.ravel(xi), np.ravel(eta))) # ravel and stack xi over eta
                    xy = np.zeros((len(np.ravel(xi)), 2)) # pre-make an array to hold the xy coords of each xi, eta pair.
                    for point in enumerate(xi_and_eta.T):
                        xy[point[0],:] = xy_shape8N(point[1], self.blocks_coords[:,:, BLOCKID]) # create the xy coord
                    X = np.reshape(xy[:,0], xi.T.shape, order='F') # make a 2D array of just x coordinates
                    Y = np.reshape(xy[:,1], eta.T.shape, order='F') # make a 2D array of just y coordinates
                    # move through the whole mesh and place coordintes in NX and NY arrays
                    NX[SPLACE : SPLACE + NSD[KS] + 1, WPLACE : WPLACE + NWD[KW] + 1 ] = X 
                    NY[SPLACE : SPLACE + NSD[KS] + 1, WPLACE : WPLACE + NWD[KW] + 1 ] = Y 
                SPLACE += NSD[KS]
                BLOCKID += 1
            WPLACE += NWD[KW]

        self.nodes = np.zeros((2, self.maxnode)) # make an object to hold the x,y coordinates of the whole mesh in a 2Xnum_nodes size

        for y in range(NNAR.shape[1]):
            for x in range(NNAR.shape[0]):
                # assign node position values only from node number entries that aren't None
                if NNAR[x,y] != None:
                    self.nodes[0,NNAR[x,y]] = NX[x,y]
                    self.nodes[1,NNAR[x,y]] = NY[x,y]

        #Make and save surfaces
        self.surfs = []
        for row in range(len(self.__surflist)):
            self.surfs.append([])
        for pair in enumerate(self.__surflist):
            point1 = pair[1][0]
            point2 = pair[1][1]
            line1 = np.array(gather_index(self.POS[point1], self.POS[point2]))
            for point in enumerate(line1):
                self.surfs[pair[0]].append(NNAR[point[1][0], point[1][1]])


class Mesh:
    """Mesh object of a meshed region with arrays containing the nodal positions and element numbering."""
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.gauss_points = None
        self.material = None

    def assign_material(self, mat_model):
        self.material = mat_model

    def make_mesh(self, mesher: Mesher):
        """Creates a mesh based on parameter objects from a Mesher object."""
        
        self.nodes = mesher.nodes
        self.elements = mesher.elements

        # Create the gauss points for each element and store in self.gauss_points.  The integration rule used is 2X2.    
        GP = np.array([[-1/np.sqrt(3), -1/np.sqrt(3), 1.0],
                       [1/np.sqrt(3), -1/np.sqrt(3), 1.0], 
                       [1/np.sqrt(3), 1/np.sqrt(3), 1.0],
                       [-1/np.sqrt(3), 1/np.sqrt(3), 1.0]])
        
        N = np.zeros((8,8))
        for i in range(0,4):
            xi = GP[i,0]
            eta = GP[i,1]
            N[i*2:i*2+2,:] = [[(1/4)*(1-xi)*(1-eta), 0,  (1/4)*(1+xi)*(1-eta), 0,  (1/4)*(1+xi)*(1+eta), 0,  (1/4)*(1-xi)*(1+eta), 0], 
                               [ 0,  (1/4)*(1-xi)*(1-eta), 0,  (1/4)*(1+xi)*(1-eta), 0,  (1/4)*(1+xi)*(1+eta), 0,  (1/4)*(1-xi)*(1+eta)]]

        num_elements = self.elements.shape[0]
        Q = self.nodes[:, self.elements.ravel(order='C')]
        Q = np.reshape(Q, (8, num_elements), order='F')
        self.gauss_points = np.vstack((np.reshape(N@Q, (2, num_elements*4), order='F'), np.tile(GP[:,2].T, [1, num_elements])))

        # Create the B matrix and determinate of the Jacobian for each gauss point.
        def BmatdetJ(self, x, loc):
            """Computes the B matrices and value of the Jacobian determinates for each element and guass point."""
            xi, eta = loc[0:2]
            x1, x2, x3, x4, y1, y2, y3, y4 = np.ravel(x)
            Jac = 1./4.*np.array([[-(1-eta)*x1 + (1-eta)*x2 + (1+eta)*x3 - (1+eta)*x4, -(1-eta)*y1 + (1-eta)*y2 + (1+eta)*y3 - (1+eta)*y4],
                                  [-(1-xi)*x1 - (1+xi)*x2 + (1+xi)*x3 + (1-xi)*x4, -(1-xi)*y1 - (1+xi)*y2 + (1+xi)*y3 + (1-xi)*y4]])
            J11 = Jac[0,0]
            J12 = Jac[0,1]
            J21 = Jac[1,0]
            J22 = Jac[1,1]
            detJ = J11*J22-J12*J21
            A = (1.0/detJ)*np.array([[J22, -J12, 0., 0.], [0., 0., -J21, J11], [-J21, J11, J22, -J12]])
            G = (1./4.)*np.array([[-(1-eta), 0, (1-eta), 0, (1+eta), 0, -(1+eta), 0],
                    [-(1-xi), 0, -(1+xi), 0, (1+xi), 0, (1-xi), 0],
                    [0, -(1-eta), 0, (1-eta), 0, (1+eta), 0, -(1+eta)],
                    [0, -(1-xi), 0, -(1+xi), 0, (1+xi), 0, (1-xi)]])
            Bmat = A@G
            return Bmat, detJ
        self.B = np.zeros((3,8,num_elements*4))
        self.detJ = np.zeros(num_elements*4)
        for elem in enumerate(self.elements):
            for i in range(4):
                output1, output2 = BmatdetJ(self, self.nodes[:, elem[1]], GP[i,:])
                self.B[:,:, elem[0]*4+(i)] = output1
                self.detJ[elem[0]*4+(i)] = output2

    def plot(self):
        node_x_locations = self.nodes[0]
        node_y_locations = self.nodes[1]
        max_x = np.max(node_x_locations)
        max_y = np.max(node_y_locations)

        fig, ax = plt.subplots()
        for element in self.elements:
            x = self.nodes[0, element]
            y = self.nodes[1, element]
            ax.fill(x,y, facecolor='None', edgecolor='black')
        ax.scatter(node_x_locations, node_y_locations, color="black")
        for i in range(self.nodes.shape[1]):
            ax.annotate('n: '+str(i), (node_x_locations[i]+0.01*max_x, node_y_locations[i]+0.02*max_y))
        
        for i in range(self.elements.shape[0]):
            ax.annotate(f' {i}', (np.mean(self.nodes[0, self.elements[i]]), np.mean(self.nodes[1, self.elements[i]])), ha='center', va='center', color="red")
        
        # for i in range(self.gauss_points.shape[1]):
            # ax.annotate(str(i), (self.gauss_points[0, i], self.gauss_points[1, i]), color='blue')
            # ax.scatter(self.gauss_points[0, i], self.gauss_points[1, i], color='blue', marker='x')
            
        # plt.subplots_adjust(top=2, right=2, bottom=1.5)
        ax.set_aspect('equal', 'box')
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.show()

class Material_model:
    """Material model class object which contains the stiffness matrix, D, and other constitutive matrices.\n
       arguments:\n
       model_inputs = a value or array of values which populate a particular model\n
       model_type = a string which specified the type of material model.\n
       Available model input and types:\n
       [E, nu] "linear elastic"\n
       """
    def __init__(self, model_inputs: float, model_type: str):
        self.type = model_type
        self.inputs = model_inputs
    def D_matrix(self, S = None, E = None):
        if self.type == "linear elastic, plane stress":
            young_modulus = self.inputs[0]
            nu = self.inputs[1]
            D = young_modulus/(1-nu**2)*np.array([[1, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5*(1.0-nu)]])
        if self.type == "linear elastic, plane strain":
            young_modulus = self.inputs[0]
            nu = self.inputs[1]
            D = young_modulus/((1 + nu)*(1 - 2*nu))*np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2*nu)/2.0]])
        if self.type == "strain locking, plane strain": 
            A, B, phi, E_1, E_2, nu_12, nu_23, G = self.inputs

            c = np.cos(phi)
            s = np.sin(phi)

            s11 = S[0]
            s22 = S[1]
            s12 = S[2]
            e11 = E[0]
            e22 = E[1]
            e33 = 0
            e12 = E[2]

            C_matrix = np.array([[ (E_1**2*(nu_23 - 1))/(2*E_2*nu_12**2 - E_1 + E_1*nu_23), -(E_1*E_2*nu_12)/(2*E_2*nu_12**2 - E_1 + E_1*nu_23), 0],
                                 [-(E_1*E_2*nu_12)/(2*E_2*nu_12**2 - E_1 + E_1*nu_23), -(E_2*(- E_2*nu_12**2 + E_1))/((nu_23 + 1)*(2*E_2*nu_12**2 - E_1 + E_1*nu_23)),   0],
                                 [0, 0, 2*G]])
            
            C11matrixbar = C_matrix[0,0]*c**4 + C_matrix[1,1]*s**4 + 2*(C_matrix[0,1]+C_matrix[2,2])*s**2*c**2
            C12matrixbar = (C_matrix[0,0] + C_matrix[1,1] - 2*C_matrix[2,2])*s**2*c**2 + C_matrix[0,1]*(c**4 + s**4)
            C16matrixbar = (C_matrix[0,0] - C_matrix[0,1] - C_matrix[2,2])*s*c**3 - (C_matrix[1,1] - C_matrix[0,1] - C_matrix[2,2])*s**3*c
            C22matrixbar = C_matrix[0,0]*s**4 + C_matrix[1,1]*c**4 + 2*(C_matrix[0,1]+C_matrix[2,2])*s**2*c**2
            C26matrixbar = (C_matrix[0,0] - C_matrix[0,1] - C_matrix[2,2])*c*s**3 - (C_matrix[1,1] - C_matrix[0,1] - C_matrix[2,2])*c**3*s
            C66matrixbar = (C_matrix[0,0] + C_matrix[1,1] - 2*C_matrix[0,1]-2*C_matrix[2,2])*c**2*s**2 + C_matrix[2,2]*(c**4+s**4)

            T = np.array([[c**2, s**2, 2*c*s], [s**2, c**2, -2*c*s], [-c*s, c*s, c**2-s**2]])
            Tinv = np.array([[c**2, s**2, -2*c*s], [s**2, c**2, 2*c*s], [c*s, -c*s, c**2-s**2]])

            sig_12 = np.array([[s11], [s22], [s12]])
            eps_12A = np.array([[e11], [e22], [e12]])
            
            sig_np = Tinv@sig_12
            eps_npA = Tinv@eps_12A

            Bmat = C_matrix@eps_12A
            sig_npmat = Tinv@Bmat
            snnm = sig_npmat[0]
            snnf = snnm # Guess
            snnf = sig_np[0]
            error = np.inf

            # Stiffness ratio to calculate stress partition approach:
            # while error > 0.01:
            #     # Cm = ((E_1**2*(nu_23 - 1))/(2*E_2*nu_12**2 - E_1 + E_1*nu_23));
            #     Cm = C11matrixbar
            #     Cf = (1/((A*B)/(B + (snnf - ((A - 1)*(B*(2*A - 1))**(1/2))/(2*A - 1))**2)**(3/2)))
            #     snnfnew = (Cm/Cf + 1)*sig_np[0] - Cm/Cf*snnm
            #     error = abs((snnfnew - snnf)/snnf)
            #     snnf = snnfnew


            ## Newton method to find fiber stress given
            f = lambda sig: A + A*(sig - sqrt(B*(2*A - 1))*(A - 1)/(2*A - 1))/sqrt(B + (sig - sqrt(B*(2*A - 1))*(A - 1)/(2*A - 1))**2) - 1 - eps_npA[0]
            fprime = lambda sig: A*B*sqrt((B*(2*A - 1)**2 + (sig*(2*A - 1) - sqrt(B*(2*A - 1))*(A - 1))**2)/(2*A - 1)**2)*(2*A - 1)**4/(B*(2*A - 1)**2 + (sig*(2*A - 1) - sqrt(B*(2*A - 1))*(A - 1))**2)**2
            xn = snnf
            # print("xn: ", xn)

            while error > 1000:
                fval = f(xn)
                fprimeval = fprime(xn)
                xnplus1 = xn - fval/fprimeval
                error = abs(xnplus1 - xn)
                # print("error: ", error)
                # print("xn", xn)
                # print("xnplus1", xnplus1)f
                # print("eps_npA[0]", eps_npA[0])
                # print("f: ", f(xn))
                # print("fprime: ", fprime(xn))
                xn = xnplus1

            Cf = (1/((A*B)/(B + (xn - ((A - 1)*(B*(2*A - 1))**(1/2))/(2*A - 1))**2)**(3/2)))


            Q11 = Cf
            Q12 = 0
            Q16 = 0
            Q22 = 0
            Q26 = 0
            Q66 = 0

            C11fbar = Q11*c**4 + Q22*s**4 + 2*(Q12 + 2*Q66)*c**2*s**2
            C22fbar = Q11*s**4 + Q22*c**4 + 2*(Q12 + 2*Q66)*c**2*s**2
            C12fbar = (Q11 + Q22 - 4*Q66)*c**2*s**2 + Q12*(c**4 + s**4)
            C66fbar = (Q11 + Q22 - 2*Q12 - 2*Q66)*c**2*s**2 + Q66*(c**4 + s**4)
            C16fbar = (Q11 - Q12 - 2*Q66)*c**3*s - (Q22 - Q12 - 2*Q66)*c*s**3
            C26fbar = (Q11 - Q12 - 2*Q66)*c*s**3 - (Q22 - Q12 - 2*Q66)*c**3*s


            C_fiber = np.array([[C11fbar.item(), C12fbar.item(), C16fbar.item()], [C12fbar.item(), C22fbar.item(), C26fbar.item()], [C16fbar.item(), C26fbar.item(), C66fbar.item()]])

            D = C_fiber + C_matrix

        return D

class Global_K_matrix:
    """The global stiffness matrix object."""
    def __init__(self, input_mesh: Mesh):
        self.mesh = input_mesh
        self.nodes = self.mesh.nodes
        elements = self.mesh.elements
        self.K_global = np.zeros((self.nodes.shape[1]*2, self.nodes.shape[1]*2))
        self.DOF_mapping = np.zeros((elements.shape[0], 8), dtype='int')
        self.material = input_mesh.material
        i = 0
        for element in elements:
            self.DOF_mapping[i,:] = np.ndarray.flatten(np.array(list(map(map_DOF, element))).T, order='F')
            i += 1
    def build(self, S, E):
        """Constructs the global stiffness matrix."""
        self.K_global = np.zeros((self.nodes.shape[1]*2, self.nodes.shape[1]*2))
        S = S.return_all()
        E = E.return_all()
        for p in enumerate(self.mesh.elements):
            gauss_index = np.arange(4*p[0], 4*p[0]+4)
            index = self.DOF_mapping[p[0],:]
            for k in gauss_index:
                D = self.material.D_matrix(S[:,k], E[:,k])
                B = self.mesh.B[:,:,k]
                weight = self.mesh.gauss_points[2,k]
                detJ = self.mesh.detJ[k]
                k_element = B.T@D@B*weight*detJ
                for n in enumerate(index):
                    for m in enumerate(index):
                        self.K_global[n[1], m[1]] += k_element[n[0],m[0]]


class Global_T_matrix:
    """The global internal nodal force vector."""
    def __init__(self, input_mesh: Mesh):
            self.mesh = input_mesh
            self.nodes = self.mesh.nodes
            elements = self.mesh.elements
            self.T_global = np.zeros(self.nodes.shape[1]*2)
            self.DOF_mapping = np.zeros((elements.shape[0], 8), dtype='int')
            i = 0
            for element in elements:
                self.DOF_mapping[i,:] = np.ndarray.flatten(np.array(list(map(map_DOF, element))).T, order='F')
                i += 1
    def build(self, S):
        """Constructs the global internal force vector."""
        self.T_global = np.zeros(self.nodes.shape[1]*2)
        i = 0
        for p in enumerate(self.mesh.elements):
            gauss_index = np.arange(4*p[0],4*p[0]+4)
            for k in gauss_index:
                B = self.mesh.B[:,:,k]
                weight = self.mesh.gauss_points[2,k]
                detJ = self.mesh.detJ[k]
                t = B.T@S.return_all()[:,k]*weight*detJ # S is the stress field
                index = self.DOF_mapping[i,:]
                n = 0
                for position1 in index:
                    self.T_global[position1] += t[n]
                    n += 1
            i += 1

class Global_F_matrix:
    """The global applied nodal force vector."""
    def __init__(self, input_mesh: Mesh):
            self.mesh = input_mesh
            nodes = self.mesh.nodes
            elements = self.mesh.elements
            self.F_global = np.zeros(nodes.shape[1]*2)
            self.DOF_mapping = np.zeros((elements.shape[0], 8), dtype='int')
            i = 0
            for element in elements:
                self.DOF_mapping[i,:] = np.ndarray.flatten(np.array(list(map(map_DOF, element))).T, order='F')
                i += 1
    def apply_traction(self, node_list, trac_value, trac_dir):
        """Constructs the global applied force vector."""
        res = []
        [res.append(x) for x in node_list if x not in res]
        node_list = res
        node_pairs = []
        for node1 in node_list:
            for node2 in node_list:
                if node1 == node2:
                    break
                else:
                    for element in self.mesh.elements:
                        if node1 in element and node2 in element:
                            node_pairs.append([node1, node2])
                            break
        for pair in node_pairs:
            n1 = pair[0]
            x1 = self.mesh.nodes[0,n1]
            y1 = self.mesh.nodes[1,n1]
            n2 = pair[1]
            x2 = self.mesh.nodes[0,n2]
            y2 = self.mesh.nodes[1,n2]
            len23 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            ty = 0
            tx = 0
            if trac_dir == 'x':
                tx = trac_value
            elif trac_dir == 'y':
                ty = trac_value
            self.F_global[n1*2] += tx*len23/2
            self.F_global[n1*2 + 1] += ty*len23/2
            self.F_global[n2*2] += tx*len23/2
            self.F_global[n2*2 + 1] += ty*len23/2
    def apply_pointload(self, node_target, load_value, load_dir):
        """Applies a point force to a given node or node set."""
        for node in enumerate(node_target):
            if load_dir == 'x':
                self.F_global[node[1]*2] += load_value[node[0]]
            else:
                self.F_global[node[1]*2 + 1] += load_value[node[0]]
    def apply_pressure(self, node_list, load_value):
        """Applies pressure load to a set of nodes."""
        # Eliminate duplicates from list
        res = []
        [res.append(x) for x in node_list if x not in res]
        node_list = res
        node_pairs = []
        normal = []
        for node1 in node_list:
            for node2 in node_list:
                if node1 == node2:
                    break
                else:
                    for element in self.mesh.elements:
                        if node1 in element and node2 in element:
                            node_pairs.append([node1, node2])
                            break
        print(node_pairs)
        print(node_list)
        for pair in node_pairs:
            tangent = [self.mesh.nodes[0, pair[0]] - self.mesh.nodes[0, pair[1]],
                       self.mesh.nodes[1, pair[0]] - self.mesh.nodes[1, pair[1]]]
            magnitude = np.sqrt(tangent[0]**2 + tangent[1]**2)
            tangent = tangent/magnitude
            # compute normal and multiply by -1 to reverse direction toward element
            normal = np.multiply([-tangent[1], tangent[0]], -load_value)
            # Apply tractions for x and y components
            self.apply_traction([pair], normal[0], 'x')
            self.apply_traction([pair], normal[1], 'y')


##### Tensor and scalar quantities #####

class Nodal_quantity:
    """An object that stores the values of nodes."""
    def __init__(self, mesh: Mesh, number_of_components: int):
        self.mesh = mesh
        self.length = mesh.nodes.shape[1]
        self.num = number_of_components
        self.values = np.zeros((self.num, self.length))
    def update(self, new_values):
        self.values = new_values

class Displacement(Nodal_quantity):
    def __init__(self, mesh: Mesh, number_of_components = 2):
        super().__init__(mesh, number_of_components)
        self.length = self.length
        self.values = {'U1': np.zeros(self.length), 'U2': np.zeros(self.length)}
    def update(self, new_values):
        self.values['U1'], self.values['U2'] = np.array([new_values[range(0,self.length*2,2)], new_values[range(1,self.length*2,2)]])
    def return_all(self):
        return np.ravel(np.array([self.values['U1'], self.values['U2']]), order='F')

class delta_Displacement(Nodal_quantity):
    def __init__(self, mesh: Mesh, number_of_components = 2):
        super().__init__(mesh, number_of_components)
        self.length = self.length
        self.values = {'delU1': np.zeros(self.length), 'delU2': np.zeros(self.length)}
    def clear(self):
        self.values = {'delU1': np.zeros(self.length), 'delU2': np.zeros(self.length)}
    def update(self, new_values):
        self.values['delU1'], self.values['delU2'] = np.array([new_values[range(0,self.length*2,2)], new_values[range(1,self.length*2,2)]])
    def return_all(self):
        return np.ravel(np.array([self.values['delU1'], self.values['delU2']]), order='F')

class Elemental_quantity:
    """An object that stores the values of elements at the gauss points."""
    def __init__(self, mesh: Mesh, number_of_components: int):
        self.mesh = mesh
        self.length = mesh.gauss_points.shape[1]
        self.num = number_of_components
        self.values = np.zeros((self.num, self.length))
    def update(self, new_values):
        self.values = new_values
    def return_all(self):
        return self.values
    def transform(self):
        transformed_values = deepcopy(self)
        for i in range(self.mesh.gauss_points.shape[1]):
            x = self.mesh.gauss_points[0, i]
            y = self.mesh.gauss_points[1, i]
            r, theta = r_theta(x,y)
            sigma_x = self.values['S11'][i]
            sigma_y = self.values['S22'][i]
            sigma_xy = self.values['S12'][i]
            transformed_values.values['S11'][i] = sigma_x*np.cos(theta)**2 + sigma_y*np.sin(theta)**2 + sigma_xy*np.sin(2*theta)
            transformed_values.values['S22'][i] = sigma_x*np.sin(theta)**2 + sigma_y*np.cos(theta)**2 - sigma_xy*np.sin(2*theta)
            transformed_values.values['S12'][i] = np.sin(theta)*np.cos(theta)*(sigma_y - sigma_x) + sigma_xy*np.cos(2*theta)
        return transformed_values
    
class Stress(Elemental_quantity):
    """Stress tensor object."""
    def __init__(self, mesh: Mesh, number_of_components = 3):
        super().__init__(mesh, number_of_components)
        self.values = {'S11': np.zeros(self.length), 'S22': np.zeros(self.length), 'S12': np.zeros(self.length)}
        self.tensor = np.array([self.values['S11'], self.values['S22'], self.values['S12']])
    def update(self, new_values):
        self.values['S11'], self.values['S22'], self.values['S12'] = new_values
    def compute(self, U):
        newS = np.zeros((3, self.length))
        material = self.mesh.material
        for k in enumerate(self.mesh.elements):
            i = k[0]
            element = k[1]
            n1, n2, n3, n4 = element
            index = [n1*2, n1*2+1, n2*2, n2*2+1, n3*2, n3*2+1, n4*2, n4*2+1]
            U_element = U[index]
            gauss_index = np.arange(4*i, 4*i+4)
            for k in gauss_index:
                D = material.D_matrix()
                B = self.mesh.B[:,:,k]
                newS[:, k] = D@B@U_element
        self.values['S11'], self.values['S22'], self.values['S12'] = newS
    def return_all(self):
        return np.array([self.values['S11'], self.values['S22'], self.values['S12']])

class delta_Stress(Elemental_quantity):
    """Delta Stress tensor object."""
    def __init__(self, mesh: Mesh, number_of_components = 3):
        super().__init__(mesh, number_of_components)
        self.values = {'delS11': np.zeros(self.length), 'delS22': np.zeros(self.length), 'delS12': np.zeros(self.length)}
        self.tensor = np.array([self.values['delS11'], self.values['delS22'], self.values['delS12']])
    def update(self, new_values):
        self.values['delS11'], self.values['delS22'], self.values['delS12'] = new_values
    def clear(self):
        self.values = {'delS11': np.zeros(self.length), 'delS22': np.zeros(self.length), 'delS12': np.zeros(self.length)}
    def compute(self, S, delE, E):
        newS = np.zeros((3, self.length))
        material = self.mesh.material
        for k in enumerate(self.mesh.elements):
            i = k[0]
            element = k[1]
            n1, n2, n3, n4 = element
            index = [n1*2, n1*2+1, n2*2, n2*2+1, n3*2, n3*2+1, n4*2, n4*2+1]
            gauss_index = np.arange(4*i, 4*i+4)
            for k in gauss_index:
                D = material.D_matrix(S[:, k], E[:, k])
                newS[:, k] = D[:,:]@delE[:, k]
        self.values['delS11'], self.values['delS22'], self.values['delS12'] = newS
    def return_all(self):
        return np.array([self.values['delS11'], self.values['delS22'], self.values['delS12']])

class Strain(Elemental_quantity):
    """Strain tensor object."""
    def __init__(self, mesh: Mesh, number_of_components = 3):
        super().__init__(mesh, number_of_components)
        self.values = {'E11': np.zeros(self.length), 'E22': np.zeros(self.length), 'E12': np.zeros(self.length)}
        self.tensor = np.array([self.values['E11'], self.values['E22'], self.values['E12']])
    def update(self, new_values):
        self.values['E11'], self.values['E22'], self.values['E12'] = new_values
    def compute(self, delU):
        E = np.zeros((3, self.length))
        for k in enumerate(self.mesh.elements):
            i = k[0]
            element = k[1]
            n1, n2, n3, n4 = element
            index = [n1*2, n1*2+1, n2*2, n2*2+1, n3*2, n3*2+1, n4*2, n4*2+1]
            delU_element = delU[index]
            gauss_index = np.arange(4*i, 4*i+4)
            for k in gauss_index:
                B = self.mesh.B[:,:,k]
                E[:, k] = B@delU_element
        self.values['E11'], self.values['E22'], self.values['E12'] = E     
    def return_all(self):
        return np.array([self.values['E11'], self.values['E22'], self.values['E12']])
    
class delta_Strain(Elemental_quantity):
    """Delta strain tensor object."""
    def __init__(self, mesh: Mesh, number_of_components = 3):
        super().__init__(mesh, number_of_components)
        self.values = {'dE11': np.zeros(self.length), 'dE22': np.zeros(self.length), 'dE12': np.zeros(self.length)}
        self.tensor = np.array([self.values['dE11'], self.values['dE22'], self.values['dE12']])
    def update(self, new_values):
        self.values['dE11'], self.values['dE22'], self.values['dE12'] = new_values
    def clear(self):
        self.values = {'dE11': np.zeros(self.length), 'dE22': np.zeros(self.length), 'dE12': np.zeros(self.length)}
    def compute(self, delU):
        delE = np.zeros((3, self.length))
        for k in enumerate(self.mesh.elements):
            i = k[0]
            element = k[1]
            n1, n2, n3, n4 = element
            index = [n1*2, n1*2+1, n2*2, n2*2+1, n3*2, n3*2+1, n4*2, n4*2+1]
            delU_element = delU[index]
            gauss_index = np.arange(4*i, 4*i+4)
            for k in gauss_index:
                B = self.mesh.B[:,:,k]
                delE[:, k] = B@delU_element
        self.values['dE11'], self.values['dE22'], self.values['dE12'] = delE     
    def return_all(self):
        return np.array([self.values['dE11'], self.values['dE22'], self.values['dE12']])

class Boundary_condition:
    """Defines a boundary condition for the model."""
    def __init__(self, Kg: Global_K_matrix):
        self.K = Kg.K_global
    def apply_BC(self, nodes: np.ndarray, values: np.ndarray, DOF_component):
        if len(nodes) != len(values):
            raise Exception('DOF and value inputs for boundary condition must be the same length.')
        
        #Check if inputs are in numpy array format and convert if not.
        if type(nodes) != 'numpy.ndarray':
            nodes = np.array(nodes)
        if type(values) != 'numpy.ndarray':
            values = np.array(values)

        #Change node numbers to appropriate DOF
        if DOF_component == 'U1':
            new_DOFs = nodes*2
        elif DOF_component == 'U2':
            new_DOFs = nodes*2+1
        new_values = values

        # if self.DOFs does not yet exist, make it; otherwise append to existing array
        if hasattr(self, 'DOFs') is False:
            self.DOFs = np.array(new_DOFs)
        else:
            self.DOFs = np.append(self.DOFs, new_DOFs)
        
        # if self.values does not yet exist, make it; otherwise append to existing array
        if hasattr(self, 'values') is False:
            self.values = np.array(new_values)
        else:
            self.values = np.append(self.values, new_values)

        self.num_DOFs = len(self.DOFs)
        self.dim_Kglobal = self.K.shape[0]
        #Creates the C and Q matrices necessary for the Langrange multiplier approach.
        self.type = "Lagrange multipliers"
        self.C = np.zeros((self.num_DOFs, self.dim_Kglobal))
        self.Q = np.zeros(self.num_DOFs)
        for k in range(self.num_DOFs):
            self.C[k, self.DOFs[k]] = 1.0
            self.Q[k] = self.values[k]

class Solver:
    def __init__(self, solver_type):
        self.type = solver_type

class Standard(Solver):
    def __init__(self, Kg: Global_K_matrix, Tg: Global_T_matrix, Fg: Global_F_matrix, BC: Boundary_condition, 
                 S: Stress, E: Strain, U: Displacement, mesh: Mesh):
        self.mesh = mesh
        self.Kmat = Kg
        self.Fmat = Fg
        self.Tmat = Tg
        self.K = self.Kmat.K_global
        self.F = self.Fmat.F_global
        self.T = self.Tmat.T_global
        self.BC = BC
        self.S = S
        self.E = E
        self.U = U

        self.delS = delta_Stress(self.mesh)
        self.delE = delta_Strain(self.mesh)
        self.delQ = delta_Displacement(self.mesh)
        
        self.nq = np.zeros((self.K.shape[0]))
        self.nS = Stress(self.mesh)
        self.nE = Strain(self.mesh)

    def start(self, initial_stepsize = 1/100, end_steptime = 1):
        tol = 0.0001
        stepsize = initial_stepsize
        steptime = initial_stepsize
        endtime = end_steptime
        complete = 0
        while complete == 0:
            errorflag = 0
            self.delQ.clear()
            self.delE.clear()
            self.delS.clear()
            n_iteration = 0
            while steptime <= endtime and n_iteration <= 10:
                n_iteration += 1
                if errorflag == 1:
                    if steptime == endtime:
                        complete = 1
                        print('### Solve complete. ###')
                        break
                    else:
                        steptime += stepsize
                        if steptime > endtime:
                            steptime = endtime
                        break

                progress = steptime/endtime
                print('-----\n\n*** step time: %1.2f, iteration = %1i ***'%(steptime, n_iteration))

                self.delE.compute(self.delQ.return_all())
                self.nE.update(self.E.return_all() + self.delE.return_all())
                # print('ne', self.nE.values)
                # print('delE', self.delE.values)

                self.delS.compute(self.S.return_all(), self.delE.return_all(), self.E.return_all())
                self.nS.update(self.S.return_all() + self.delS.return_all())
                # print('ns', self.nS.values)
                # print('S', self.S.values)
                # print('delS', self.delS.values)

                # Construct K and T:
                self.Kmat.build(self.S, self.E)
                self.Tmat.build(self.nS)
                # Residual force vector:
                R = progress*self.F - self.Tmat.T_global


                # Construct BCs with [K C';C zeros(size(C,1))]\[R;Q]
                Q = progress*self.BC.Q

                a = np.block([[self.Kmat.K_global, self.BC.C.T],
                        [self.BC.C, np.zeros((self.BC.C.shape[0], self.BC.C.shape[0]))]])
                b = np.append(R, Q.T)

                # Solve
                self.nsol = np.linalg.solve(a, b)
                self.nq = self.nsol[0:self.K.shape[0]]
                self.constraint_force = self.nsol[self.K.shape[0]:len(self.nsol)]
                self.delQ.update(self.delQ.return_all() + self.nq)
                

                # Compute error
                # error = np.sqrt((self.nq.T@self.nq)/(np.transpose(self.U.return_all() + self.delQ.return_all())@(self.U.return_all() + self.delQ.return_all())))
                R[self.BC.DOFs] -= self.constraint_force #Add constraint forces back into the residual
                error = np.sqrt((R@R)/(self.F@self.F))
                # error = np.max(np.absolute(R))/np.mean(np.absolute(self.F))
                print('Force residual norm: %1.3E \nMax. R: %1.3E'%(error, np.max(R)))
                if error < tol:
                    errorflag = 1
                else:
                    errorflag = 0
        
            self.U.update(self.U.return_all() + self.delQ.return_all())
            self.S.update(self.nS.return_all())
            self.E.update(self.nE.return_all())



def plot_result(mesh: Mesh, result, component: str, U, deformed=True, avg_threshold = 0.75, plot_mesh = True):
    """Creates a contour plot of a Nodal_quantity or Element_quantity"""
    fig, ax = plt.subplots()

    zmin = result.values[component].min()
    zmax = result.values[component].max()

    if isinstance(result, Elemental_quantity):
        averaged = np.zeros((mesh.nodes.shape[1])) # Array to store nodal averages
        count = np.zeros((mesh.nodes.shape[1])) # Array to store the count of elements a node is attached to
        extrapolated = np.zeros((mesh.elements.shape[0], 4)) # also make an array to store the extrapolated node values of each element to use later.
        for element in enumerate(mesh.elements):
            extrapolated[element[0],:] = BESTFITQ(result.values[component][element[0]*4:element[0]*4+4])
            averaged[element[1]] += extrapolated[element[0],:]
            count[element[1]] += [1, 1, 1, 1]
        averaged = np.divide(averaged, count) # complete the average by dividing by the counts
        zmin = np.min(averaged)
        zmax = np.max(averaged)
        delz = np.abs(np.min(extrapolated) - np.max(extrapolated))
        # Spread out the min and max limits if they are too close.
        # if np.absolute(zmin - zmax) <= 1e-10:
        #     zmin = zmin*0.9
        #     zmax = zmax*1.1

    levels = MaxNLocator().tick_values(zmin, zmax)
    cmap = plt.colormaps['hsv_r']
    cmap = ListedColormap(cmap(np.linspace(0.31, 1, 256)))
    norm = BoundaryNorm(levels, ncolors=cmap.N)

    if deformed == True:
        node_positions_x = mesh.nodes[0,:] + U.values['U1']
        node_positions_y = mesh.nodes[1,:] + U.values['U2']
    else:
        node_positions_x = mesh.nodes[0,:]
        node_positions_y = mesh.nodes[1,:]

    for element in enumerate(mesh.elements):
        x1, x2, x3, x4 = node_positions_x[element[1]]
        y1, y2, y3, y4 = node_positions_y[element[1]]
        X = [[x4, x3], [x1, x2]]
        Y = [[y4, y3], [y1, y2]]

        if isinstance(result, Elemental_quantity):
            #check the average difference of values at each node
            dont_avg = []
            for node_num in enumerate(element[1]):
                # print(node_num, np.where(mesh.elements == node_num))
                index = np.where(mesh.elements == node_num[1])
                values = extrapolated[index]
                local_minmax = np.abs(np.min(values) - np.max(values))
                rel_avg = local_minmax/delz
                # print('rel_avg: ',rel_avg)
                if rel_avg > avg_threshold:
                    dont_avg.append(node_num[0])
            # print(dont_avg)
            Z = averaged[element[1]]
            # print('averaged: ',Z)
            # if node is on dont avg list, replace Z with etrapolated value
            for node in dont_avg:
                Z[node] = extrapolated[element[0],:][node]
                if Z[node] < zmin:
                    zmin = Z[node]
                elif Z[node] > zmax:
                    zmax = Z[node]            
            # print('subbed: ', Z)
            Z = [[Z[3], Z[2]], [Z[0], Z[1]]]
            
        elif isinstance(result, Nodal_quantity):
            Z = result.values[component][element[1]]
            Z = [[Z[3], Z[2]], [Z[0], Z[1]]]

        # Update levels in case zmin and zmax changed.
        levels = MaxNLocator().tick_values(zmin, zmax)
        im = ax.contourf(X, Y, Z, levels = levels, cmap=cmap, norm=norm)
        if plot_mesh == True:
            ax.fill([X[1][0], X[1][1], X[0][1], X[0][0]],
                     [Y[1][0], Y[1][1], Y[0][1], Y[0][0]],
                       edgecolor = 'black', facecolor = 'none')
    cbar = fig.colorbar(im, ticks=levels, drawedges=True, extend='both', extendrect=True, format='%.3E')
    if isinstance(result, Elemental_quantity):
        ax.set_title(component)
        fig.text(0.90,0, ' (Avg: {:3.0%})'.format(avg_threshold), ha='right', va='bottom')
    else:
        ax.set_title(component)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()



# Test code
start = timeit.default_timer()

mesher1 = Mesher()
# mesher1.set_params([2,2], [[3,3], [3,3]], mesher1.coords_quarterCircle(1), [3], [[4,7,4,5]], surfs=[[0,2], [0,6], [6,7], [5,2]])
mesher1.set_params([1,1], [[2],[2]], mesher1.coords_Quad(1, 1), surfs=[[0,1], [0,2], [2,3]])
mesher1.create()

mesher1.nodes[:,4] = np.array([[.6, .2]])

mesh1 = Mesh()
mesh1.make_mesh(mesher1)
# steel = Material_model([30e6, 0.30], "linear elastic, plane strain")
# mesh1.assign_material(steel)
strainlock = Material_model([0.55, 1e13, np.deg2rad(90), 1e5, 0.1, 1e5, 0.1, 1e5], "strain locking, plane strain")
mesh1.assign_material(strainlock)
K = Global_K_matrix(mesh1)
mesh1.plot()

E = Strain(mesh1)
dE = delta_Strain(mesh1)
S = Stress(mesh1)
U = Displacement(mesh1)
T = Global_T_matrix(mesh1)
F = Global_F_matrix(mesh1)

topsurf1 = mesher1.surfs[2]
bottomsurf = mesher1.surfs[0]
sidesurf = mesher1.surfs[1]

BC1 = Boundary_condition(K)
BC1.apply_BC(sidesurf, np.zeros(len(sidesurf)), 'U1')
BC1.apply_BC(bottomsurf, np.zeros(len(sidesurf)), 'U2')

F.apply_traction(topsurf1, 1e8, 'y')
             
solution = Standard(K, T, F, BC1, S, E, U, mesh1)
solution.start()

stop = timeit.default_timer()
print('Elapsed time: ', f'{stop-start} seconds.')

# E.compute(U.return_all())
# S.compute(U.return_all())

plot_result(mesh1, S, 'S22', U)
# plot_result(mesh1, S, 'S11', U)
# plot_result(mesh1, S, 'S12', U)

plot_result(mesh1, E, 'E22', U)
plot_result(mesh1, E, 'E11', U)
plot_result(mesh1, E, 'E12', U)

plot_result(mesh1, U, 'U1', U)
plot_result(mesh1, U, 'U2', U)