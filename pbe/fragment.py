import numpy, sys
from .helper import get_core
from .autofrag import autogen


class fragpart:

    def __init__(self, natom=0, dim=1, frag_type='hchain_simple',
                 unitcell=1,auxcell=None,
                 nx=False, ny=False, nz=False,closed=False,
                 kpt = None, molecule=True,valence_basis=None,valence_only=False,
                 be_type='be2', mol=None, frozen_core=False, self_match=False, allcen=False):

        # No. of unitcells to use for fragment construction
        self.unitcell = unitcell
        self.molecule = molecule
        self.mol = mol
        self.frag_type=frag_type
        self.fsites = []
        self.Nfrag = 0
        self.edge= []
        self.center = []
        self.ebe_weight = []
        self.edge_idx = []
        self.center_idx = []
        self.centerf_idx = []
        self.be_type = be_type
        self.natom = natom
        self.frozen_core = frozen_core
        self.self_match = self_match
        self.allcen=allcen
        self.valence_basis = valence_basis
        self.valence_only = valence_only
        if frozen_core:
            self.ncore, self.no_core_idx, self.core_list = get_core(mol)
        if frag_type=='hchain_simple':
            self.hchain_simple()
        elif frag_type=='chain':
            if mol is None:
                print('Provide pyscf gto.M object in fragpart() and restart!',
                      flush=True)
                print('exiting',flush=True)
                sys.exit()
            self.chain(mol, frozen_core=frozen_core,closed=closed)
            
        elif frag_type=='autogen':
            if mol is None:
                print('Provide pyscf gto.M object in fragpart() and restart!',
                      flush=True)
                print('exiting',flush=True)
                sys.exit()
            if not molecule:
                if kpt is None:
                    print('Provide kpt mesh in fragpart() and restart!',
                          flush=True)
                    print('exiting',flush=True)
                    sys.exit()
                    
            fgs = autogen(mol, kpt, be_type=be_type, frozen_core=frozen_core,
                          valence_basis=valence_basis, unitcell=unitcell, valence_only=valence_only,
                          nx=nx, ny=ny, nz=nz,#auxcell=auxcell,
                          molecule=molecule)
            self.fsites, self.edge, self.center, self.edge_idx, self.center_idx, self.centerf_idx, self.ebe_weight = fgs
                
            self.Nfrag = len(self.fsites)
            
        else:
            print('Fragmentation type = ',frag_type,' not implemented!',
                  flush=True)
            print('exiting',flush=True)
            sys.exit()   
    from .lchain import chain
    def hchain_simple(self):
        
        if self.be_type=='be1':
            for i in range(self.natom):
                self.fsites.append([i])
            self.Nfrag = len(self.fsites)
                
        elif self.be_type=='be2':
            for i in range(self.natom-2):
                self.fsites.append([i, i+1, i+2])
                self.centerf_idx.append([1])
            self.Nfrag = len(self.fsites)

            self.edge.append([[2]])
            for i in self.fsites[1:-1]:
                self.edge.append([[i[0]],[i[-1]]])
            self.edge.append([[self.fsites[-1][0]]])
                        
            self.center.append([1])
            for i in range(self.Nfrag-2):
                self.center.append([i,i+2])
            self.center.append([self.Nfrag-2])
                     
        elif self.be_type=='be3':
            for i in range(self.natom-4):
                self.fsites.append([i, i+1, i+2, i+3, i+4])
                self.centerf_idx.append([2])
            self.Nfrag = len(self.fsites)
                
            self.edge.append([[3],[4]])
            for i in self.fsites[1:-1]:
                self.edge.append([[i[0]],[i[1]],[i[-2]],[i[-1]]])
            self.edge.append([[self.fsites[-1][0]],[self.fsites[-1][1]]])
            
            self.center.append([1,2])
            self.center.append([0,0,2,3])
            for i in range(self.Nfrag-4):
                self.center.append([i,i+1, i+3,i+4])
            
            self.center.append([self.Nfrag-4,self.Nfrag-3,
                                self.Nfrag-1,self.Nfrag-1])
            self.center.append([self.Nfrag-3,self.Nfrag-2])   
                
        for ix, i in enumerate(self.fsites):
            tmp_ = []
            elist_ = [ xx for yy in self.edge[ix] for xx in yy]
            for j in i:
                if not j in elist_: tmp_.append(i.index(j))
            self.ebe_weight.append([1.0, tmp_])
            
                
        if not self.be_type=='be1':
            for i in range(self.Nfrag):
                idx = []
                for j in self.edge[i]:
                    
                    idx.append([self.fsites[i].index(k) for k in j])
                self.edge_idx.append(idx)
                        
            for i in range(self.Nfrag):
                idx = []
                for j in range(len(self.center[i])):
                    idx.append([self.fsites[self.center[i][j]].index(k)
                                for k in self.edge[i][j]])
                self.center_idx.append(idx)

