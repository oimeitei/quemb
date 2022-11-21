def printatom(l1,unit=0.52917721092 ):
    print(len(l1))
    print()
    for i in l1:
        print(i[0],end='     ')
        for j in i[1]:
            print(j*unit,end='    ')
        print()


def kpts_to_kmesh(cell, kpts):
    import numpy as np
    
    '''Guess kmesh'''
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = (len(np.unique(scaled_k[:,0])),
             len(np.unique(scaled_k[:,1])),
             len(np.unique(scaled_k[:,2])))
    return kmesh


def sgeom(cell, kpts=None, kmesh=None):
    from pyscf.pbc import tools
    
    if kmesh is None:
        if kpts is None:
            print('Provide kpts or kmesh(kpt)',flush=True)
            sys.exit()
        kmesh = kpts_to_kmesh(cell, kpts)
        
    scell = tools.super_cell(cell, kmesh)

    return scell
