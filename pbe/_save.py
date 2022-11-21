import pickle



# pickle

class storePBE:


    def __init__(self, Nocc, hf_veff, hcore,
                 S, C, hf_dm, hf_etot, W, lmo_coeff,
                 enuc, ek):


        self.Nocc = Nocc
        self.hf_veff = hf_veff
        self.hcore = hcore
        self.S = S
        self.C = C
        self.hf_dm = hf_dm
        self.hf_etot = hf_etot
        self.W = W
        self.lmo_coeff = lmo_coeff
        self.enuc = enuc
        self.ek = ek
