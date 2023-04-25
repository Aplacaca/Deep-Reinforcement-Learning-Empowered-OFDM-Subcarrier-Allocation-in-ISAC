import numpy as np
import random
import pdb
from copy import deepcopy
class UE(object):
    def __init__(self, id,f_c=30*1e9,delta_f=120*1e3,N=8):
        # Channel Generation
        self.N = N
        self.delta_f = delta_f
        self.f_c = f_c
        self.d,self.h,self.h_r = self._generate_channel()
        
    def _generate_channel(self,):
        d = np.random.rand()*10.0+0.5 # Range
        theta = np.random.rand()*np.pi # Azimuth Angle
        f_series = np.expand_dims(np.linspace(1,self.N+1,self.N)*self.delta_f+self.f_c, axis=0)
        h = (3*10e8/(4*np.pi*d*f_series)) # SQRT PATHLOSS
        h_r = (3*10e8/(4*np.pi*(d**2)*f_series)) # SQRT PATHLOSS
        return d,h,h_r
    
    def _update_channel(self,):
        self.d,self.h,self.h_r = self._generate_channel()


class ISAC_BS(object):
    def __init__(self, N=8, N_c=8, N_r=8, seed=777):
        # Seed Configuration
        np.random.seed(seed)
        random.seed(seed)
        # Subcarrier Configurations
        self.N = N # Number of Subcarriers
        self.delta_f = 30*1e3 # Subcarrier Spacing (Hz)
        self.f_c = 6*1e9 # Carrier Frequency (Hz)
        self.S = 14 # Number of Symbols Per Slot
        self.T_s = 33.3*1e-6 # Duration Per Symbol (Guarded) (s)
        self.T_0 = 0.5*1e-3 # Minimum Scheduling Cycle (s)
        self.C = 1000 # Scheduling Cycles During Interested Time Window
        self.T_tot = self.C*self.T_0 # Scheduling Cycles During Interested Time Window (s)
        # UE Generate
        self.M_c = [UE(id="c"+str(i),f_c=self.f_c,delta_f = self.delta_f,N=self.N) for i in range(N_c)] # Communication Users
        self.M_r = [UE(id="r"+str(i),f_c=self.f_c,delta_f = self.delta_f,N=self.N) for i in range(N_r)] # Sensing Targets
        self.N_c = N_c # Number of Communication Users
        self.N_r = N_r # Number of Sensing Targets
        # Subcarrier Scheduling Scheme
        self.U = np.ones((N_c,N))
        self.D = np.ones((N_r,N))
        # Power Configurations
        self.P_r = np.diag(np.ones(N))*1.0 # Power of Sensing
        self.P_c = np.diag(np.ones(N))*0.5 # Power of Communication
        self.P_n = 1e-10 # Power of Background Noise
        # ENV Configurations
        self.time = 0
        self.H_c = self.get_H_c()
        self.H_r,self.H_cr = self.get_H_r()
        self.EE_c_s = np.zeros(self.C)
        self.EE_r_s = np.zeros(self.C)
        self.reward = 0.0

    def get_H_c(self,):
        H_c = np.concatenate([UE.h for UE in self.M_c],axis=0)
        # print(H_c.shape)
        return H_c
    
    def get_H_r(self,):
        H_r = np.concatenate([UE.h_r for UE in self.M_r],axis=0)
        H_cr = np.concatenate([UE.h_r for UE in self.M_c],axis=0)
        # print(H_c.shape)
        return H_r,H_cr
    
    def update_channels(self,):
        # Update Communication UE Channel
        for UE in self.M_c:
            UE._update_channel()
        # Update Sensing UE Channel
        for UE in self.M_r:
            UE._update_channel()
        # Recat Channels
        self.H_c = self.get_H_c()
        self.H_r,self.H_cr = self.get_H_r()
        
    def _get_comm_sinr(self,):
        # Calculate Initial Power On Each Subcarrier
        P_C = self.U@self.P_c
        P_R = self.D@self.P_r
        IP_R = np.sum(P_R, axis=0)
        SUM_P_C = np.sum(P_C, axis=0)
        SINR = np.zeros((self.N_c,self.N))
        for k in range(self.N_c):
            H_C = np.power(self.H_c[k,:],2)
            IP_C = SUM_P_C - P_C[k,:]
            P_useful = H_C*P_C[k,:]
            P_in = H_C*(IP_R+IP_C)
            SINR[k,:] = P_useful/(P_in+self.P_n)
        return SINR    
    
    def _get_sensing_sinr(self,):
        # Calculate Initial Power On Each Subcarrier
        HP_C = self.U@self.P_c*np.power(self.H_cr,2)*self.S*(self.T_s)**2
        HP_R = self.D@self.P_r*np.power(self.H_r,2)*self.S*(self.T_s)**2
        IP_H_C = np.sum(HP_C, axis=0)
        SUM_HP_R = np.sum(HP_R, axis=0)
        SINR = np.zeros((self.N_r,self.N))
        for l in range(self.N_r):
            P_useful = HP_R[l,:]
            IHP_R = SUM_HP_R - HP_R[l,:]
            P_in = (IHP_R+IP_H_C)
            SINR[l,:] = P_useful/(P_in+self.P_n*self.S*self.T_s)
        return SINR
    
    def _random_allocation(self,):
        action_u = np.random.randint(self.N_c,size=self.N)
        action_d = np.random.randint(self.N_r,size=self.N)
        U_SET = np.eye(self.N_c)
        D_SET = np.eye(self.N_r)
        self.U = U_SET[:,action_u]
        self.D = D_SET[:,action_d]
        
    def _naive_allocation(self,):
        # First we allocate CUs
        # action_u = np.random.randint(self.N_c,size=self.N)
        action_u = np.argsort(-self.H_c,axis=0)[:,0]
        U_SET = np.eye(self.N_c)
        self.U = U_SET[:,action_u]
        # Then we assign SUs to CUs with smaller H_cr
        H_cr_list = np.argsort(self.H_cr,axis=0)[:,0]
        matched_su = np.argsort(self.H_r,axis=0)[:,0]
        matched_cu = H_cr_list[:self.N_r]
        # 
        self.D = np.zeros((self.N_r,self.N))
        for l in range(matched_su.shape[0]):
            self.D[matched_su[l],:] = deepcopy(self.U[matched_cu[l],:])
        
    def get_performance(self, print_log=False):
        # Calculate SINR On Each Subcarrier
        C_SINR = self._get_comm_sinr()
        R_SINR = self._get_sensing_sinr()
        # Cal COMM RATE
        SUM_R_c = self.delta_f*np.sum(np.log2(1+C_SINR))
        # Cal Sensing SUM MI
        SUM_MI_r = 0.5*self.S*self.T_s*self.delta_f*np.sum(np.log2(1+R_SINR))
        # Energy Efficiency
        SUM_R_P = np.sum(self.D@self.P_r)
        SUM_C_P = np.sum(self.U@self.P_c)
        if abs(SUM_C_P-0.0)>1e-4:
            EE_C = SUM_R_c/SUM_C_P
        else:
            EE_C = 0.0
        if abs(SUM_R_P-0.0)>1e-4:
            EE_R = SUM_MI_r/SUM_R_P
        else:
            EE_R = 0.0
        if print_log:
            print("C_SINR(dB): ",10*np.log10(C_SINR))
            print("R_SINR(dB): ",10*np.log10(R_SINR))
            print("SUM_R_c: ",SUM_R_c," EE: ",EE_C)
            print("SUM_MI_r: ",SUM_MI_r," EE: ",EE_R)
        return SUM_R_P,SUM_C_P,SUM_R_c,SUM_MI_r,EE_C,EE_R
    
    def reward_to_go(self,EE_C,EE_R):
        # mean_EE_c = np.mean(self.EE_c_s[0:self.time+1])
        # mean_EE_r = np.mean(self.EE_r_s[0:self.time+1])
        # return (mean_EE_c/(1e7)+mean_EE_r/(100))
        return (EE_C/(5*1e4)+EE_R/10.0)

    def _penalty(self,):
        if np.any(np.sum(self.U,axis=0) > 1) or np.any(np.sum(self.D,axis=0) > 1):
            return True
        else:
            return False
        
    # def _get_state(self,):
    #     H_c = self.get_H_c()    
    #     H_r,H_cr = self.get_H_r()
    #     H_c = H_c.flatten()
    #     H_r = H_r.flatten()
    #     H_cr = H_cr.flatten()
    #     U_SUM = np.sum(self.U,axis = -1)
    #     D_SUM = np.sum(self.D,axis = -1)
    #     SCHED = np.concatenate([U_SUM,D_SUM],axis = 0).flatten()
    #     state = np.concatenate([H_c,H_r,H_cr])
    #     return state
    
    def _get_state(self,):
        H_c = self.get_H_c()    
        H_r,H_cr = self.get_H_r()
        H_c = H_c[:,0].flatten()
        H_r = H_r[:,0].flatten()
        H_cr = H_cr[:,0].flatten()
        U_SUM = np.sum(self.U,axis = -1)
        D_SUM = np.sum(self.D,axis = -1)
        SCHED = np.concatenate([U_SUM,D_SUM],axis = 0).flatten()
        state = np.concatenate([H_c,H_r,H_cr])
        return state
    
    def step(self,action=None,freeze=False):
        done = 0
        # DO ACTION
        self._naive_allocation()
        # self._random_allocation()
        _,_,_,_,bl_random_EE_C,bl_random_EE_R = self.get_performance()
        if action is not None:
            action_u = np.argsort(-self.H_c,axis=0)[:,0]
            U_SET = np.eye(self.N_c)
            self.U = U_SET[:,action_u]
            
            # action_u = action[:self.N_c]
            # U_SET = np.eye(self.N_c)
            # D_SET = np.eye(self.N_r)
            # self.U = U_SET[:,action_u]
            self.D = deepcopy(self.U[action,:])
        # 
        SUM_R_P,SUM_C_P,SUM_R_c,SUM_MI_r,EE_C,EE_R = self.get_performance()
        self.EE_c_s[self.time] = EE_C
        self.EE_r_s[self.time] = EE_R
        # reward = (self._penalty()==0)*5 + (self._penalty()==1)*-5
        reward = self.reward_to_go(EE_C,EE_R)
        # reward = self.reward_to_go(EE_C,EE_R) + (self._penalty()==0)*-5
        bl_random_reward = self.reward_to_go(bl_random_EE_C,bl_random_EE_R) 
        if not freeze:
            self.update_channels()
        self.time += 1
        if self.time % self.C == 0:
            done = 1
            next_state,_ = self.reset(freeze=freeze)
            return next_state,reward,done,bl_random_reward
        next_state = self._get_state()
        return next_state,reward,done,bl_random_reward
    
    def reset(self,freeze=False):
        done = 0
        self.time = 0
        self.U = np.ones((self.N_c,self.N))
        self.D = np.ones((self.N_r,self.N))
        self.EE_c_s = np.zeros(self.C)
        self.EE_r_s = np.zeros(self.C)
        if not freeze:
            self.update_channels()
        next_state = self._get_state()
        return next_state,done
        