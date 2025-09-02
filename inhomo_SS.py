# fdtd/inhomo_SS.py
from .config import *  # <-- add this line FIRST

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN_mLor_SS_inhomo(nn.Module):
  def __init__(self):
    super().__init__()

    self.register_buffer('Ez', torch.zeros(txsize + 1, tysize + 1, device=device))
    self.register_buffer('Hx', torch.zeros(txsize + 1, tysize, device=device))
    self.register_buffer('Hy', torch.zeros(txsize, tysize + 1, device=device))
    self.register_buffer('Jz', torch.zeros(tot_pol, txsize + 1, tysize + 1, device=device))
    self.register_buffer('Jz_p', torch.zeros(tot_pol, txsize + 1, tysize + 1, device=device))
    self.register_buffer('Ez_p', torch.zeros(txsize + 1, tysize + 1, device=device))
    self.register_buffer('Ez_pp', torch.zeros(txsize + 1, tysize + 1, device=device))

    self.register_buffer('fzx1', torch.zeros(PMLX + 1, tysize + 1, device=device))
    self.register_buffer('fzx2', torch.zeros(PMLX + 1, tysize + 1, device=device))
    self.register_buffer('fzy1', torch.zeros(txsize + 1, PMLY + 1, device=device))
    self.register_buffer('fzy2', torch.zeros(txsize + 1, PMLY + 1, device=device))
    self.register_buffer('gyx1', torch.zeros(PMLX, tysize + 1, device=device))
    self.register_buffer('gyx2', torch.zeros(PMLX, tysize + 1, device=device))
    self.register_buffer('gxy1', torch.zeros(txsize + 1, PMLY, device=device))
    self.register_buffer('gxy2', torch.zeros(txsize + 1, PMLY, device=device))

    self.register_buffer('Cxa1', torch.zeros(PMLX + 1, device=device))
    self.register_buffer('Cxa2', torch.zeros(PMLX + 1, device=device))
    self.register_buffer('Cxb1', torch.zeros(PMLX + 1, device=device))
    self.register_buffer('Cxb2', torch.zeros(PMLX + 1, device=device))
    self.register_buffer('Cya1', torch.zeros(PMLY + 1, device=device))
    self.register_buffer('Cya2', torch.zeros(PMLY + 1, device=device))
    self.register_buffer('Cyb1', torch.zeros(PMLY + 1, device=device))
    self.register_buffer('Cyb2', torch.zeros(PMLY + 1, device=device))

    self.register_buffer('Dxa1', torch.zeros(PMLX, device=device))
    self.register_buffer('Dxa2', torch.zeros(PMLX, device=device))
    self.register_buffer('Dxb1', torch.zeros(PMLX, device=device))
    self.register_buffer('Dxb2', torch.zeros(PMLX, device=device))
    self.register_buffer('Dya1', torch.zeros(PMLY, device=device))
    self.register_buffer('Dya2', torch.zeros(PMLY, device=device))
    self.register_buffer('Dyb1', torch.zeros(PMLY, device=device))
    self.register_buffer('Dyb2', torch.zeros(PMLY, device=device))

    self.register_buffer('Ca', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Cb', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Ca_temp', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Cb_temp', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Cc', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Cd', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('Ce', torch.zeros(txsize + 1, tysize + 1, tot_pol, device=device))
    self.register_buffer('C1', torch.ones(txsize + 1, tysize + 1, device=device))
    self.register_buffer('C2', torch.zeros(txsize + 1, tysize + 1, device=device))
    self.register_buffer('Cb_dx', torch.ones(txsize + 1, tysize + 1, device=device))
    self.register_buffer('Cb_dy', torch.ones(txsize + 1, tysize + 1, device=device))
    self.register_buffer('Db_dx', torch.ones(txsize + 1, tysize, device=device))
    self.register_buffer('Db_dy', torch.ones(txsize, tysize + 1, device=device))

    self.register_buffer('kappa_e_xprof', torch.ones(txsize, device=device))
    self.register_buffer('kappa_e_yprof', torch.ones(tysize, device=device))
    self.register_buffer('kappa_h_xprof', torch.ones(txsize, device=device))
    self.register_buffer('kappa_h_yprof', torch.ones(tysize, device=device))

    self.register_buffer('pole', torch.zeros((max_pol, 2), dtype=torch.float32, device=device))
    self.register_buffer('residue', torch.zeros((max_pol, 2), dtype=torch.float32, device=device))
    self.register_buffer('a0', torch.ones(tot_pol, dtype=torch.float32, device=device))
    self.register_buffer('a1', torch.ones(tot_pol, dtype=torch.float32, device=device))
    self.register_buffer('b0', torch.ones(tot_pol, dtype=torch.float32, device=device))
    self.register_buffer('b1', torch.ones(tot_pol, dtype=torch.float32, device=device))
    self.register_buffer('b2', torch.ones(tot_pol, dtype=torch.float32, device=device))
            
    self.register_buffer('PMLE',torch.ones(txsize + 1, tysize + 1, dtype=torch.float32, device=device))

    self.to(device)

  def param(self):

    # # fat Table
    # self.pole[2, 0] = -8.374186e+08    # real
    # self.pole[2, 1] = 3.032112e+09    # imag
    # self.residue[2, 0] = 5.923876e+07  # real
    # self.residue[2, 1] = -1.122860e+07 # imag
    # self.pole[3, 0] = -3.029324e+08    # real
    # self.pole[3, 1] = 1.419929e+10     # imag
    # self.residue[3, 0] = 3.990040e+06  # real
    # self.residue[3, 1] = 5.144260e+06 # imag
    # self.pole[0, 0] = -6.700849e+07    # real
    # self.pole[0, 1] = 0     # imag
    # self.residue[0, 0] = 4.420158e+09  # real
    # self.residue[0, 1] = 0 # imag
    # self.pole[1, 0] = -3.737597e+10    # real
    # self.pole[1, 1] = 0     # imag
    # self.residue[1, 0] = 5.082420e+10  # real
    # self.residue[1, 1] = 0 # imag
      
    # # Au Table
    # self.pole[0, 0] = -5.6511e+05    # real
    # self.pole[0, 1] = -1.4394e+14     # imag
    # self.residue[0, 0] = 3.5070e+15  # real
    # self.residue[0, 1] = 5.76374e+17 # imag
    # self.pole[1, 0] = -2.6075e+05    # real
    # self.pole[1, 1] = -1.5757e+15     # imag
    # self.residue[1, 0] = 2.0574e+14  # real
    # self.residue[1, 1] = 1.0700e+14 # imag
    # self.pole[2, 0] = -3.2320e+14    # real
    # self.pole[2, 1] = -3.8109e+15     # imag
    # self.residue[2, 0] = 4.7956e+14  # real
    # self.residue[2, 1] = -3.2670e+13 # imag
    # self.pole[3, 0] = -1.5803e+15    # real
    # self.pole[3, 1] = -3.5830e+15     # imag
    # self.residue[3, 0] = 9.3313e+15  # real
    # self.residue[3, 1] = -4.4765e+14 # imag
      
    # # Ag Table
    # self.pole[0, 0] = -4.8113e+12    # real
    # self.pole[0, 1] = 6.8512e+13     # imag
    # self.residue[0, 0] = 4.9581e+14  # real
    # self.residue[0, 1] = -1.511e+18 # imag

    # TiO2 Table
    self.pole[0, 0] = -6.112383e+00    # real
    self.pole[0, 1] = 5.701421e+15     # imag
    self.residue[0, 0] = 1.271769e+01  # real
    self.residue[0, 1] = -4.000896e+14 # imag
    self.pole[1, 0] = -1.042920e+01    # real
    self.pole[1, 1] = 7.383881e+15     # imag
    self.residue[1, 0] = 3.934968e+00  # real
    self.residue[1, 1] = -7.489729e+15 # imag

    for pol in range(real_pol // 2):
        self.a0[pol] = -(self.residue[2 * pol,0]) * (self.pole[2 * pol + 1,0]) - (self.pole[2 * pol,0]) * (self.residue[2 * pol + 1,0])
        self.a1[pol] = (self.residue[2 * pol,0]) + (self.residue[2 * pol + 1,0])
        self.b0[pol] = (self.pole[2 * pol,0]) * (self.pole[2 * pol + 1,0])
        self.b1[pol] = -((self.pole[2 * pol,0]) + (self.pole[2 * pol + 1,0]))
        self.b2[pol] = 1
    
    for pol in range(real_pol // 2, min(tot_pol, len(self.pole) - real_pol // 2)):
        
        self.a0[pol] = -2 * (self.pole[pol + real_pol // 2,0] * self.residue[pol + real_pol // 2,0] + self.pole[pol + real_pol // 2,1] * self.residue[pol + real_pol // 2,1])
        self.a1[pol] = 2 * (self.residue[pol + real_pol // 2,0])
        self.b0[pol] = self.pole[pol + real_pol // 2,0] * self.pole[pol + real_pol // 2,0] + self.pole[pol + real_pol // 2,1] * self.pole[pol + real_pol // 2,1]
        self.b1[pol] = -2 * (self.pole[pol + real_pol // 2,0])
        self.b2[pol] = 1
        
    num_freq = 50
    frequency_NFFF = torch.linspace(400e12, 800e12, num_freq, device = device)
    jw = 1j * frequency_NFFF[:, None] * 2 * math.pi

    eps_r_freq = torch.sum((self.a0 + jw * self.a1) / (self.b0 + jw * self.b1 + jw ** 2 * self.b2), dim=1)
    eps_r_freq += epsr_inf

    sigma = -eps0 * 2 * math.pi * frequency_NFFF * eps_r_freq.imag

    self.Db_dx[:, :] = dt/(mu0*mur)/dx
    self.Db_dy[:, :] = dt/(mu0*mur)/dy

      
    self.Ca[MX_start:MX_end, MY_start:MY_end, :] = -(2 * self.b0[:] * dt * dt - 8 * self.b2[:]) / (self.b0[:] * dt * dt + 2 * self.b1[:] * dt + 4 * self.b2[:])
    self.Cb[MX_start:MX_end, MY_start:MY_end, :] = -(self.b0[:] * dt * dt - 2 * self.b1[:] * dt + 4 * self.b2[:]) / (self.b0[:] * dt * dt + 2 * self.b1[:] * dt + 4 * self.b2[:])
    self.Cc[MX_start:MX_end, MY_start:MY_end, :] = (2 * self.a0[:] * eps0 * dt + 4 * self.a1[:] * eps0) / (self.b0[:] * dt * dt + 2 * self.b1[:] * dt + 4 * self.b2[:])
    self.Cd[MX_start:MX_end, MY_start:MY_end, :] = -(8 * self.a1[:] * eps0) / (self.b0[:] * dt * dt + 2 * self.b1[:] * dt + 4 * self.b2[:])
    self.Ce[MX_start:MX_end, MY_start:MY_end, :] = (-2 * self.a0[:] * eps0 * dt + 4 * self.a1[:] * eps0) / (self.b0[:] * dt * dt + 2 * self.b1[:] * dt + 4 * self.b2[:])

    temp1 = torch.sum(self.Cc[MX_start, MY_start, :]).to(device)
    temp2 = torch.sum(self.Cd[MX_start, MY_start, :]).to(device)
    temp3 = torch.sum(self.Ce[MX_start, MY_start, :]).to(device)

    self.PMLE[:, :] = dt / eps0 / 2.
    self.PMLE[MX_start:MX_end, MY_start:MY_end] = 1. / (eps_inf / dt + temp1 * 0.5) / 2.

    self.Ca_temp[MX_start:MX_end, MY_start:MY_end, :] = (self.Ca[MX_start:MX_end, MY_start:MY_end, :] + 1.0) * 0.5 / (eps_inf / dt + temp1 * 0.5)
    self.Cb_temp[MX_start:MX_end, MY_start:MY_end, :] = self.Cb[MX_start:MX_end, MY_start:MY_end, :] * 0.5 / (eps_inf / dt + temp1 * 0.5)

    self.Cb_dy[:, :] = dt / eps0 / dy
    self.Cb_dx[:, :] = dt / eps0 / dx
    self.Cb_dy[MX_start:MX_end, MY_start:MY_end] = 1 / (eps_inf / dt + temp1 * 0.5) / dy
    self.Cb_dx[MX_start:MX_end, MY_start:MY_end] = 1 / (eps_inf / dt + temp1 * 0.5) / dx

    # self.C1[:, :] = 1
    # self.C2[:, :] = 0
    self.C1[MX_start:MX_end, MY_start:MY_end] = (eps_inf / dt - temp2 * 0.5) / (eps_inf / dt + temp1 * 0.5)
    self.C2[MX_start:MX_end, MY_start:MY_end] = temp3 * 0.5 / (eps_inf / dt + temp1 * 0.5)


  def pml_param(self,PMLX,PMLY):
    m = 4
    dz = 1e-3
    Eps = 1
    sigma_opt = 0.8 * (m + 1) / (eta0 * dx * math.sqrt(Eps))
    sigma_max = sigma_opt
    alpha_max = 1e-12
    kappa_max = 15

    sigma_e_xprof = torch.zeros(PMLX + 1, device=device)
    sigma_h_xprof = torch.zeros(PMLX + 1, device=device)
    sigma_e_yprof = torch.zeros(PMLY + 1, device=device)
    sigma_h_yprof = torch.zeros(PMLY + 1, device=device)

    kappa_e_xprof_cal = torch.zeros(PMLX + 1, device=device)
    kappa_h_xprof_cal = torch.zeros(PMLX + 1, device=device)
    kappa_e_yprof_cal = torch.zeros(PMLY + 1, device=device)
    kappa_h_yprof_cal = torch.zeros(PMLY + 1, device=device)

    alpha_e_xprof = torch.zeros(PMLX + 1, device=device)
    alpha_h_xprof = torch.zeros(PMLX + 1, device=device)
    alpha_e_yprof = torch.zeros(PMLY + 1, device=device)
    alpha_h_yprof = torch.zeros(PMLY + 1, device=device)
      
    self.kappa_e_xprof[:txsize] = 1
    self.kappa_h_xprof[:txsize] = 1
    self.kappa_e_yprof[:tysize] = 1
    self.kappa_h_yprof[:tysize] = 1

    # x-direction
    index_PMLX = torch.arange(0, PMLX + 1)
    index_PMLY = torch.arange(0, PMLY + 1)

    sigma_e_xprof[:PMLX + 1] = sigma_max * ((index_PMLX + 0.0) / PMLX) ** m
    sigma_h_xprof[:PMLX + 1] = sigma_max * ((index_PMLX + 1.0 / 2.0) / PMLX) ** m

    kappa_e_xprof_cal[:PMLX + 1] = 1.0 + (kappa_max - 1.0) * ((index_PMLX + 0.0) / PMLX) ** m
    kappa_h_xprof_cal[:PMLX + 1] = 1.0 + (kappa_max - 1.0) * ((index_PMLX + 1.0 / 2.0) / PMLX) ** m

    alpha_e_xprof[:PMLX + 1] = alpha_max * (1.0 - (index_PMLX + 0.0) / PMLX)
    alpha_h_xprof[:PMLX + 1] = alpha_max * (1.0 - (index_PMLX + 1.0 / 2.0) / PMLX)

    # y-direction
    sigma_e_yprof[:PMLY + 1] = sigma_max * ((index_PMLY + 0.0) / PMLY) ** m
    sigma_h_yprof[:PMLY + 1] = sigma_max * ((index_PMLY + 1.0 / 2.0) / PMLY) ** m

    kappa_e_yprof_cal[:PMLY + 1] = 1.0 + (kappa_max - 1.0) * ((index_PMLY + 0.0) / PMLY) ** m
    kappa_h_yprof_cal[:PMLY + 1] = 1.0 + (kappa_max - 1.0) * ((index_PMLY + 1.0 / 2.0) / PMLY) ** m

    alpha_e_yprof[:PMLY + 1] = alpha_max * (1.0 - (index_PMLY + 0.0) / PMLY)
    alpha_h_yprof[:PMLY + 1] = alpha_max * (1.0 - (index_PMLY + 1.0 / 2.0) / PMLY)

    # using recursive index
    self.kappa_e_xprof[1:PMLX + 1] = kappa_e_xprof_cal[:PMLX].flip(0)
    self.kappa_e_xprof[txsize - PMLX:txsize] = kappa_e_xprof_cal[:PMLX]
    self.kappa_h_xprof[:PMLX] = kappa_h_xprof_cal[:PMLX].flip(0)
    self.kappa_h_xprof[txsize - PMLX:txsize] = kappa_h_xprof_cal[:PMLX]

    self.kappa_e_yprof[1:PMLY + 1] = kappa_e_yprof_cal[:PMLY].flip(0)
    self.kappa_e_yprof[tysize - PMLY:tysize] = kappa_e_yprof_cal[:PMLY]
    self.kappa_h_yprof[:PMLY] = kappa_h_yprof_cal[:PMLY].flip(0)
    self.kappa_h_yprof[tysize - PMLY:tysize] = kappa_h_yprof_cal[:PMLY]

    temp1x = torch.exp(-(sigma_e_xprof[:PMLX].flip(0) / kappa_e_xprof_cal[:PMLX].flip(0) + alpha_e_xprof[:PMLX].flip(0)) * dt / eps0)
    self.Cxa1[1:PMLX + 1] = temp1x
    self.Cxb1[1:PMLX + 1] = (temp1x - 1) * sigma_e_xprof[:PMLX].flip(0) / (dx * kappa_e_xprof_cal[:PMLX].flip(0) * (sigma_e_xprof[:PMLX].flip(0) + kappa_e_xprof_cal[:PMLX].flip(0) * alpha_e_xprof[:PMLX].flip(0)))

    temp2x = torch.exp(-(sigma_e_xprof[:PMLX] / kappa_e_xprof_cal[:PMLX] + alpha_e_xprof[:PMLX]) * dt / eps0)
    self.Cxa2[:PMLX] = temp2x
    self.Cxb2[:PMLX] = (temp2x - 1) * sigma_e_xprof[:PMLX] / (dx * kappa_e_xprof_cal[:PMLX] * (sigma_e_xprof[:PMLX] + kappa_e_xprof_cal[:PMLX] * alpha_e_xprof[:PMLX]))

    temp1y = torch.exp(-(sigma_e_yprof[:PMLY].flip(0) / kappa_e_yprof_cal[:PMLY].flip(0) + alpha_e_yprof[:PMLY].flip(0)) * dt / eps0)
    self.Cya1[1:PMLY + 1] = temp1y
    self.Cyb1[1:PMLY + 1] = (temp1y - 1) * sigma_e_yprof[:PMLY].flip(0) / (dy * kappa_e_yprof_cal[:PMLY].flip(0) * (sigma_e_yprof[:PMLY].flip(0) + kappa_e_yprof_cal[:PMLY].flip(0) * alpha_e_yprof[:PMLY].flip(0)))

    temp2y = torch.exp(-(sigma_e_yprof[:PMLY + 1] / kappa_e_yprof_cal[:PMLY + 1] + alpha_e_yprof[:PMLY + 1]) * dt / eps0)
    self.Cya2[:PMLY + 1] = temp2y
    self.Cyb2[:PMLY + 1] = (temp2y - 1) * sigma_e_yprof[:PMLY + 1] / (dy * kappa_e_yprof_cal[:PMLY + 1] * (sigma_e_yprof[:PMLY + 1] + kappa_e_yprof_cal[:PMLY + 1] * alpha_e_yprof[:PMLY + 1]))

    temp1xh = torch.exp(-(sigma_h_xprof[:PMLX].flip(0) / kappa_h_xprof_cal[:PMLX].flip(0) + alpha_h_xprof[:PMLX].flip(0)) * dt / eps0)
    self.Dxa1[:PMLX] = temp1xh
    self.Dxb1[:PMLX] = (temp1xh - 1) * sigma_h_xprof[:PMLX].flip(0) / (dx * kappa_h_xprof_cal[:PMLX].flip(0) * (sigma_h_xprof[:PMLX].flip(0) + kappa_h_xprof_cal[:PMLX].flip(0) * alpha_h_xprof[:PMLX].flip(0)))

    temp2xh = torch.exp(-(sigma_h_xprof[:PMLX] / kappa_h_xprof_cal[:PMLX] + alpha_h_xprof[:PMLX]) * dt / eps0)
    self.Dxa2[:PMLX] = temp2xh
    self.Dxb2[:PMLX] = (temp2xh - 1) * sigma_h_xprof[:PMLX] / (dx * kappa_h_xprof_cal[:PMLX] * (sigma_h_xprof[:PMLX] + kappa_h_xprof_cal[:PMLX] * alpha_h_xprof[:PMLX]))

    temp1yh = torch.exp(-(sigma_h_yprof[:PMLY].flip(0) / kappa_h_yprof_cal[:PMLY].flip(0) + alpha_h_yprof[:PMLY].flip(0)) * dt / eps0)
    self.Dya1[:PMLY] = temp1yh
    self.Dyb1[:PMLY] = (temp1yh - 1) * sigma_h_yprof[:PMLY].flip(0) / (dy * kappa_h_yprof_cal[:PMLY].flip(0) * (sigma_h_yprof[:PMLY].flip(0) + kappa_h_yprof_cal[:PMLY].flip(0) * alpha_h_yprof[:PMLY].flip(0)))

    temp2yh = torch.exp(-(sigma_h_yprof[:PMLY] / kappa_h_yprof_cal[:PMLY] + alpha_h_yprof[:PMLY]) * dt / eps0)
    self.Dya2[:PMLY] = temp2yh
    self.Dyb2[:PMLY] = (temp2yh - 1) * sigma_h_yprof[:PMLY] / (dy * kappa_h_yprof_cal[:PMLY] * (sigma_h_yprof[:PMLY] + kappa_h_yprof_cal[:PMLY] * alpha_h_yprof[:PMLY]))

      
  def E_field_cal(self):
    self.Ez_pp[1:-1, 1:-1] = self.Ez_p[1:-1, 1:-1].clone()
    self.Ez_p[1:-1, 1:-1] = self.Ez[1:-1, 1:-1].clone()

    temp5 = 0
    temp6 = 0

    temp5 = torch.einsum('ijk,kij->ij', self.Ca_temp[1:-1, 1:-1, :], self.Jz[:, 1:-1, 1:-1])
    temp6 = torch.einsum('ijk,kij->ij', self.Cb_temp[1:-1, 1:-1, :], self.Jz_p[:, 1:-1, 1:-1])

    self.Ez[1:-1, 1:-1] = self.C1[1:-1, 1:-1] * self.Ez_p[1:-1, 1:-1] - self.C2[1:-1, 1:-1] * self.Ez_pp[1:-1, 1:-1] + (self.Cb_dx[1:-1, 1:-1] / self.kappa_e_xprof[1:].unsqueeze(1).expand(-1, tysize-1)) * (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) - (self.Cb_dy[1:-1, 1:-1] / self.kappa_e_yprof[1:].unsqueeze(0).expand(txsize-1, -1)) * (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1]) - temp5 - temp6
    
    # return self.Ez
      
  def J_field_cal(self):
    Jz_temp = self.Jz[:, 1:-1, 1:-1].clone()
    self.Jz[:, 1:-1, 1:-1] = (torch.einsum('ijk,kij->kij', self.Ca[1:-1, 1:-1, :], self.Jz[:, 1:-1, 1:-1]) + torch.einsum('ijk,kij->kij', self.Cb[1:-1, 1:-1, :], self.Jz_p[:, 1:-1, 1:-1]) + torch.einsum('ijk,ij->kij', self.Cc[1:-1, 1:-1, :], self.Ez[1:-1, 1:-1]) + torch.einsum('ijk,ij->kij', self.Cd[1:-1, 1:-1, :], self.Ez_p[1:-1, 1:-1]) + torch.einsum('ijk,ij->kij', self.Ce[1:-1, 1:-1, :], self.Ez_pp[1:-1, 1:-1]))
    self.Jz_p[:, 1:-1, 1:-1] = Jz_temp
    # return self.Jz

  def H_field_cal(self):
    self.Hx[1:-1, :] = Da * self.Hx[1:-1, :] - (self.Db_dy[1:,:-1] / self.kappa_h_yprof) * (self.Ez[1:-1, 1:] - self.Ez[1:-1, :-1])
    self.Hy[:, 1:-1] = Da * self.Hy[:, 1:-1] + (self.Db_dx[:-1,1:] / self.kappa_h_xprof.unsqueeze(1)) * (self.Ez[1:, 1:-1] - self.Ez[:-1, 1:-1])
    # return self.Hx, self.Hy
    
  def F_field_cal(self):
      
    # left x-pml region (Subdomain 1)
    fzx1_temp = self.fzx1[1:PMLX+1, :].clone()
    self.fzx1[1:PMLX+1, :] = self.Cxa1[1:PMLX+1].unsqueeze(1) * self.fzx1[1:PMLX+1, :] + self.Cxb1[1:PMLX+1].unsqueeze(1) * (self.Hy[1:PMLX+1, :] - self.Hy[:PMLX, :])
    self.Ez[1:PMLX+1, :] = self.Ez[1:PMLX+1, :] + self.PMLE[1:PMLX+1, :] * (self.fzx1[1:PMLX+1, :] + fzx1_temp)

    # right x-pml region (Subdomain 2)
    fzx2_temp = self.fzx2[:PMLX, :].clone()
    self.fzx2[:PMLX, :] = self.Cxa2[:PMLX].unsqueeze(1) * self.fzx2[:PMLX, :] + self.Cxb2[:PMLX].unsqueeze(1) * (self.Hy[txsize-PMLX:txsize, :] - self.Hy[txsize-PMLX-1:txsize-1, :])
    self.Ez[txsize-PMLX:txsize, :] = self.Ez[txsize-PMLX:txsize, :] + self.PMLE[:PMLX, :] * (self.fzx2[:PMLX, :] + fzx2_temp)

    # left y-pml region (Subdomain 1)
    fzy1_temp = self.fzy1[:, 1:PMLY+1].clone()
    self.fzy1[:, 1:PMLY+1] = self.Cya1[1:PMLY+1].unsqueeze(0) * self.fzy1[:, 1:PMLY+1] + self.Cyb1[1:PMLY+1].unsqueeze(0) * (self.Hx[:, 1:PMLY+1] - self.Hx[:, :PMLY])
    self.Ez[:, 1:PMLY+1] = self.Ez[:, 1:PMLY+1] - self.PMLE[:, 1:PMLY+1] * (self.fzy1[:, 1:PMLY+1] + fzy1_temp)

    # right y-pml region (Subdomain 2)
    fzy2_temp = self.fzy2[:, :PMLY].clone()
    self.fzy2[:, :PMLY] = self.Cya2[:PMLY].unsqueeze(0) * self.fzy2[:, :PMLY] + self.Cyb2[:PMLY].unsqueeze(0) * (self.Hx[:, tysize-PMLY:tysize] - self.Hx[:, tysize-PMLY-1:tysize-1])
    self.Ez[:, tysize-PMLY:tysize] = self.Ez[:, tysize-PMLY:tysize] - self.PMLE[:, :PMLY] * (self.fzy2[:, :PMLY] + fzy2_temp)


  def G_field_cal(self):
        
    # left x-pml region (Subdomain 1)
    gyx1_temp = self.gyx1[:, :].clone()
    self.gyx1[:, :] = self.Dxa1[:].unsqueeze(1) * self.gyx1[:, :] + self.Dxb1[:].unsqueeze(1) * (self.Ez[1:PMLX+1, :] - self.Ez[:PMLX, :])
    self.Hy[:PMLX, :] = self.Hy[:PMLX, :] + PMLH * (self.gyx1[:, :] + gyx1_temp)

    # right x-pml region (Subdomain 2)
    gyx2_temp = self.gyx2[:, :].clone()
    self.gyx2[:, :] = self.Dxa2[:].unsqueeze(1) * self.gyx2[:, :] + self.Dxb2[:].unsqueeze(1) * (self.Ez[txsize-PMLX+1:txsize+1, :] - self.Ez[txsize-PMLX:txsize, :])
    self.Hy[txsize-PMLX:txsize, :] = self.Hy[txsize-PMLX:txsize, :] + PMLH * (self.gyx2[:, :] + gyx2_temp)

    # left y-pml region (Subdomain 1)
    gxy1_temp = self.gxy1[:, :].clone()
    self.gxy1[:, :] = self.Dya1[:] * self.gxy1[:, :] + self.Dyb1[:] * (self.Ez[:, 1:PMLY+1] - self.Ez[:, :PMLY])
    self.Hx[:, :PMLY] = self.Hx[:, :PMLY] - PMLH * (self.gxy1[:, :] + gxy1_temp)

    # right y-pml region (Subdomain 2)
    gxy2_temp = self.gxy2[:, :].clone()
    self.gxy2[:, :] = self.Dya2[:] * self.gxy2[:, :] + self.Dyb2[:] * (self.Ez[:, tysize-PMLY+1:tysize+1] - self.Ez[:, tysize-PMLY:tysize])
    self.Hx[:, tysize-PMLY:tysize] = self.Hx[:, tysize-PMLY:tysize] - PMLH * (self.gxy2[:, :] + gxy2_temp)

      
  def rnn_step(self,src):

    self.E_field_cal()
    self.F_field_cal()

    self.Ez[CX,CY] += src[CX,CY].item()

    self.J_field_cal()
    self.H_field_cal()
    self.G_field_cal()

    return self.Ez

  def forward(self,src):

    obs_Ez = []
    obs_src = []
    self.param()
    self.pml_param(PMLX,PMLY)

    for t in range(NSTEPS):
      src[CX,CY] = E0 * math.exp(-(dt * t - n_T * T) * (dt * t - n_T * T) / (T * T)) 

      self.Ez = self.rnn_step(src)
      obs_Ez.append(self.Ez.clone())
      obs_src.append(src[CX,CY].item())

    return obs_Ez, obs_src
