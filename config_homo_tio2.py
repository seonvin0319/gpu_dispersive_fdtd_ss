# fdtd/config_homo_tio2.py
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mu0 = 4 * math.pi * 1e-7
mur = 1
eps0 = 1e-9 / 36 / math.pi
eta0 = 120 * math.pi
c_0 = 1 / math.sqrt(mu0 * eps0)

dx = 25e-9
dy = 25e-9
CFL = 0.99
dt = CFL * 1.0 / c_0 / math.sqrt((1.0 / dx)**2 + (1.0 / dy)**2)
NSTEPS = 16000

PX = 700; PY = 700
PMLX = 10; PMLY = 10
txsize = PMLX + PX + PMLX
tysize = PMLY + PY + PMLY

# 균질 매질: 내부 영역 전체(=PML 제외)를 물질로
MX_start = PMLX
MX_end   = txsize - PMLX
MY_start = PMLY
MY_end   = tysize - PMLY

CX = int(txsize/2)
CY = int(tysize/2)
OX = CX - 15
OY = CY - 15

C_e = dt / 2 / eps0
C_h = dt / 2 / mu0

import math as _math
f0 = 547e12
E0 = 1
BW = f0 * 0.1
T = _math.sqrt(2.) * _math.sqrt(_math.log(2.)) / (_math.pi * BW)
n_T = 10
alpha = 0.5

# TiO2 material params
pol = 0
real_pol = 0
max_pol = 2
tot_pol = 2
epsr_inf = 3.19207429174285
eps_inf = eps0 * epsr_inf

PMLH = dt / mu0 / 2
Da = 1
