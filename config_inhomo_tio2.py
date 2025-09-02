# fdtd/config_inhomo_tio2.py
import math
import torch

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# physics
mu0 = 4 * math.pi * 1e-7
mur = 1
eps0 = 1e-9 / 36 / math.pi
eta0 = 120 * math.pi
c_0 = 1 / math.sqrt(mu0 * eps0)

# grid/time
dx = 25e-9
dy = 25e-9
CFL = 0.99
dt = CFL * 1.0 / c_0 / math.sqrt((1.0 / dx)**2 + (1.0 / dy)**2)
NSTEPS = 16000

# inhomogeneous: PX, PY = 700x700, PML=10, 중앙 100x100 영역에 TiO2
PX = 700; PY = 700
PMLX = 10; PMLY = 10
txsize = PMLX + PX + PMLX
tysize = PMLY + PY + PMLY

# 물질 배치 윈도우 (TiO2 직사각형): 100x100
MX_start = 400
MX_end   = 500
MY_start = 400
MY_end   = 500

CX = int(txsize / 2)
CY = int(tysize / 2)
OX = CX - 15
OY = CY - 15

C_e = dt / 2 / eps0
C_h = dt / 2 / mu0

# source params
import math as _math
f0 = 547e12
T0 = 1 / f0
E0 = 1
BW = f0 * 0.1
T = _math.sqrt(2.) * _math.sqrt(_math.log(2.)) / (_math.pi * BW)  # 50% power at band edge
n_T = 10
alpha = 0.5

# material (TiO2 example)
pol = 0
real_pol = 0
max_pol = 2
tot_pol = 2
epsr_inf = 3.19207429174285
eps_inf = 1.0 * eps0 * epsr_inf

PMLH = dt / mu0 / 2
Da = 1