# fdtd/main.py

import os
import math
import torch
import matplotlib.pyplot as plt

from fdtd.config import *                  # use shared constants & device
from fdtd.inhomo_SS import RNN_mLor_SS_inhomo

with torch.no_grad():
    rnn_mLor = RNN_mLor_SS_inhomo().to(device)
    src = torch.zeros((txsize + 1, tysize + 1), device=device)
    src[CX, CY] = 0.0
    obs_Ez_with_SS, obs_src_with_SS = rnn_mLor.forward(src)

# time axis
time_values = torch.arange(NSTEPS, device=device) * dt

# You commented out appending Ez in forward(); plotting Ez will fail.
# Instead, plot the source waveform we recorded (obs_src_with_SS).
obs_src_with_SS_time = torch.tensor(obs_src_with_SS, device=device, dtype=torch.float32)

plt.plot(time_values.cpu().numpy(), obs_src_with_SS_time.cpu().numpy(), label='Src @ (CX,CY)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend(frameon=False)
plt.grid(True)
plt.xlim([0, NSTEPS * dt])
plt.show()

# Save source to file (Linux-safe path)
out_dir = 'output/inhomo_TiO2'
os.makedirs(out_dir, exist_ok=True)
file_path = os.path.join(out_dir, 'src_Ez_SS.txt')
with open(file_path, 'wb') as f:
    obs_Ez_with_SS.cpu().numpy().astype('float64').tofile(f)
print(f"Saved: {file_path}")
