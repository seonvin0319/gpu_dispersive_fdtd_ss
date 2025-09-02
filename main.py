# fdtd/main.py
import os
import math
import numpy as np
import torch

from fdtd.config import *                  # 모든 공용 상수/파라미터/디바이스
from fdtd.inhomo_SS import RNN_mLor_SS_inhomo

# ----------------------------
# 저장 설정: MATLAB 코드에 맞춘 파일명/포맷
# ----------------------------
OUT_DIR = 'output/inhomo_TiO2'
os.makedirs(OUT_DIR, exist_ok=True)

# MATLAB 1번 스크립트는 ss=10:10:1200까지만 사용 → 현실적 파일 크기를 위해 1200까지 저장 권장
# 전 스텝(=NSTEPS) 저장하려면 SAVE_STEPS = NSTEPS 로 변경 (매우 큰 파일 주의)
SAVE_STEPS = min(1200, NSTEPS)

# 파일 경로 (MATLAB 코드에서 읽는 이름과 동일)
FIELD_PATH = os.path.join(OUT_DIR, 'inhomo_TiO2_Ez.txt')     # 2D 필드 프레임들(스텝별 순서 저장)
TRACE_PATH = os.path.join(OUT_DIR, 'inhomo_TiO2_Ez_1D.txt')  # Ez(CX,CY) 1D 트레이스

def main():
    with torch.no_grad():
        model = RNN_mLor_SS_inhomo().to(device)

        # 내부 파라미터/계수 준비 (rnn_step를 직접 돌리므로 여기서 초기화)
        model.param()
        model.pml_param(PMLX, PMLY)

        # 소스 맵
        src = torch.zeros((txsize + 1, tysize + 1), device=device)

        # 바이너리 파일 오픈(덮어쓰기)
        f_field = open(FIELD_PATH, 'wb')   # double(binary)
        f_trace = open(TRACE_PATH, 'wb')   # double(binary)

        try:
            for t in range(NSTEPS):
                # 가우시안 펄스 소스
                src[CX, CY] = E0 * math.exp(-(dt * t - n_T * T) * (dt * t - n_T * T) / (T * T))

                # 한 스텝 진행
                Ez = model.rnn_step(src)

                # 1D 트레이스: Ez(CX,CY) 저장 (전 스텝)
                ez_scalar = float(Ez[CX, CY].item())
                np.array([ez_scalar], dtype=np.float64).tofile(f_trace)

                # 2D 필드 프레임 저장 (처음 SAVE_STEPS 스텝까지만)
                if (t + 1) <= SAVE_STEPS:
                    # MATLAB은 column-major(Fortran) → Fortran 순서로 직렬화해서 저장
                    frame = Ez.detach().to('cpu').numpy().astype(np.float64)
                    frame.ravel(order='F').tofile(f_field)

            print(f"[OK] Saved 1D trace to: {TRACE_PATH}")
            print(f"[OK] Saved field frames (1..{SAVE_STEPS}) to: {FIELD_PATH}")

            if SAVE_STEPS < NSTEPS:
                approx_gb = (txsize + 1) * (tysize + 1) * SAVE_STEPS * 8 / 1024 / 1024 / 1024
                print(f"Note: Saved only first {SAVE_STEPS} steps (~{approx_gb:.2f} GB). "
                      f"Set SAVE_STEPS = NSTEPS to save all (HUGE).")

        finally:
            f_field.close()
            f_trace.close()

if __name__ == "__main__":
    main()
