# RNN-Inspired Dispersive FDTD Solver

This repository provides **GPU-accelerated dispersive FDTD (Finite Difference Time Domain) solvers** implemented in PyTorch.  
The codes adopt an **RNN-inspired state-space formulation** to model material dispersion (Drude–Lorentz model) efficiently, avoiding expensive convolution in time.  

We provide simulation examples for both **homogeneous** and **inhomogeneous** domains with **Ag (silver)** and **TiO₂** materials.

---

## 🚀 Research Contributions & Novelty

- **State-Space Reformulation of Dispersion**  
  Unlike conventional recursive convolution or ADE approaches, this solver reformulates the dispersive polarization current update in an **RNN-like recurrence structure**, bridging numerical electromagnetics with deep learning concepts.  

- **GPU-Accelerated PyTorch Implementation**  
  Built on PyTorch tensors with CUDA support, enabling **massively parallel FDTD** simulations without relying on traditional C/C++ or MPI implementations.  

- **Unified Framework for Materials & Structures**  
  The same solver framework supports both **homogeneous** and **inhomogeneous** material domains, with flexible pole–residue tables for Ag and TiO₂.  

- **Research-Oriented Outputs**  
  Simulation results are stored in binary + plotted automatically, making it easy to perform **spectral analysis (FFT), resonance study, or device characterization**.  

- **Novel Cross-Disciplinary Perspective**  
  By interpreting FDTD dispersion updates as an **RNN-inspired state-space recurrence**, this work offers a fresh viewpoint that connects **computational electromagnetics** and **sequence modeling in AI**.  

---

## 📐 Methodology Overview

- The solver is based on **2D FDTD with dispersive materials** using the Drude–Lorentz model.  
- The polarization current is updated via a **state-space recurrence**, analogous to an RNN hidden state update.  
- Perfectly Matched Layers (PML) are implemented for absorbing boundaries.

### 🔲 Cell Diagram (to be added)

> *Here a figure will illustrate:*  
> - Yee grid arrangement (Ez, Hx, Hy)  
> - Location of dispersive material update (Jz, W1, W2)  
> - State-space recurrence flow  
>   
> *(Figure placeholder — will be inserted later as `docs/cell_diagram.png`)*

---

## 📂 Repository Structure

```
.
├── homo_ag.py        # Homogeneous Ag medium
├── homo_tio2.py      # Homogeneous TiO₂ medium
├── inhomo_ag.py      # Inhomogeneous Ag structure
├── inhomo_tio2.py    # Inhomogeneous TiO₂ structure
└── outputs/          # Simulation results (binary & plots)
```

---

## ⚡ Features
- **Dispersive FDTD in PyTorch**
  - State-space recursive convolution for Drude–Lorentz dispersion
- **GPU acceleration**
  - Runs on CUDA if available
- **Absorbing boundaries**
  - PML (Perfectly Matched Layer) implemented in both x & y directions
- **Material support**
  - Ag and TiO₂ with fitted pole-residue models
- **Data output**
  - Binary storage of full 2D fields (`Ez.bin`) and center point time series (`Ez_center.bin`)
  - Automatic plotting of field evolution at the domain center

---

## ▶️ How to Run

Each script is self-contained. Run directly with Python:

```bash
# Homogeneous Ag
python homo_ag.py

# Homogeneous TiO₂
python homo_tio2.py

# Inhomogeneous Ag
python inhomo_ag.py

# Inhomogeneous TiO₂
python inhomo_tio2.py
```

---

## 📊 Outputs

Simulation results are stored under `outputs/` in subdirectories:

- `Ez.bin` – 2D Ez field snapshots (float32, binary, saved every 10 timesteps by default)
- `Ez_center.bin` – Time-series of Ez at the domain center
- `obs_Ez@center.png` – Plot of the center Ez vs. time

Example (`outputs/homo_ag/obs_Ez@center.png`):

```
Electric Field at Domain Center vs Time
```

---

## ⚙️ Key Parameters

Inside each script:

- `NSTEPS` – Total simulation timesteps (default: 16000)
- `dx, dy` – Grid size (default: 25 nm)
- `PMLX, PMLY` – Thickness of PML layers
- `SAVE_EVERY` – Frame down-sampling rate for storage
- `epsr_inf`, `pole`, `residue` – Material dispersion parameters

Modify these constants to adjust resolution, domain size, or material response.

---

## 📚 References
- Taflove, A., & Hagness, S. C. *Computational Electrodynamics: The Finite-Difference Time-Domain Method*. Artech House, 2005.  
- Sullivan, D. M. *Electromagnetic Simulation Using the FDTD Method*. Wiley, 2013.  

---

## 💡 Notes
- Make sure you have **PyTorch**, **Matplotlib**, and **NumPy** installed.  
- The code automatically selects **GPU (CUDA)** if available, otherwise defaults to CPU.  
- Output binary files can be post-processed for spectral analysis (e.g., FFT) or visualization.  
