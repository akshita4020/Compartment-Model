import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define constants
k_in = lambda t: -4.709 * np.log(t) + 34
k12 = 0.1
k21 = 0.2
k23 = 0.1
k32 = 0.2
k34 = 0.1
k43 = 0.2
k45 = 0.1
k54 = 0.2
k85 = 0.1
k_loss = 0.05
k16 = 0.1
k58 = 0.1
k61 = 0.1
k67 = 0.1
k76 = 0.1
k78 = 0.1
k87 = 0.1
k0 = 0.05
k_d = np.log(2) / 2.0  # half-life of 2 days

# Initial concentrations
C0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Define the system of ODEs
def model(C, t):
    CPA, CC0, CS, CCh, CR, CCr, CAc, CVb = C
    dCPA_dt = k_in(t) - (k12 + k_loss + k_d) * CPA + k21 * CC0
    dCC0_dt = k12 * CPA + k32 * CS - (k21 + k23 + k_d) * CC0
    dCS_dt = k23 * CC0 + k43 * CCh - (k32 + k34 + k_d) * CS
    dCCh_dt = k34 * CS + k54 * CR - (k43 + k45 + k_d) * CCh - C0[0]
    dCR_dt = k45 * CCh - (k54 + k85 + k_d) * CR
    dCCr_dt = k16 * CAc + k58 * CR - (k61 + k67 + k_d) * CCr
    dCAc_dt = k67 * CCr + k87 * CVb - (k76 + k78 + k_d) * CAc
    dCVb_dt = k78 * CS + k54 * CR - (k87 + k85 + k0 + k_d) * CVb
    return [dCPA_dt, dCC0_dt, dCS_dt, dCCh_dt, dCR_dt, dCCr_dt, dCAc_dt, dCVb_dt]

# Time points
t = np.linspace(1, 100, 1000)  # avoid log(0) by starting at t=1

# Solve the system of ODEs
sol = odeint(model, C0, t)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(t, sol[:, 0], label='C_PA')
plt.plot(t, sol[:, 1], label='C_C0')
plt.plot(t, sol[:, 2], label='C_S')
plt.plot(t, sol[:, 3], label='C_Ch')
plt.plot(t, sol[:, 4], label='C_R')
plt.plot(t, sol[:, 5], label='C_Cr')
plt.plot(t, sol[:, 6], label='C_Ac')
plt.plot(t, sol[:, 7], label='C_Vb')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Compartmental Model with Half-Life Consideration')
plt.grid()
plt.show()
