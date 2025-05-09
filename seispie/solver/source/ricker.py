import numpy as np

# def dgaussian(t, f0, t0, ang, amp):
# 	stf = -amp * (t - t0) * np.exp(-(np.pi * f0 * (t - t0)) ** 2)
# 	return stf * np.cos(ang), stf, stf * np.sin(ang)

def ricker(t, f0, t0, ang, amp):
	a = (np.pi * f0) ** 2
	stf = amp * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)
	return stf * np.cos(ang), stf, stf * np.sin(ang)
