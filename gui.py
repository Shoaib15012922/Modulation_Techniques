import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.signal import convolve, periodogram
from scipy.special import sinc
import matplotlib.pyplot as plt
from scipy.special import erfc

import matplotlib as mpl
if mpl.__version__ < '3.3.0':
    stem_kwargs = {'use_line_collection': True}
else:
    stem_kwargs = {}

ctk.set_appearance_mode("system")  # "light" or "dark"
ctk.set_default_color_theme("blue")  # Try "blue", "green", "dark-blue"

# ------------------- OOK Simulation Functions ------------------- #

def ook_psd_nrz(params):
    try:
        Rb = float(params.get('Bitrate (bps)', 1))
        Tb = 1 / Rb
        p_avg = 1
        R = 1
        df = Rb / 100
        f = np.arange(0, 5*Rb, df)
        x = f * Tb
        a = 2 * R * p_avg
        p_nrz = (a**2 * Tb) * (np.sinc(x)**2)
        p_nrz /= ((p_avg * R)**2 * Tb)
        plt.figure()
        plt.plot(f, p_nrz, label='OOK-NRZ')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized PSD")
        plt.title(f"Analytical PSD for OOK-NRZ\nBitrate: {Rb} bps")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run OOK-NRZ PSD: {str(e)}")

def ook_psd_rz(params):
    try:
        Rb = float(params.get('Bitrate (bps)', 1))
        Tb = 1 / Rb
        p_avg = 1
        R = 1
        df = Rb / 100
        f = np.arange(0, 5*Rb, df)
        x_rz = f * Tb / 2
        a = R * p_avg
        p_rz = (a**2 * Tb) * (np.sinc(x_rz)**2)
        for n in range(1, 5):
            idx = int(n*Rb/df)
            if idx < len(p_rz):
                p_rz[idx] += ((a**2)*Tb) * (np.sinc(n*Rb*Tb/2)**2) / Tb
        p_rz /= ((p_avg * R)**2 * Tb)
        plt.figure()
        plt.plot(f, p_rz, label='OOK-RZ')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized PSD")
        plt.title(f"Analytical PSD for OOK-RZ\nBitrate: {Rb} bps")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run OOK-RZ PSD: {str(e)}")

def ook_psd_simulated(params):
    try:
        Rb = float(params.get('Bitrate (bps)', 1))
        SigLen = int(params.get('Number of Bits', 1000))
        nsamp = int(params.get('Samples per Bit', 10))
        fsamp = Rb * nsamp
        Tx_filter = np.ones(nsamp)
        bin_data = np.random.randint(0, 2, SigLen)
        bin_signal = np.repeat(bin_data, nsamp)
        f_sim, Pxx = periodogram(bin_signal, fs=fsamp)
        plt.figure()
        plt.plot(f_sim, 10*np.log10(Pxx))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.title(f"Simulated PSD of OOK Signal (NRZ)\n{Rb} bps, {SigLen} bits, {nsamp} samples/bit")
        plt.grid()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run OOK simulated PSD: {str(e)}")

def ook_multipath_simulation(params):
    try:
        sig_length = int(params.get('Number of Bits', 1000))
        nsamp = int(params.get('Samples per Bit', 10))
        Rb = float(params.get('Bitrate (bps)', 1))
        i_peak = float(params.get('Peak Amplitude', 1.0))
        Dt = float(params.get('Delay Spread (Dt)', 0.1))
        sgma = float(params.get('Noise Sigma', 0.2))
        Tb = 1 / Rb
        Tsamp = Tb / nsamp
        Drms = Dt * Tb
        a = 12 * np.sqrt(11 / 13) * Drms
        K = int(10 * nsamp)
        k = np.arange(0, K + 1)
        h = (6 * a**6) / ((k * Tsamp + a)**7)
        h = h / np.sum(h)
        pt = np.ones(nsamp) * i_peak
        matched_filter = convolve(pt, h)
        matched_filter = matched_filter / np.linalg.norm(matched_filter)
        system_response = convolve(matched_filter, pt)
        delay = np.argmax(system_response)
        OOK = np.random.randint(0, 2, sig_length)
        Tx_signal = np.repeat(OOK, nsamp) * i_peak
        channel_output = convolve(Tx_signal, h)
        Rx_signal = channel_output + sgma * np.random.randn(len(channel_output))
        MF_out = convolve(Rx_signal, matched_filter) * Tsamp
        MF_out_downsamp = MF_out[delay::nsamp][:sig_length]
        Ep = np.sum(matched_filter**2) * Tsamp
        threshold = Ep / 2
        Rx_th = (MF_out_downsamp > threshold).astype(int)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(Tx_signal[:200])
        plt.title(f'Transmitted Signal (first 200 samples)\nBitrate: {Rb} bps, Amplitude: {i_peak}')
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(Rx_signal[:200])
        plt.title(f'Received Signal with Noise (σ={sgma})\nDelay Spread: {Dt}, Samples/bit: {nsamp}')
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.stem(OOK[:50], linefmt='b-', markerfmt='bo', basefmt=' ')
        plt.stem(Rx_th[:50], linefmt='r-', markerfmt='ro', basefmt=' ')
        error_indices = np.where(OOK[:50] != Rx_th[:50])[0]
        plt.scatter(error_indices, Rx_th[error_indices], color='black', marker='x', s=100, label='Errors')
        plt.title('Original vs Received Bits (first 50 bits)')
        plt.legend(['Original', 'Received', 'Errors'])
        plt.grid()
        plt.tight_layout()
        plt.show()
        BER = np.mean(OOK != Rx_th)
        messagebox.showinfo("Results",
            f"Simulation Parameters:\n"
            f"- Bitrate: {Rb} bps\n"
            f"- Bits: {sig_length}\n"
            f"- Samples/bit: {nsamp}\n"
            f"- Peak Amplitude: {i_peak}\n"
            f"- Delay Spread: {Dt}\n"
            f"- Noise Sigma: {sgma}\n\n"
            f"Bit Error Rate: {BER:.6f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run OOK multipath simulation: {str(e)}")

# ------------------- PPM Functions ------------------- #

def generate_ppm_sequence(params):
    try:
        M = int(params.get('Bit Resolution (M)', 4))
        nsym = int(params.get('Number of Symbols', 100))
        L = 2**M
        ppm_signal = []
        for _ in range(nsym):
            temp = np.random.randint(0, 2, M)
            dec_value = int("".join(str(bit) for bit in temp), 2)
            symbol = np.zeros(L)
            symbol[dec_value] = 1
            ppm_signal.extend(symbol)
        plt.figure(figsize=(12, 4))
        plt.stem(ppm_signal[:5*L])
        plt.title(f"First 5 Symbols of {L}-PPM Sequence\n(M={M}, {nsym} symbols)")
        plt.xlabel("Slot Index")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.show()
        messagebox.showinfo("PPM Sequence Generated",
                          f"Successfully generated {L}-PPM sequence:\n"
                          f"- Bit resolution (M): {M}\n"
                          f"- Number of symbols: {nsym}\n"
                          f"- Sequence length: {len(ppm_signal)} slots")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate PPM sequence: {str(e)}")

def ppm_psd_analytical(params):
    try:
        Rb = float(params.get('Bitrate (bps)', 1e6))
        M = int(params.get('Bit Resolution (M)', 4))
        p_avg = float(params.get('Avg Power (p_avg)', 1))
        R = float(params.get('Responsivity (R)', 1))
        snr_db = float(params.get('SNR (dB)', 10))
        snr_linear = 10 ** (snr_db / 10)
        Tb = 1 / Rb
        L = 2 ** M
        a = R * L * p_avg
        Ts = M / (L * Rb)
        Rs = 1 / Ts
        df = Rs / 1000
        f = np.arange(0, 8 * Rb, df)
        P_sq = (a * Ts)**2 * (np.sinc(f * Ts))**2
        temp1 = np.zeros_like(f)
        for k in range(1, L):
            temp1 += (k / L - 1) * np.cos(2 * np.pi * k * f * Ts)
        S_c = (1 / (L * Ts)) * (((L - 1) / L) + (2 / L) * temp1)
        S = P_sq * S_c
        S /= ((p_avg * R)**2 * Tb)
        plt.figure(figsize=(10, 5))
        plt.plot(f, S, label=f'{L}-PPM PSD')
        plt.title(f"Analytical PSD of {L}-PPM\nRb={Rb/1e6} Mbps, M={M}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized PSD")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        ber = 0.5 * erfc(np.sqrt(snr_linear / 2))
        messagebox.showinfo("BER Calculation", f"Calculated BER for {L}-PPM modulation: {ber:.6f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate PPM PSD: {str(e)}")

def ppm_ser_simulation(params):
    try:
        M = int(params.get('Bit Resolution (M)', 4))
        nsym = int(params.get('Number of Symbols', 500))
        Rb = float(params.get('Bitrate (bps)', 1e6))
        min_EbN0 = float(params.get('Min Eb/N0 (dB)', -10))
        max_EbN0 = float(params.get('Max Eb/N0 (dB)', 10))
        step_EbN0 = float(params.get('Step Eb/N0 (dB)', 2))
        EbN0_dB = np.arange(min_EbN0, max_EbN0 + step_EbN0, step_EbN0)
        L = 2**M
        SNR = 10**(EbN0_dB / 10)
        EsN0_dB = EbN0_dB + 10 * np.log10(M)
        ser_hdd = []
        ser_sdd = []
        for i, ebn0 in enumerate(EbN0_dB):
            ppm = []
            for _ in range(nsym):
                rand_bits = np.random.randint(0, 2, M)
                dec_val = int("".join(str(b) for b in rand_bits), 2)
                symbol = np.zeros(L)
                symbol[dec_val] = 1
                ppm.extend(symbol)
            ppm = np.array(ppm)
            noisy_ppm = ppm + np.sqrt(1/(2*10**(EsN0_dB[i]/10))) * np.random.randn(len(ppm))
            Rx_th = np.zeros_like(noisy_ppm)
            Rx_th[noisy_ppm > 0.5] = 1
            hdd_ser = np.sum(ppm != Rx_th) / len(ppm)
            ser_hdd.append(hdd_ser)
            PPM_SDD = []
            for k in range(nsym):
                slot = noisy_ppm[k*L:(k+1)*L]
                max_val = np.max(slot)
                symbol = np.zeros(L)
                symbol[np.where(slot == max_val)[0]] = 1
                PPM_SDD.extend(symbol)
            sdd_ser = np.sum(ppm != np.array(PPM_SDD)) / len(ppm)
            ser_sdd.append(sdd_ser)
        Ps_theoretical_hdd = 0.5 * erfc(np.sqrt(0.5 * M * L * SNR))
        Ps_theoretical_sdd = 0.5 * erfc(np.sqrt(M * L * SNR))
        plt.figure(figsize=(10, 6))
        plt.semilogy(EbN0_dB, ser_hdd, 'bo-', label='HDD Simulated')
        plt.semilogy(EbN0_dB, ser_sdd, 'go-', label='SDD Simulated')
        plt.semilogy(EbN0_dB, Ps_theoretical_hdd, 'k--', label='HDD Theoretical')
        plt.semilogy(EbN0_dB, Ps_theoretical_sdd, 'r--', label='SDD Theoretical')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Symbol Error Rate (SER)')
        plt.title(f'{L}-PPM SER Performance (M={M}, {nsym} symbols)')
        plt.legend()
        plt.grid(True, which='both', linestyle=':')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run PPM SER simulation: {str(e)}")

# ------------------- DPIM Functions ------------------- #

def generate_dpim_sequence(params):
    try:
        M = int(params.get('Bit Resolution (M)', 4))
        nsym = int(params.get('Number of Symbols', 100))
        NGS = int(params.get('Guard Slots (NGS)', 0))
        DPIM = []
        inpb = np.random.randint(0, 2, size=(nsym, M))
        for i in range(nsym):
            inpd = int("".join(map(str, inpb[i])), 2)
            temp = [0] * (inpd + NGS)
            DPIM.extend([1] + temp)
        DPIM = np.array(DPIM)
        plt.figure(figsize=(12, 4))
        plt.stem(DPIM[:100], linefmt='b-', markerfmt='bo', basefmt='r-')
        plt.title(f"First 100 Slots of DPIM Sequence\n(M={M}, {nsym} symbols, NGS={NGS})")
        plt.xlabel("Slot Index")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.show()
        messagebox.showinfo("DPIM Sequence Generated",
                          f"Successfully generated DPIM sequence:\n"
                          f"- Bit resolution (M): {M}\n"
                          f"- Number of symbols: {nsym}\n"
                          f"- Guard slots (NGS): {NGS}\n"
                          f"- Sequence length: {len(DPIM)} slots")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate DPIM sequence: {str(e)}")

def dpim_psd(params):
    try:
        Rb = float(params.get('Bitrate (bps)', 1e6))
        M = int(params.get('Bit Resolution (M)', 4))
        p_avg = float(params.get('Avg Power (p_avg)', 1))
        R = float(params.get('Responsivity (R)', 1))
        Tb = 1 / Rb
        L = 2**M
        Lavg = 0.5 * (2**M + 1)
        a = R * Lavg * p_avg
        Ts = M / (Lavg * Rb)
        Rs = 1 / Ts
        df = Rs / 100
        f = np.arange(0, 8 * Rb, df)
        x = f * Ts
        r = [2/(L+1)]
        for k in range(1, L+1):
            r.append(2/((L**k) * (L+1)**(k-2)))
        for k in range(L+1, 5*L):
            temp = sum(r[k-i] for i in range(1, L+1))
            r.append((1/L) * temp)
        for k in range(5*L, 1000):
            r.append((1/Lavg)**2)
        r = np.array(r)
        P_sq = (a * Ts)**2 * (np.sinc(f * Ts))**2
        term2 = 0
        for ii in range(len(r)-1):
            term2 += (r[ii+1] - (1/Lavg)**2) * np.cos(2 * np.pi * ii * f * Ts)
        p = (1/Ts) * P_sq * ((r[0] - 1/Lavg**2) + 2 * term2)
        plt.figure(figsize=(10, 5))
        plt.plot(f, p, label=f'DPIM PSD (M={M})')
        plt.title(f"Analytical PSD of DPIM\nRb={Rb/1e6} Mbps, M={M}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized PSD")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate DPIM PSD: {str(e)}")

def dpim_ser_simulation(params):
    try:
        M = int(params.get('Bit Resolution (M)', 4))
        nsym = int(params.get('Number of Symbols', 500))
        NGS = int(params.get('Guard Slots (NGS)', 0))
        Rb = float(params.get('Bitrate (bps)', 200e6))
        min_EbN0 = float(params.get('Min Eb/N0 (dB)', -10))
        max_EbN0 = float(params.get('Max Eb/N0 (dB)', 10))
        step_EbN0 = float(params.get('Step Eb/N0 (dB)', 2))
        EbN0_dB = np.arange(min_EbN0, max_EbN0 + step_EbN0, step_EbN0)
        Lavg = 0.5*(2**M + 1) + NGS
        Tb = 1/Rb
        Ts = M / (Lavg * Rb)
        SNR = 10**(EbN0_dB / 10)
        ser = []
        for Eb in EbN0_dB:
            DPIM = generate_DPIM(M, nsym, NGS)
            Lsig = len(DPIM)
            noise_power = 10**(-(Eb + 3)/10)
            noise = np.random.normal(0, np.sqrt(noise_power/2), size=Lsig)
            MF_out = DPIM + noise
            Rx_DPIM_th = np.zeros(Lsig)
            Rx_DPIM_th[MF_out > 0.5] = 1
            errors = np.sum(Rx_DPIM_th != DPIM)
            ser.append(errors / Lsig)
        Pse_DPIM = 0.5 * erfc(np.sqrt(0.5 * M * Lavg * SNR / 2))
        plt.figure(figsize=(10, 6))
        plt.semilogy(EbN0_dB, ser, 'bo-', label='Simulated SER')
        plt.semilogy(EbN0_dB, Pse_DPIM, 'r--', label='Theoretical SER')
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Symbol Error Rate (SER)')
        plt.title(f'DPIM SER Performance (M={M}, {nsym} symbols, NGS={NGS})')
        plt.legend()
        plt.grid(True, which='both', linestyle=':')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run DPIM SER simulation: {str(e)}")

def generate_DPIM(M, nsym, NGS=0):
    DPIM = []
    inpb = np.random.randint(0, 2, size=(nsym, M))
    for i in range(nsym):
        inpd = int("".join(map(str, inpb[i])), 2)
        temp = [0] * (inpd + NGS)
        DPIM.extend([1] + temp)
    return np.array(DPIM)

# ------------------- GUI Functions ------------------- #

def open_input_dialog(callback_func):
    input_win = ctk.CTkToplevel(root)
    input_win.title("Input Parameters")
    func_name = callback_func.__name__
    if func_name in ['ook_psd_nrz', 'ook_psd_rz']:
        fields = ["Bitrate (bps)"]
        input_win.geometry("300x150")
    elif func_name == 'ook_psd_simulated':
        fields = ["Bitrate (bps)", "Number of Bits", "Samples per Bit"]
        input_win.geometry("300x250")
    elif func_name == 'ook_multipath_simulation':
        fields = [
            "Bitrate (bps)",
            "Number of Bits",
            "Samples per Bit",
            "Peak Amplitude",
            "Delay Spread (Dt)",
            "Noise Sigma"
        ]
        input_win.geometry("300x400")
    elif func_name in ['ppm_psd_analytical', 'dpim_psd']:
        fields = [
            "Bitrate (bps)",
            "Bit Resolution (M)",
            "Avg Power (p_avg)",
            "Responsivity (R)"
        ]
        input_win.geometry("350x300")
    elif func_name == 'generate_ppm_sequence':
        fields = [
            "Bit Resolution (M)",
            "Number of Symbols"
        ]
        input_win.geometry("300x200")
    elif func_name == 'generate_dpim_sequence':
        fields = [
            "Bit Resolution (M)",
            "Number of Symbols",
            "Guard Slots (NGS)"
        ]
        input_win.geometry("350x250")
    elif func_name in ['ppm_ser_simulation', 'dpim_ser_simulation']:
        fields = [
            "Bit Resolution (M)",
            "Number of Symbols",
            "Bitrate (bps)",
            "Min Eb/N0 (dB)",
            "Max Eb/N0 (dB)",
            "Step Eb/N0 (dB)"
        ]
        if func_name == 'dpim_ser_simulation':
            fields.insert(2, "Guard Slots (NGS)")
        input_win.geometry("350x400")
    else:
        fields = ["Bitrate", "Num Bits", "SNR (if needed)"]
        input_win.geometry("300x250")
    entries = {}
    defaults = {
        "Bitrate (bps)": "1",
        "Number of Bits": "1000",
        "Samples per Bit": "10",
        "Peak Amplitude": "1.0",
        "Delay Spread (Dt)": "0.1",
        "Noise Sigma": "0.2",
        "Bit Resolution (M)": "4",
        "Number of Symbols": "100",
        "Avg Power (p_avg)": "1",
        "Responsivity (R)": "1",
        "Guard Slots (NGS)": "0",
        "Min Eb/N0 (dB)": "-10",
        "Max Eb/N0 (dB)": "10",
        "Step Eb/N0 (dB)": "2",
        "Bitrate": "1",
        "Num Bits": "1000",
        "SNR (if needed)": "10"
    }
    for field in fields:
        frame = ctk.CTkFrame(input_win)
        frame.pack(fill='x', padx=5, pady=5)
        ctk.CTkLabel(frame, text=field, font=ctk.CTkFont(size=12)).pack(side='left')
        entry = ctk.CTkEntry(frame)
        entry.insert(0, defaults.get(field, ""))
        entry.pack(side='right', expand=True, fill='x')
        entries[field] = entry
    def submit():
        params = {field: entries[field].get() for field in fields}
        input_win.destroy()
        callback_func(params)
    btn_frame = ctk.CTkFrame(input_win)
    btn_frame.pack(fill='x', pady=10)
    ctk.CTkButton(btn_frame, text="Run", command=submit).pack(side='right', padx=10)

def on_modulation_select(modulation_type, selected_modulation):
    func_map = {
        'OOK': {
            'OOK-NRZ PSD': ook_psd_nrz,
            'OOK-RZ PSD': ook_psd_rz,
            'OOK-PERIODOGRAM': ook_psd_simulated,
            'OOK-MULTIPATH': ook_multipath_simulation
        },
        'PPM': {
            'PPM-PSD': ppm_psd_analytical,
            'PPM-SER': ppm_ser_simulation,
            'PPM-SEQUENCE': generate_ppm_sequence
        },
        'DPIM': {
            'DPIM-SEQUENCE': generate_dpim_sequence,
            'DPIM-PSD': dpim_psd,
            'DPIM-SER': dpim_ser_simulation
        }
    }
    func = func_map[modulation_type][selected_modulation]
    open_input_dialog(func)

def show_modulation_options():
    start_frame.pack_forget()
    modulation_frame.pack(fill='both', expand=True)

def back_to_start():
    modulation_frame.pack_forget()
    start_frame.pack(fill='both', expand=True)

# ------------------- GUI Setup ------------------- #

root = ctk.CTk()
root.title("FSO Modulation Simulator 3000")
root.geometry("700x520")

# Start Frame
start_frame = ctk.CTkFrame(root)
ctk.CTkLabel(start_frame, text="Step 1: Choose FSO Type", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=30)
ctk.CTkButton(start_frame, text="Indoor", font=ctk.CTkFont(size=14), command=show_modulation_options, width=180).pack(pady=16)
ctk.CTkButton(start_frame, text="Outdoor", font=ctk.CTkFont(size=14), command=lambda: messagebox.showinfo("Info", "Outdoor module coming soon!"), width=180).pack(pady=16)
start_frame.pack(fill='both', expand=True)

# Modulation Frame
modulation_frame = ctk.CTkFrame(root)
ctk.CTkLabel(modulation_frame, text="Step 2: Select Modulation Scheme", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=25)

def make_buttons_row(frame, items, type_):
    row = ctk.CTkFrame(frame)
    for label in items:
        b = ctk.CTkButton(row, text=label, width=180, font=ctk.CTkFont(size=12, weight="bold"),
                          command=lambda l=label: on_modulation_select(type_, l))
        b.pack(side=tk.LEFT, padx=8, pady=8)
    row.pack()

ctk.CTkLabel(modulation_frame, text="OOK Modulation", font=ctk.CTkFont(size=14, weight="bold")).pack()
make_buttons_row(modulation_frame,
                ['OOK-NRZ PSD', 'OOK-RZ PSD', 'OOK-PERIODOGRAM', 'OOK-MULTIPATH'],
                'OOK')
ctk.CTkLabel(modulation_frame, text="PPM Modulation", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12,0))
make_buttons_row(modulation_frame,
                ['PPM-PSD', 'PPM-SER', 'PPM-SEQUENCE'],
                'PPM')
ctk.CTkLabel(modulation_frame, text="DPIM Modulation", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(12,0))
make_buttons_row(modulation_frame,
                ['DPIM-SEQUENCE', 'DPIM-PSD', 'DPIM-SER'],
                'DPIM')
ctk.CTkButton(modulation_frame, text="⬅ Back", font=ctk.CTkFont(size=13), width=120, command=back_to_start).pack(pady=25)

root.mainloop()
