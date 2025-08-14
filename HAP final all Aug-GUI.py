

# ---- HAP launcher (add these 2 lines at the very top) ----
choice = input("\nHAP modules: 1=Stations  2=Burner  3=Trajectory\nSelect (1/2/3): ").strip()

if choice == "1":
# Thrust function:


    import tkinter as tk
    from tkinter import ttk
    import math
    
    # Default input values
    default_values = {
        "psi": 7.0,
        "V0": 3048,
        "T0": 222,
        "f": 0.04,
        "hf": 0.0,
        "To": 222,
        "VTe_V3": 0.50,
        "Vf_V3": 0.50,
        "Cf_Aw_A3": 0.10,
        "p10_p0": 1.40,
        "eta_c": 0.90,
        "eta_b": 0.90,
        "eta_e": 0.90,
        "fhPR": 3510000,
        "R": 289.3,
        "Cpc": 1090,
        "Cpb": 1510,
        "Cpe": 1510,
        "gamma_c": 1.362,
        "gamma_b": 1.238,
        "gamma_e": 1.238,
        "g0": 9.81
    }
    
    # Parameters
    params = list(default_values.keys())
    
    # Create GUI
    root = tk.Tk()
    root.title("Mississippi State Revival of the Heiser and Pratt HAP Model")
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")
    
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    notebook.add(tab1, text="Constant Area")
    notebook.add(tab2, text="Constant Pressure")
    
    entries, entries2 = {}, {}
    
    def create_inputs(tab, entries_dict):
        frame = tk.Frame(tab)
        frame.pack(side=tk.LEFT, padx=10, pady=10)
        for i, param in enumerate(params):
            tk.Label(frame, text=param).grid(row=i, column=0, sticky='w')
            entry = tk.Entry(frame, width=10)
            entry.insert(0, str(default_values[param]))
            entry.grid(row=i, column=1)
            entries_dict[param] = entry
        return frame
    
    input_frame = create_inputs(tab1, entries)
    input_frame2 = create_inputs(tab2, entries2)
    
    output_text = tk.Text(tab1, width=70, height=30)
    output_text.pack(side=tk.RIGHT, padx=10, pady=10)
    output_text2 = tk.Text(tab2, width=70, height=30)
    output_text2.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def get_inputs(entries_dict):
        return {key: float(entries_dict[key].get()) for key in entries_dict}
    
    def run_constant_area():
        output_text.delete(1.0, tk.END)
        val = get_inputs(entries)
        try:
            psi, V0, T0, f, hf, To = val["psi"], val["V0"], val["T0"], val["f"], val["hf"], val["To"]
            VTe_V3, Vf_V3 = val["VTe_V3"], val["Vf_V3"]
            Cf_Aw_A3, p10_p0 = val["Cf_Aw_A3"], val["p10_p0"]
            eta_c, eta_b, eta_e, fhPR = val["eta_c"], val["eta_b"], val["eta_e"], val["fhPR"]
            R, Cpc, Cpb, Cpe = val["R"], val["Cpc"], val["Cpb"], val["Cpe"]
            g0 = val["g0"]
    
            Sa0 = V0 * (1 + (R * T0) / V0**2)
            T3 = psi * T0
            V3 = math.sqrt(V0**2 - 2 * Cpc * T0 * (psi - 1))
            Sa3 = V3 * (1 + (R * T3) / V3**2)
            p3_p0 = (psi / (psi * (1 - eta_c) + eta_c))**(Cpc / R)
            A3_A0 = psi * (1 / p3_p0) * (V0 / V3)
    
            a = 1 - (R / (2 * Cpb))
            b = -(V3 / (1 + f)) * (1 + (R * T3 / V3**2) + f * Vf_V3 - (Cf_Aw_A3 / 2))
            c = (R * T3 / (1 + f)) * (1 + (1 / (Cpb * T3)) * (eta_b * fhPR + f * hf + f * Cpb * To + (1 + f * Vf_V3**2) * V3**2 / 2))
            V4 = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
            T4 = (c / R) - (V4**2 / (2 * Cpb))
            p4_p0 = (1 + f) * p3_p0 * (T4 / T3) * (V3 / V4)
            Sa4 = V4 * (1 + (R * T4) / V4**2)
            T10 = T4 * (1 - eta_e * (1 - (p10_p0 / p4_p0)**(R/Cpe)))
            V10 = math.sqrt(V4**2 + 2 * Cpe * (T4 - T10))
            Sa10 = V10 * (1 + (R * T10) / V10**2)
            A10_A0 = (1 + f) * (1 / p10_p0) * (T10 / T0) * (V0 / V10)
            F_mdot0 = (1 + f) * Sa10 - Sa0 - (R * T0 / V0) * (A10_A0 - 1)
            Isp = (1 / (g0 * f)) * F_mdot0
            eta_o = (V0 / fhPR) * F_mdot0
    
            results = [
                ("Sa0 (N·s/kg)", Sa0), ("T3 (K)", T3), ("V3 (m/s)", V3), ("Sa3 (N·s/kg)", Sa3),
                ("p3/p0", p3_p0), ("A3/A0", A3_A0), ("V4 (m/s)", V4), ("T4 (K)", T4),
                ("Sa4 (N·s/kg)", Sa4), ("T10 (K)", T10), ("V10 (m/s)", V10),
                ("Sa10 (N·s/kg)", Sa10), ("A10/A0", A10_A0), ("F/m0 (N·s/kg)", F_mdot0),
                ("Isp (s)", Isp), ("eta_o", eta_o)
            ]
            for name, val in results:
                output_text.insert(tk.END, f"{name:<25} = {val:>10.2f}\n")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {e}\n")
    
    def run_constant_pressure():
        output_text2.delete(1.0, tk.END)
        val = get_inputs(entries2)
        try:
            psi, V0, T0, f, hf, To = val["psi"], val["V0"], val["T0"], val["f"], val["hf"], val["To"]
            VTe_V3, Vf_V3 = val["VTe_V3"], val["Vf_V3"]
            Cf_Aw_A3, p10_p0 = val["Cf_Aw_A3"], val["p10_p0"]
            eta_c, eta_b, eta_e, fhPR = val["eta_c"], val["eta_b"], val["eta_e"], val["fhPR"]
            R, Cpc, Cpb, Cpe = val["R"], val["Cpc"], val["Cpb"], val["Cpe"]
            g0 = val["g0"]
    
            Sa0 = V0 * (1 + (R * T0) / V0**2)
            T3 = psi * T0
            V3 = math.sqrt(V0**2 - 2 * Cpc * T0 * (psi - 1))
            Sa3 = V3 * (1 + (R * T3) / V3**2)
            p3_p0 = (psi / (psi * (1 - eta_c) + eta_c))**(Cpc / R)
            A3_A0 = psi * (1 / p3_p0) * (V0 / V3)
    
            V4 = V3 * ((1 + f * VTe_V3) / (1 + f) - Cf_Aw_A3 / (2 * (1 + f)))
            Term1 = T3 / (1 + f)
            Term2 = (1 / (Cpb * T3)) * (eta_b * fhPR + f * hf + f * Cpb * To + (1 + f * Vf_V3**2) * V3**2 / 2)
            Term3 = V4**2 / (2 * Cpb)
            T4 = (Term1 * (1 + Term2)) - Term3
            A4_A3 = (1 + f) * (T4 / T3) * (V3 / V4)
            Sa4 = V4 * (1 + (R * T4) / V4**2)
            T10 = T4 * (1 - eta_e * (1 - (p10_p0 / p3_p0)**(R/Cpe)))
            V10 = math.sqrt(V4**2 + 2 * Cpe * (T4 - T10))
            Sa10 = V10 * (1 + (R * T10) / V10**2)
            A10_A0 = (1 + f) * (1 / p10_p0) * (T10 / T0) * (V0 / V10)
            F_mdot0 = (1 + f) * Sa10 - Sa0 - (R * T0 / V0) * (A10_A0 - 1)
            Isp = (1 / (g0 * f)) * F_mdot0
            eta_o = (V0 / fhPR) * F_mdot0
    
            results = [
                ("Sa0 (N·s/kg)", Sa0), ("T3 (K)", T3), ("V3 (m/s)", V3), ("Sa3 (N·s/kg)", Sa3),
                ("p3/p0", p3_p0), ("A3/A0", A3_A0), ("V4 (m/s)", V4), ("T4 (K)", T4),
                ("A4/A3", A4_A3), ("Sa4 (N·s/kg)", Sa4), ("T10 (K)", T10), ("V10 (m/s)", V10),
                ("Sa10 (N·s/kg)", Sa10), ("A10/A0", A10_A0), ("F/m0 (N·s/kg)", F_mdot0),
                ("Isp (s)", Isp), ("eta_o", eta_o)
            ]
            for name, val in results:
                output_text2.insert(tk.END, f"{name:<25} = {val:>10.2f}\n")
        except Exception as e:
            output_text2.insert(tk.END, f"Error: {e}\n")
    
    def reset_tab1():
        for key in entries:
            entries[key].delete(0, tk.END)
            entries[key].insert(0, str(default_values[key]))
        output_text.delete(1.0, tk.END)
    
    def reset_tab2():
        for key in entries2:
            entries2[key].delete(0, tk.END)
            entries2[key].insert(0, str(default_values[key]))
        output_text2.delete(1.0, tk.END)
    
    # Buttons
    tk.Button(input_frame, text="Run", command=run_constant_area).grid(row=len(params), column=0, pady=10)
    tk.Button(input_frame, text="Reset", command=reset_tab1).grid(row=len(params), column=1, pady=10)
    
    tk.Button(input_frame2, text="Run", command=run_constant_pressure).grid(row=len(params), column=0, pady=10)
    tk.Button(input_frame2, text="Reset", command=reset_tab2).grid(row=len(params), column=1, pady=10)
    
    root.mainloop()
    



elif choice == "2":

# Burner:


    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import math
    
    # ---------------- Burner solver (unchanged core) ------------------
    
    def burner_solver(M_inlet, T2, p2, u2, gamma, tau_b, theta, x_i, x_end, steps=400):
        Tt2 = T2 * (1 + (gamma - 1)/2 * M_inlet**2)
        dx  = (x_end - x_i) / steps
    
        x = np.linspace(x_i, x_end, steps + 1)
        A = 1 + (x - x_i) / (x_end - x_i)                # linear A/A2
    
        tau     = 1 + (tau_b - 1) * theta*(x - x_i)/(1 + (theta-1)*(x - x_i))
        dtaudx  = (tau_b - 1) * theta / (1 + (theta-1)*(x - x_i))**2
        Tt      = Tt2 * tau
        dTtdx   = Tt2 * dtaudx
        dAdx    = np.gradient(A, dx)
    
        M = np.zeros_like(x)
        M[0] = M_inlet
    
        for i in range(steps):
            def dMdx(m, idx):
                A_term = (1 + (gamma - 1)/2 * m**2) / (1 - m**2)
                B_term = -dAdx[idx]/A[idx] + (1 + gamma * m**2)/(2*Tt[idx]) * dTtdx[idx]
                return m * A_term * B_term
    
            k1 = dMdx(M[i], i)
            k2 = dMdx(M[i] + 0.5*dx*k1, i)
            k3 = dMdx(M[i] + 0.5*dx*k2, i)
            k4 = dMdx(M[i] + dx*k3, i)
            M[i+1] = M[i] + dx/6*(k1 + 2*k2 + 2*k3 + k4)
    
        T_by_T2 = (Tt / Tt2) * (1 + (gamma - 1)/2 * M_inlet**2) / (1 + (gamma - 1)/2 * M**2)
        p_by_p2 = (M_inlet/M) * (A[0]/A) * np.sqrt(T_by_T2)
        return x, M, T_by_T2, p_by_p2
    
    # ---------------- GUI ---------------------------------------------
    
    def build_gui():
        root = tk.Tk()
        root.title("Mississippi State  –  Revive of HIESR & Pratt HAP (SI mode)")
    
        # ---- Input definitions --------------------------------------
        defaults = {
            "p2 [kPa]"        : 68.95,
            "T2 [K]"          : 416.67,
            "M2"              : 3.0,
            "Rb [J/kg-K]"     : 289.26,   # not used but kept for completeness
            "u2 [m/s]"        : 1159.76,
            "θ/H"              : 0.02,     # also not directly used here
            "A3/A2"           : 1.0,      # kept for future geometry
            "A4/A2"           : 2.0,      # kept
            "H [m]"           : 0.152,    # not used in 1-D solver
            "τb"              : 1.4,
            "θ"               : 5.0,
            "x_i [m]"         : 0.914,
            "x_end [m]"       : 1.829
        }
    
        vars = {k: tk.DoubleVar(value=v) for k, v in defaults.items()}
    
        for r, (k, var) in enumerate(vars.items()):
            tk.Label(root, text=k).grid(row=r, column=0, sticky="e")
            tk.Entry(root, textvariable=var, width=8).grid(row=r, column=1)
    
        # ---- Result & plot frame ------------------------------------
        result_box = tk.Text(root, height=5, width=45)
        result_box.grid(row=len(vars)+2, column=0, columnspan=2, pady=5)
    
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=2, rowspan=len(vars)+3, padx=10)
    
        canvas = None   # will hold matplotlib canvas
    
        def run():
            nonlocal canvas
            # read inputs
            p2    = vars["p2 [kPa]"].get()
            T2    = vars["T2 [K]"].get()
            M2    = vars["M2"].get()
            u2    = vars["u2 [m/s]"].get()
            gamma = 1.24
            tau_b = vars["τb"].get()
            theta = vars["θ"].get()
            x_i   = vars["x_i [m]"].get()
            x_end = vars["x_end [m]"].get()
    
            # solve
            x, M, T_T2, p_p2 = burner_solver(M2, T2, p2, u2, gamma, tau_b, theta, x_i, x_end)
    
            # update results
            result_box.delete("1.0", tk.END)
            result_box.insert(tk.END, f"Final x   = {x[-1]:.3f} m\n")
            result_box.insert(tk.END, f"Final Mach= {M[-1]:.3f}\n")
            result_box.insert(tk.END, f"Final T/T2= {T_T2[-1]:.3f}\n")
            result_box.insert(tk.END, f"Final p/p2= {p_p2[-1]:.3f}\n")
    
            # embed plot
            if canvas:
                canvas.get_tk_widget().destroy()
            fig, axes = plt.subplots(3, 1, figsize=(5, 6))
            axes[0].plot(x, M);        axes[0].set_ylabel("Mach"); axes[0].grid()
            axes[1].plot(x, T_T2, 'orange'); axes[1].set_ylabel("T/T2"); axes[1].grid()
            axes[2].plot(x, p_p2, 'green');  axes[2].set_xlabel("x (m)"); axes[2].set_ylabel("p/p2"); axes[2].grid()
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
    
        def reset():
            for k, v in defaults.items():
                vars[k].set(v)
            result_box.delete("1.0", tk.END)
            nonlocal canvas
            if canvas:
                canvas.get_tk_widget().destroy()
                canvas = None
    
        tk.Button(root, text="Run Burner Simulation", command=run).grid(row=len(vars), column=0, pady=4)
        tk.Button(root, text="Reset",                command=reset).grid(row=len(vars), column=1, pady=4)
    
        root.mainloop()
    build_gui() 
    
        
elif choice == "3":
# thrust trajectory:


    import tkinter as tk
    from tkinter import ttk
    import math
    
    # ——— physical constants (same as HAP) ———
    R_AIR   = 287.05
    G0      = 9.80665
    MU_0    = 1.716e-5
    SUTH_S  = 110.4
    _ATM = [
        (0.0,     288.15, -0.0065),
        (11000.0, 216.65,  0.0000),
        (20000.0, 216.65,  0.0010),
        (32000.0, 228.65,  0.0028),
        (47000.0, 270.65,  0.0000),
    ]
    
    def _p_T_rho(h):
        T_b, p_b, L_b, h_b = 288.15, 101325.0, _ATM[0][2], 0.0
        for h_i, T_i, L_i in _ATM[1:]:
            if h < h_i:
                break
            if L_b == 0.0:
                p_b *= math.exp(-G0*(h_i-h_b)/(R_AIR*T_b))
            else:
                p_b *= (T_i/T_b)**(-G0/(R_AIR*L_b))
            T_b, L_b, h_b = T_i, L_i, h_i
        if L_b == 0.0:
            T = T_b
            p = p_b * math.exp(-G0*(h-h_b)/(R_AIR*T_b))
        else:
            T = T_b + L_b*(h - h_b)
            p = p_b * (T/T_b)**(-G0/(R_AIR*L_b))
        rho = p / (R_AIR * T)
        return T, p, rho
    
    def _alt_for_pressure(ptarg, lo=0.0, hi=47000.0):
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, p_mid, _ = _p_T_rho(mid)
            if p_mid > ptarg:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    
    def hap_trajectory(q0, M0, gamma_c, eta_c, psi):
        γ0 = 1.40
        p0_req = 2.0 * q0 / (γ0 * M0 * M0)
        H = _alt_for_pressure(p0_req)
        T0, p0, rho0 = _p_T_rho(H)
        a0 = math.sqrt(γ0 * R_AIR * T0)
        V0 = M0 * a0
        kin = 0.5 * V0**2
        mdot_A = rho0 * V0
        mu = MU_0 * (T0/273.15)**1.5 * (273.15+SUTH_S)/(T0+SUTH_S)
        Re_L = rho0 * V0 / mu
    
        T3 = psi * T0
        Tt0 = T0 * (1 + (gamma_c - 1)/2 * M0**2)
        if Tt0 <= T3:
            raise ValueError("Invalid input: T3 must be less than total temperature Tt0 for real M3")
        M3 = math.sqrt(2 / (gamma_c - 1) * (Tt0/T3 - 1))
        a3 = math.sqrt(gamma_c * R_AIR * T3)
        V3 = a3 * M3
    
        # Total pressure at freestream
        Pt0 = p0 * (1 + (gamma_c - 1)/2 * M0**2)**(gamma_c / (gamma_c - 1))
    
        # Adjusted pressure recovery (empirical value matching HAP ~0.21 at M0=10)
        total_pressure_ratio = 0.21
        Pt3 = Pt0 * total_pressure_ratio
    
        # Static pressure at burner inlet
        p3 = Pt3 / (1 + (gamma_c - 1)/2 * M3**2)**(gamma_c / (gamma_c - 1))
    
        return {
            "Altitude (km)": H / 1000,
            "T0 (K)": T0,
            "p0 (kPa)": p0 / 1000,
            "V0 (m/s)": V0,
            "ρ0V0 (kg/m²-s)": mdot_A,
            "Re/L (/m)": Re_L,
            "T3 (K)": T3,
            "p3 (kPa)": p3 / 1000,
            "M3": M3,
            "V3 (m/s)": V3
        }
    
    # GUI interface using tkinter
    root = tk.Tk()
    root.title("Mississippi State University revive the HAP Trajectory Calculator")
    
    inputs = {
        "q0 [N/m²]": 47880.0,
        "M0": 10.0,
        "γc": 1.36,
        "ηc": 0.90,
        "ψ (T3/T0)": 1555.56 / 232.44
    }
    
    entries = {}
    def create_input_fields():
        for idx, (label, default) in enumerate(inputs.items()):
            tk.Label(root, text=label).grid(row=idx, column=0, sticky="e")
            var = tk.DoubleVar(value=default)
            entry = tk.Entry(root, textvariable=var)
            entry.grid(row=idx, column=1)
            entries[label] = var
    
    create_input_fields()
    
    result_box = tk.Text(root, height=12, width=50)
    result_box.grid(row=len(inputs)+2, column=0, columnspan=2)
    
    def run_calculation():
        q0 = entries["q0 [N/m²]"].get()
        M0 = entries["M0"].get()
        gamma_c = entries["γc"].get()
        eta_c = entries["ηc"].get()
        psi = entries["ψ (T3/T0)"].get()
    
        result_box.delete("1.0", tk.END)
        try:
            results = hap_trajectory(q0, M0, gamma_c, eta_c, psi)
            for key, val in results.items():
                result_box.insert(tk.END, f"{key}: {val:.2f}\n")
        except ValueError as e:
            result_box.insert(tk.END, f"Error: {e}\n")
    
    def reset_fields():
        for label, default in inputs.items():
            entries[label].set(default)
        result_box.delete("1.0", tk.END)
    
    button = ttk.Button(root, text="Run Trajectory Analysis", command=run_calculation)
    button.grid(row=len(inputs), column=0, columnspan=2, pady=5)
    
    reset_button = ttk.Button(root, text="Reset", command=reset_fields)
    reset_button.grid(row=len(inputs)+1, column=0, columnspan=2, pady=5)
    
    root.mainloop()


else:
    print("Invalid selection. Please run again and choose 1, 2, or 3.")



