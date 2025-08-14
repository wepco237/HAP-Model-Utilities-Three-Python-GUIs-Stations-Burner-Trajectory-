# HAP-Model-Utilities-Three-Python-GUIs-Stations-Burner-Trajectory-
Revival and Development of Heiser–Pratt Hypersonic Propulsion Models Utilizing Python
Title: HAP Model Utilities — Three Python GUIs (Stations, Burner, Trajectory)

MAKE SURE to Run Each Module Seperately by Copy and Paste 

Overview
This package contains three small desktop applications built with Python and Tkinter. They are intended for quick exploration of hypersonic airbreathing propulsion ideas: station-by-station thrust metrics, a 1-D burner integrator, and a trajectory pre-calculator. Each tool opens as a simple GUI and prints results clearly.

1) Thrust Function GUI (two tabs: Constant Area, Constant Pressure)

What it does (brief):
Computes station properties and performance for a simple engine stream with losses and efficiencies. The Constant Area tab solves velocity and temperature at the burner exit from an energy balance. The Constant Pressure tab updates exit velocity directly and returns the corresponding temperature and area ratio.

Inputs (as shown in the GUI; names use plain words):

psi: temperature ratio from freestream to station 3

V0: initial velocity in m/s

T0: initial temperature in K

f: fuel to air ratio

hf: fuel sensible enthalpy offset (usually zero)

To: reference temperature in K

VTe_V3: exit to station-3 velocity ratio for the fuel jet term

Vf_V3: fuel to station-3 velocity ratio

Cf_Aw_A3: effective burner drag coefficient term

p10_p0: nozzle exit to freestream pressure ratio

eta_c: compressor efficiency

eta_b: burner efficiency

eta_e: nozzle efficiency

fhPR: fuel heating value in J/kg

R: specific gas constant in (m/s)^2 per K

Cpc, Cpb, Cpe: specific heat at constant pressure for compressor leg, burner leg, and exit, in J/kg-K

gamma_c, gamma_b, gamma_e: ratio of specific heats for compressor leg, burner leg, and exit

g0: gravitational acceleration in m/s^2

Outputs (printed in the right panel; names use plain words):

Sa0, Sa3, Sa4, Sa10: thrust function at stations 0, 3, 4, 10 in N·s/kg

T3, T4, T10: temperatures at stations 3, 4, 10 in K

V3, V4, V10: velocities at stations 3, 4, 10 in m/s

p3_p0: station-3 to freestream pressure ratio

A3_A0: station-3 to freestream area ratio

A4_A3: station-4 to station-3 area ratio (Constant Pressure tab only)

A10_A0: station-10 to freestream area ratio

F_mdot0: thrust per unit freestream mass flow in N·s/kg

Isp: specific impulse in s

eta_o: overall efficiency (unitless)

2) Burner GUI (1-D RK4 integrator with heat addition and area change)

What it does (brief):
Solves the axial evolution of Mach number through a burner with changing total temperature and area. Total temperature follows a smooth heating profile. Area grows linearly. The GUI displays final values and plots Mach, temperature ratio, and pressure ratio along the length.

Inputs (as shown in the GUI; names use plain words):

p2: station-2 static pressure in kPa

T2: station-2 static temperature in K

M2: station-2 Mach number

Rb: gas constant placeholder in J/kg-K (kept for completeness)

u2: station-2 velocity in m/s

theta_over_H: placeholder for heat input per height (kept for completeness)

A3_A2, A4_A2: geometry ratios placeholders (kept for future geometry linkage)

H: burner height in m (not used in the 1-D solver)

tau_b: total temperature ratio across the burner

theta: profile shaping parameter for heating distribution

x_i: axial start location in m

x_end: axial end location in m

Outputs (shown in the text box and plots):

Final Mach at x_end

Final temperature ratio relative to T2

Final pressure ratio relative to p2

Plots vs axial distance: Mach, temperature ratio, pressure ratio

3) Trajectory GUI (ISA atmosphere and inlet pre-calculator)

What it does (brief):
Finds an altitude that meets a target dynamic pressure and a given flight Mach number. Returns freestream properties, estimated burner-inlet pressure and temperature ratio, and predicted station-3 Mach and velocity with a simple pressure-recovery assumption.

Inputs (as shown in the GUI; names use plain words):

q0: target dynamic pressure in N/m^2

M0: flight Mach number

gamma_c: ratio of specific heats for compressor leg

eta_c: compressor efficiency (reserved for future use)

psi: temperature ratio from freestream to station 3

Outputs (listed in the results box):

Altitude in km

T0: freestream temperature in K

p0: freestream pressure in kPa

V0: freestream velocity in m/s

rho0V0: mass flux in kg/m^2-s

Re_per_L: Reynolds number per meter

T3: station-3 temperature in K

p3: station-3 static pressure in kPa

M3: station-3 Mach number

V3: station-3 velocity in m/s

How to Run
Option A — Run locally on your computer (recommended for Tkinter windows)

Install Python 3.9 or newer.

(Optional) Create and activate a virtual environment.

Install required libraries:

pip install numpy matplotlib


Most Python distributions include Tkinter by default. If not:

Windows: install the standard Python from python.org.

Ubuntu/Debian: sudo apt-get install python3-tk

Fedora: sudo dnf install python3-tkinter

macOS (Homebrew Python): brew install python-tk@3.x (match your Python version)

Save each GUI as its own file and run:

python thrust_function_gui.py
python burner_gui.py
python trajectory_gui.py

Option B — Use online tools

Tkinter is a desktop GUI toolkit, so pure browser notebooks are not ideal. These options can work:

Replit (Tkinter template): Create a new Python repl using the “Tkinter” template. Paste the code and click Run.

GitHub Codespaces with VS Code desktop forwarding or X11: Advanced option; requires GUI forwarding setup.

Local Jupyter Notebook: You can run the burner solver core inside a notebook and use matplotlib inline, but the Tkinter windows themselves are best run as standalone scripts.

If you need a non-GUI version for browser notebooks, I can provide CLI wrappers that print results and skip window creation.

Notes and Good Practices

Units: keep inputs and outputs in the units listed above to avoid invalid states.

Physical validity: if you see messages like “negative value inside square root,” adjust inputs to keep energy balances and total/static temperature relationships physically consistent.

Plots not appearing: make sure you are running as a normal Python script, not inside a restricted environment.
