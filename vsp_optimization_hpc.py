import os
import csv
import subprocess
import pyvista as pv
import numpy as np
import glob
import uuid
from scipy.optimize import differential_evolution, NonlinearConstraint

vsp_exe = r"/home/prahaas123/OpenVSP/vsp"

wing_span_res = 20
wing_chord_res = 50
velocity = 10 # m/s
alpha = 3 # degrees AoA

airfoil_file = r"/home/prahaas123/openvsp_scripts/Airfoils/naca4415.dat"

MAX_WEIGHT = 5   # Newtons
MIN_S_REF = 0.13  # m2
MAX_S_REF = 0.21  # m2
STATIC_MARGIN = 0.05
CM_MIN = -0.15   # lower bound on CM about CG
CM_MAX =  0.15   # upper bound on CM about CG
AR_MIN = 3.0
AR_MAX = 6.0
TIP_CHORD_MIN = 0.05

LOG_CSV = "optimization_results.csv"
LOG_FIELDS = [
    "run_id",
    "root_chord", "taper", "sweep", "twist", "span",
    "LD", "CL", "CD", "CM_cg",
    "AR", "lift", "x_cg"
]

def main():   
    init_log()
    print("Starting SciPy DE Optimization...")
    bounds = [(0.2, 0.4),   # Root chord
              (0.1, 0.8),   # Taper ratio
              (20.0, 50.0), # Sweep angle
              (-10.0, 0.0), # Washout angle
              (0.4, 0.9)]   # Wingspan

    # Early-rejection geometric constraints
    geom_constraint = NonlinearConstraint(
        evaluate_geometry, 
        lb=[MIN_S_REF, AR_MIN, TIP_CHORD_MIN], 
        ub=[MAX_S_REF, AR_MAX, np.inf]
    )

    # Run SciPy DE
    try:
        result = differential_evolution(
            evaluate_aero_objective,
            bounds=bounds,
            constraints=(geom_constraint,),
            strategy='rand1bin',
            recombination=0.9,
            mutation=(0.5, 1.0),
            popsize=10,               # Population members = popsize * parameters 
            maxiter=30,              # n_max_gen
            tol=1e-6,                # ftol
            seed=1,
            disp=True
        )
        print(f"\nOptimization finished.")
    except KeyboardInterrupt:
        # This catches the Ctrl+C
        print("\n\nOptimization interrupted by user! Generating results from the best logged design so far...")

    # Look up values from the log (safest way to get the best strictly feasible run)
    row = lookup_best()
    if row is None:
        print("No feasible designs were logged — cannot print results summary.")
        return
        
    best_root   = float(row["root_chord"])
    best_taper  = float(row["taper"])
    best_sweep  = float(row["sweep"])
    best_twist  = float(row["twist"])
    best_span   = float(row["span"])
    ld          = float(row["LD"])
    cm          = float(row["CM_cg"])
    ar          = float(row["AR"])
    lift        = float(row["lift"])
    x_cg        = float(row["x_cg"])

    print("\n--- OPTIMAL WING GEOMETRY ---")
    print(f"\n--- AERODYNAMICS ---")
    print(f"L/D         : {ld:.4f}")
    print(f"CL          : {float(row['CL']):.4f}")
    print(f"CD          : {float(row['CD']):.4f}")
    print(f"CM_cg       : {cm:.4f}")
    print(f"Total lift  : {lift:.4f} N")
    print(f"\n--- GEOMETRY ---")
    print(f"Aspect ratio: {ar:.4f}  (min: {AR_MIN}  max: {AR_MAX})")
    print(f"CG          : {x_cg:.4f} m  (aft of root LE,  SM={STATIC_MARGIN*100:.0f}% MAC)")
    print(f"\n--- PARAMETERS ---")
    print(f"Params      : Root={best_root:.4f}  Taper={best_taper:.4f}  Sweep={best_sweep:.4f}  Twist={best_twist:.4f}  Span={best_span:.4f}")
    print(f"\nFeasible designs logged to: {LOG_CSV}\n")
    
    # Generate STL
    stl_path, _ = generate_wing("Optimized_Wing", best_span * 1000, best_root * 1000, best_taper, best_sweep, 0.0, best_twist, airfoil_file)
    visualize_stl(stl_path)
    
def evaluate_geometry(x):
    root_chord, taper, sweep, twist, span = x
    Sref = 0.5 * (root_chord + root_chord * taper) * span
    AR = aspect_ratio(root_chord, taper, span)
    tip_chord = root_chord * taper
    return np.array([Sref, AR, tip_chord])

def evaluate_aero_objective(x):
    root_chord, taper, sweep, twist, span = x
    run_id = f"wing_{uuid.uuid4().hex[:8]}" 
    
    try:
        stl_path, analysis_path = generate_wing(run_id, span, root_chord, taper, sweep, 0.0, twist, airfoil_file)
        Sref = 0.5 * (root_chord + root_chord * taper) * span
        x_cg = calc_cg(root_chord, taper, span, sweep)
        aero  = vsp_point(analysis_path, velocity, alpha, Sref, span, root_chord, x_cg)
        CL, CD, LD, CM_cg = aero["CL"], aero["CD"], aero["LD"], aero["CM"]
        lift  = 0.5 * CL * 1.225 * velocity**2 * Sref
        AR    = aspect_ratio(root_chord, taper, span)
        
        # Check Aerodynamic Constraints
        penalty = 0
        is_feasible = True
        if lift < MAX_WEIGHT:
            penalty += 1e5 * (MAX_WEIGHT - lift) # Scale penalty by how badly it failed
            is_feasible = False
        if CM_cg < CM_MIN or CM_cg > CM_MAX:
            penalty += 1e5
            is_feasible = False

        if is_feasible:
            print(f"  [{run_id}] FEASIBLE! L/D={LD:.3f}")
            append_log({
                "run_id"      : run_id,
                "root_chord"  : round(float(root_chord),  6),
                "taper"       : round(float(taper),       6),
                "sweep"       : round(float(sweep),       6),
                "twist"       : round(float(twist),       6),
                "span"        : round(float(span),        6),
                "LD"          : round(LD,                 6),
                "CL"          : round(CL,                 6),
                "CD"          : round(CD,                 6),
                "CM_cg"       : round(CM_cg,              6),
                "AR"          : round(AR,                 6),
                "lift"        : round(lift,               6),
                "x_cg"        : round(x_cg,               6),
            })
        else:
            print(f"  [{run_id}] Failed Aero Constraints. Lift={lift:.2f}N, CM={CM_cg:.3f}")

        return -LD + penalty

    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        return 1e10 # Massive penalty for crashed runs
    finally:
        for filename in glob.glob(f"{run_id}*"):
            try:
                os.remove(filename)
            except OSError:
                pass

def generate_wing(wing_name, wingspan, root_chord, taper_ratio, sweep_angle, dihedral_angle, twist_angle, airfoil_file):
    tip_chord = root_chord * taper_ratio
    airfoil_fwd = airfoil_file.replace("\\", "/")

    script_lines = [
        "void main() {",
        "    VSPCheckSetup();",
        "    ClearVSPModel();",
        f'    string wing_id = AddGeom( "WING" );',
        f'    SetParmVal( wing_id, "TotalSpan",      "WingGeom", {wingspan} );',
        f'    SetParmVal( wing_id, "Root_Chord",     "XSec_1",   {root_chord} );',
        f'    SetParmVal( wing_id, "Tip_Chord",      "XSec_1",   {tip_chord} );',
        f'    SetParmVal( wing_id, "Sweep",          "XSec_1",   {sweep_angle} );',
        f'    SetParmVal( wing_id, "Dihedral",       "XSec_1",   {dihedral_angle} );',
        f'    SetParmVal( wing_id, "Twist",          "XSec_1",   {twist_angle} );',
        f'    SetParmVal( wing_id, "Twist_Location", "XSec_1",   1.0 );',
        f'    SetParmVal( wing_id, "SectTess_U",     "XSec_1",   {wing_span_res}.0 );',
        f'    SetParmVal( wing_id, "Tess_W",         "Shape",    {wing_chord_res}.0 );',
        f'    string root_surf = GetXSecSurf( wing_id, 0 );',
        f'    ChangeXSecShape( root_surf, 0, XS_FILE_AIRFOIL );',
        f'    string root_xsec = GetXSec( root_surf, 0 );',
        f'    ReadFileAirfoil( root_xsec, "{airfoil_fwd}" );',
        f'    string tip_surf = GetXSecSurf( wing_id, 1 );',
        f'    ChangeXSecShape( tip_surf, 1, XS_FILE_AIRFOIL );',
        f'    string tip_xsec = GetXSec( tip_surf, 1 );',
        f'    ReadFileAirfoil( tip_xsec, "{airfoil_fwd}" );',
        f'    SetSetFlag( wing_id, 1, true );',
        f'    Update();',
        f'    WriteVSPFile( "{wing_name}.vsp3", SET_ALL );',
        f'    ExportFile( "{wing_name}.stl", 0, EXPORT_STL );',
        "}",
    ]

    script_path = f"{wing_name}_geom.vspscript"
    with open(script_path, 'w') as f:
        f.write("\n".join(script_lines))

    print(f"--- Running geometry generation ({script_path}) ---")
    subprocess.run([vsp_exe, "-script", script_path], check=True)
    os.remove(script_path)

    stl_path = f"{wing_name}.stl"
    vsp3_path = f"{wing_name}.vsp3"
    print(f"STL generated: {stl_path}")
    print(f"VSP file saved: {vsp3_path}")
    return stl_path, vsp3_path

def visualize_stl(stl_path):
    if os.path.exists(stl_path):
        mesh = pv.read(stl_path)
        plotter = pv.Plotter(title="WatArrow Delta Wing")
        plotter.add_mesh(mesh, color="lightblue", show_edges=True, smooth_shading=True)
        plotter.add_axes()
        plotter.add_floor(face='-z', i_resolution=10, j_resolution=10, color='gray', opacity=0.2)
        print("Opening PyVista window...")
        plotter.show()
    else:
        print("Error: STL not found.")

def vsp_point(vsp3_path, vin, alpha, Sref, bref, cref, x_cg):
    mach = vin / 343.0

    script_lines = [
        "void main() {",
        f'    ClearVSPModel();',
        f'    ReadVSPFile( "{vsp3_path}" );',
        f'    SetAnalysisInputDefaults( "VSPAEROComputeGeometry" );',
        f'    array< int > thick_set = GetIntAnalysisInput( "VSPAEROComputeGeometry", "GeomSet" );',
        f'    array< int > thin_set = GetIntAnalysisInput( "VSPAEROComputeGeometry", "ThinGeomSet" );',
        f'    thick_set[0] = ( SET_TYPE::SET_NONE );',
        f'    thin_set[0] = ( SET_TYPE::SET_ALL );',
        f'    SetIntAnalysisInput( "VSPAEROComputeGeometry", "GeomSet", thick_set );',
        f'    SetIntAnalysisInput( "VSPAEROComputeGeometry", "ThinGeomSet", thin_set );',
        f'    Print( "--- Running Meshing ---" );',
        f'    ExecAnalysis( "VSPAEROComputeGeometry" );',
        f'    SetAnalysisInputDefaults( "VSPAEROSweep" );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "Sref",           {darr(Sref)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "cref",           {darr(cref)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "bref",           {darr(bref)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "Xcg",            {darr(x_cg)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "AlphaStart",     {darr(float(alpha))}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "AlphaEnd",       {darr(float(alpha))}, 0 );',
        f'    SetIntAnalysisInput(    "VSPAEROSweep", "AlphaNpts",      {iarr(1)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "MachStart",      {darr(mach)}, 0 );',
        f'    SetIntAnalysisInput(    "VSPAEROSweep", "MachNpts",       {iarr(1)}, 0 );',
        f'    SetDoubleAnalysisInput( "VSPAEROSweep", "Vinf",           {darr(100.0)}, 0 );',
        f'    SetIntAnalysisInput(    "VSPAEROSweep", "WakeNumIter",    {iarr(15)}, 0 );',
        f'    SetIntAnalysisInput(    "VSPAEROSweep", "NCPU",           {iarr(1)}, 0 );',
        f'    Print( "--- Running Aero Point ---" );',
        f'    SetStringAnalysisInput( "VSPAEROSweep", "RedirectFile", array<string> = {{"{vsp3_path}_log.txt"}}, 0 );'
        f'    ExecAnalysis( "VSPAEROSweep" );',
        "}",
    ]

    script_path = f"{vsp3_path.replace('.vsp3', '')}_aero.vspscript"
    with open(script_path, 'w') as f:
        f.write("\n".join(script_lines))

    print(f"--- Running aero point (alpha={alpha}) ---")
    subprocess.run([vsp_exe, "-script", script_path], check=True)
    os.remove(script_path)

    polar_file = vsp3_path.replace(".vsp3", ".polar")
    CL, CD, Cm = parse_polar(polar_file)
    cl = CL[0]
    cd = CD[0]
    cm = Cm[0]
    return {
        "CL": cl,
        "CD": cd,
        "LD": cl / cd,
        "CM": cm,
    }

def parse_polar(polar_path):
    CL, CD, Cm = [], [], []
    col_cl = col_cd = col_cm = None
 
    with open(polar_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            tokens = stripped.split()
            if tokens[0] == 'Beta':
                col_cl = tokens.index('CLtot')
                col_cd = tokens.index('CDtot')
                col_cm = tokens.index('CMytot')
                continue
            if col_cl is None:
                continue
            try:
                CL.append(float(tokens[col_cl]))
                CD.append(float(tokens[col_cd]))
                Cm.append(float(tokens[col_cm]))
            except (ValueError, IndexError):
                continue
 
    return CL, CD, Cm

def estimate_wetted_area(root_chord, taper, span):
    tip_chord     = root_chord * taper
    planform_area = 0.5 * (root_chord + tip_chord) * span
    return 2.0 * planform_area * 1.02

def calc_cg(root_chord, taper, span, sweep_angle, static_margin=STATIC_MARGIN):
    mac   = (2.0 / 3.0) * root_chord * (1 + taper + taper**2) / (1 + taper)
    y_mac = (span / 2.0) * (1 + 2 * taper) / (3 * (1 + taper))
    x_ac  = y_mac * np.tan(np.radians(sweep_angle)) + 0.25 * mac
    x_cg  = x_ac - static_margin * mac
    return x_cg

def aspect_ratio(root_chord, taper, span):
    tip_chord     = root_chord * taper
    planform_area = 0.5 * (root_chord + tip_chord) * span
    return span**2 / planform_area

def stall_speed(cl, sref, w=MAX_WEIGHT, rho=1.225):
    return np.sqrt((2*w) / (rho*sref*cl))

def root_bending_moment(root_chord, taper, span, CL, vin=velocity, rho=1.225):
    q = 0.5 * rho * vin**2
    s = span / 2.0
    return q * CL * root_chord * s**2 * (1 + 2 * taper) / 6

def init_log():
    with open(LOG_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

def append_log(row: dict):
    with open(LOG_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore").writerow(row)

def lookup_best():
    best = None
    with open(LOG_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if best is None or float(row["LD"]) > float(best["LD"]):
                best = row
    return best

# AngelScript array helpers
def iarr(v):
    return f"array<int> = {{{v}}}"

def darr(v):
    return f"array<double> = {{{v}}}"

if __name__ == "__main__":
    main()