import numpy as np
import pandas as pd
import io
from flask import Flask, render_template, request

app = Flask(__name__)

# Db
battery_df = pd.read_excel("CellDatabase.xlsx")
supercap_df = pd.read_excel("EDLCDatabase.xlsx")

def generate_power_signal(duration=10, dt=0.1):
    # synthetic Signal
    t = np.arange(0, duration, dt)
    power_signal = 600 + 400* np.sin(2 * np.pi * 0.2 * t) + 10 * (t > 5)  #power required signal
    return t, power_signal


def compute_energy_capacity(power_signal, t, cut, total_energy_req, total_power_req):
    # max power and capacity
    Ps = np.max(power_signal)  # Peak of power signal
    Es = np.max(np.cumsum(power_signal) * (t[1] - t[0]))  # Total energy capacity, maximum integral of power signal 

    # Power battery and Power sc, cut is the ratio of power split btwn Pb and Pp, ie Pb = X*ps
    Pb = cut * total_power_req  # Bat
    Pp = (1 - cut) * total_power_req  # SC

    print(f"Base power (Pb): {Pb}, Peak power (Pp): {Pp}")
    print(f"Power Signal :{power_signal}")
    P_residual = np.clip(power_signal - Pb, 0, None)  # Residual power

    print(f"Residual power (P_residual): {P_residual}")

    Ep = np.max(np.cumsum(P_residual) * (t[1] - t[0]))  # Peak energy, minimum requirement to satisfy power signal 

    
    #Eb = max(total_energy_req - Ep, 0)  


    Eb = Es - Ep
    # total energy constraint: Eb + Ep >= total_energy_req
    if Eb + Ep < total_energy_req:
        Eb = total_energy_req - Ep  

    print(f"Base energy (Eb): {Eb}, Peak energy (Ep): {Ep}, Total energy (Es): {Es}")

    return Pb, Pp, Eb, Ep, Es


def compute_battery_energy_capacity(row):
    try:
        return row["Max Capacity (AH)"] * row["Nominal Voltage (V)"]
    except:
        return 0

def compute_battery_power_output(row):
    try:
        return row["Max Constant Discharge Current (A)"] * row["Nominal Voltage (V)"]
    except:
        return 0

def compute_supercap_power_output(row):
    try:
        voltage = float(str(row["Voltage - Rated"]).replace("V", "").strip())
        esr_str = str(row["ESR (Equivalent Series Resistance)"]).split("@")[0].strip()
        if "Ohm" in esr_str:
            resistance = float(esr_str.replace("Ohm", "").replace("m", "")) / (1000 if "m" in esr_str else 1)
            return (voltage ** 2) / resistance
        return 0
    except:
        return 0
    
def compute_supercap_energy_capacity(row):
    try:
        capacitance = float(str(row["Capacitance"]).replace("F", "").strip())
        voltage = float(str(row["Voltage - Rated"]).replace("V", "").strip())
        return 0.5 * capacitance * voltage**2 / 3600  # J to Wh
    except:
        return 0


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        total_energy = float(request.form['total_energy']) * 1000  # kWh to Wh
        cut = float(request.form['cut_parameter'])

        uploaded_file = request.files.get('power_csv')
        print(uploaded_file)
        if uploaded_file and uploaded_file.filename != '' and uploaded_file.filename.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 2:
                t = df.iloc[:, 0].values
                power_signal = df.iloc[:, 1].values
                total_power_req = np.max(power_signal)
            elif df.shape[1] == 1:
                power_signal = df.iloc[:, 0].values
                dt = 0.1
                t = np.arange(0, len(power_signal) * dt, dt)
                total_power_req = np.max(power_signal)  
            else:
                return "CSV must have 1 or 2 columns only", 400
        else:
            t, power_signal = generate_power_signal()
            total_power_req = float(request.form['total_power']) * 1000    # kW to W
        Pb, Pp, Eb, Ep, Es = compute_energy_capacity(power_signal, t, cut, total_energy, total_power_req)

        battery_df["EnergyCapacity"] = battery_df.apply(compute_battery_energy_capacity, axis=1)
        battery_df["PowerOutput"] = battery_df.apply(compute_battery_power_output, axis=1)

        supercap_df["PowerOutput"] = supercap_df.apply(compute_supercap_power_output, axis=1)
        supercap_df["EnergyCapacity"] = supercap_df.apply(compute_supercap_energy_capacity, axis=1)


        # Add unit count and price estimate
        battery_df["UnitsRequired"] = np.ceil(np.maximum(Eb / battery_df["EnergyCapacity"], Pb / battery_df["PowerOutput"]))
        battery_df["TotalPrice"] = battery_df["UnitsRequired"] * battery_df.get("Price(INR)", 0)

        supercap_df["UnitsRequired"] = np.ceil(np.maximum(Ep / supercap_df["EnergyCapacity"], Pp / supercap_df["PowerOutput"]))
        supercap_df["TotalPrice"] = supercap_df["UnitsRequired"] * supercap_df.get("Price", 0)

        # Filter those that can meet both constraints
        filtered_batteries = battery_df[
            (battery_df["EnergyCapacity"] * battery_df["UnitsRequired"] >= Eb) &
            (battery_df["PowerOutput"] * battery_df["UnitsRequired"] >= Pb) &
            (battery_df["UnitsRequired"] <= 10)
        ].sort_values(by="TotalPrice").head(10).drop(columns=["Image"], errors="ignore")

        filtered_supercaps = supercap_df[
            (supercap_df["EnergyCapacity"] * supercap_df["UnitsRequired"] >= Ep) &
            (supercap_df["PowerOutput"] * supercap_df["UnitsRequired"] >= Pp) &
            (supercap_df["UnitsRequired"]<= 20)
        ].sort_values(by="TotalPrice").head(10).drop(columns=["Image"], errors="ignore")



        def hyperlink_datasheet(df):
            if 'Datasheet' in df.columns:
                df['Datasheet'] = df['Datasheet'].apply(lambda url: f'<a href="{url}" target="_blank">Datasheet</a>' if pd.notna(url) else '')
            return df

        filtered_batteries = hyperlink_datasheet(filtered_batteries)
        filtered_supercaps = hyperlink_datasheet(filtered_supercaps)

        best_battery = filtered_batteries.iloc[0] if not filtered_batteries.empty else None
        best_supercap = filtered_supercaps.iloc[0] if not filtered_supercaps.empty else None

        recommended_bom = {
            "battery": {
                "company": best_battery.get("Company Name", ""),
                "part": best_battery.get("Part #", ""),
                "units": int(best_battery.get("UnitsRequired", 0)),
                "price": round(best_battery.get("TotalPrice", 0), 2)
            } if best_battery is not None else None,
            "supercap": {
                "datasheet": best_supercap.get("Datasheet", ""),
                "mfr": best_supercap.get("Mfr", ""),
                "desc": best_supercap.get("Description", ""),
                "units": int(best_supercap.get("UnitsRequired", 0)),
                "price": round(best_supercap.get("TotalPrice", 0), 2)
            } if best_supercap is not None else None,
            "total_price": round(
                (best_battery.get("TotalPrice", 0) if best_battery is not None else 0) +
                (best_supercap.get("TotalPrice", 0) if best_supercap is not None else 0), 2
            )
        }
        
        return render_template("results.html",
            recommended_bom=recommended_bom,
            batteries=filtered_batteries.to_dict(orient='records'),
            supercaps=filtered_supercaps.to_dict(orient='records'),
            hybrid_params={
                "base_energy": Eb / 1000,
                "base_power": Pb / 1000,
                "peak_energy": Ep / 1000,
                "peak_power": Pp / 1000,
                "total_energy": Es / 1000,
                "total_power": total_power_req / 1000,
                "synthetic_curve": power_signal.tolist()  
            })

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
