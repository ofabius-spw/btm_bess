import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import itertools

# -----------------------
# Helper functions
# -----------------------

def get_user_inputs():
    st.sidebar.header("Simulation settings")
    ptu_count = st.sidebar.number_input("PTUs per day (96 => 15 min)", value=96, min_value=1)
    dt = 24.0 / float(ptu_count)

    st.sidebar.subheader("Battery sizing")
    max_power_kw = st.sidebar.number_input("Battery max power (kW)", value=20.0)
    duration_hrs = st.sidebar.number_input("Cycle duration (h)", value=2.0)
    capacity_kwh = max_power_kw * duration_hrs

    st.sidebar.subheader("Efficiencies")
    eta_ch = st.sidebar.number_input("Charging/discharging efficiency ", value=0.95, min_value=0.5, max_value=1.0, step=0.01)

    st.sidebar.subheader("SoC / operational settings")
    soc_init = st.sidebar.number_input("Start SoC (kWh)", value=capacity_kwh * 0.5)
    soc_min = st.sidebar.number_input("Minimum SoC (kWh)", value=0.0, min_value=0.0, max_value=capacity_kwh)
    end_of_day_soc_enforced = st.sidebar.checkbox("Enforce end-of-day SoC = start SoC", value=True)

    st.sidebar.subheader("Grid limits & fees")
    import_cap = st.sidebar.number_input("Import cap (kW)", value=50)
    export_cap = st.sidebar.number_input("Export cap (kW)", value=50)
    grid_energy_fee = st.sidebar.number_input("Grid energy fee (€/kWh)", value=0.1)
    demand_charge = st.sidebar.number_input("Demand charge (€/kW)", value=1.0)

    st.sidebar.subheader("Objective prices")
    use_fixed_feedin = st.sidebar.checkbox("Use single fixed feed-in price", value=True)
    fixed_feedin = st.sidebar.number_input("Fixed feed-in price (€/kWh)", value=0.05)

    st.sidebar.subheader("Misc")
    epsilon = st.sidebar.number_input("Penalty ε", value=1e-4, step=1e-6, format="%f")

    # Keep uploader out of sidebar per your request (we'll use main page uploader).
    return dict(ptu_count=ptu_count, dt=dt, max_power_kw=max_power_kw, duration_hrs=duration_hrs, capacity_kwh=capacity_kwh,
                eta_ch=eta_ch, eta_dis=eta_dis, soc_init=soc_init, soc_min=soc_min, end_of_day_soc_enforced=end_of_day_soc_enforced,
                import_cap=import_cap, export_cap=export_cap, grid_energy_fee=grid_energy_fee, demand_charge=demand_charge,
                use_fixed_feedin=use_fixed_feedin, fixed_feedin=fixed_feedin, epsilon=epsilon)

def generate_mock_profiles(T, pv_max_kw, base_load, daytime_peak, evening_peak, avg_price, use_fixed_feedin, fixed_feedin):
    hours = (np.arange(T) / T) * 24.0
    mu, sigma = 13.0, 3.0
    pv_profile = pv_max_kw * np.maximum(0, np.exp(-0.5 * ((hours - mu) / sigma) ** 2))

    load_profile = np.full(T, base_load)
    for i, h in enumerate(hours):
        if 6 <= h < 10:
            load_profile[i] = base_load + (daytime_peak - base_load) * ((h - 6) / 4)
        elif 10 <= h < 18:
            load_profile[i] = daytime_peak
        elif 18 <= h <= 20:
            load_profile[i] = daytime_peak + (evening_peak - daytime_peak) * ((h - 18) / 2)
        else:
            load_profile[i] = base_load

    price_profile = avg_price + 0.05 * np.sin(2 * np.pi * (hours - 14) / 24)
    feedin_profile = np.full(T, fixed_feedin) if use_fixed_feedin else price_profile * 0.5

    return load_profile, pv_profile, price_profile, feedin_profile

def generate_full_year_profiles(T, pv_max_kw, base_load, daytime_peak, evening_peak, avg_price, use_fixed_feedin, fixed_feedin):
    # single-day
    day_load, day_pv, day_price, day_feedin = generate_mock_profiles(
        T, pv_max_kw, base_load, daytime_peak, evening_peak, avg_price, use_fixed_feedin, fixed_feedin
    )

    # tile for 365 days
    load_year = np.tile(day_load, 365)
    pv_year = np.tile(day_pv, 365)
    price_year = np.tile(day_price, 365)
    feedin_year = np.tile(day_feedin, 365)

    # optional small daily noise
    noise_scale = 0.05
    for i in range(365):
        idx = slice(i*T, (i+1)*T)
        load_year[idx] *= 1 + noise_scale * (np.random.rand(T)-0.5)
        pv_year[idx] *= 1 + noise_scale * (np.random.rand(T)-0.5)
        price_year[idx] *= 1 + noise_scale * (np.random.rand(T)-0.5)

    return load_year, pv_year, price_year, feedin_year



def build_and_solve_lp(T, dt, load_profile, pv_profile, price_profile, feedin_profile, grid_fee_profile, params):
    prob = LpProblem("BESS_day_opt_linear", LpMinimize)

    # Single battery power variable per PTU
    P = [LpVariable(f"P_{t}", lowBound=-params['max_power_kw'], upBound=params['max_power_kw']) for t in range(T)]

    # Import/export to grid
    Import = [LpVariable(f"Import_{t}", lowBound=0, upBound=params['import_cap']) for t in range(T)]
    Export = [LpVariable(f"Export_{t}", lowBound=0, upBound=params['export_cap']) for t in range(T)]

    # Peak import for demand charges
    Peak = LpVariable("PeakImport", lowBound=0)

    # SoC
    SoC = [LpVariable(f"SoC_{t}", lowBound=params['soc_min'], upBound=params['capacity_kwh']) for t in range(T)]

    # SoC dynamics (linear, symmetric efficiency)
    for t in range(T):
        if t == 0:
            prob += SoC[0] == params['soc_init'] + params['eta_ch'] * P[0] * dt
        else:
            prob += SoC[t] == SoC[t-1] + params['eta_ch'] * P[t] * dt
    if params['end_of_day_soc_enforced']:
        prob += SoC[T-1] == params['soc_init']

    # Grid constraints
    for t in range(T):
        prob += Import[t] - Export[t] == load_profile[t] - pv_profile[t] + P[t]
        prob += Peak >= Import[t]

    # Objective
    obj_terms = [
        lpSum(Import[t] * price_profile[t] * dt + 
            Export[t] * feedin_profile[t] * dt + 
            Import[t] * grid_fee_profile[t] * dt
            )
        for t in range(T)
    ]
    obj_terms.append(params['demand_charge'] * Peak)
    prob += lpSum(obj_terms)
    
    prob.solve()

    results = build_and_solve_lp(
        T=T_day,
        dt=params['dt'],
        load_profile=load,
        pv_profile=pv,
        price_profile=price,
        feedin_profile=feedin,
        grid_fee_profile=grid_fee,  # new argument
        params=params_copy
    )
    return results

def run_daily_optimisation(size_kw, load, pv, price, feedin, params):
    """
    Run the single-day optimisation for a given battery size.
    """
    # Override battery sizing
    params_copy = params.copy()
    params_copy['max_power_kw'] = size_kw
    params_copy['capacity_kwh'] = size_kw * params_copy['duration_hrs']

    results = build_and_solve_lp(
        T=len(load),
        dt=params_copy['dt'],
        load_profile=load,
        pv_profile=pv,
        price_profile=price,
        feedin_profile=feedin,
        params=params_copy
    )
    return results

def run_yearly_optimisation(sizes, df, params):
    """
    Run optimisation for each battery size across all days.
    Store detailed daily results only for the battery size with the best annual savings.
    """
    T_day = params['ptu_count']
    n_days = len(df) // T_day

    summary_records = []
    daily_records = []

    best_savings = -np.inf
    best_size = None
    detailed_results = {}  # only store daily results for best size

    progress = st.progress(0)
    status_text = st.empty()
    total_steps = len(sizes) * n_days
    step = 0

    for size in sizes:
        total_cost_with_batt = 0.0
        total_cost_no_batt = 0.0
        daily_results_temp = {}  # temporary store results for this size

        for d in range(n_days):
            idx = slice(d * T_day, (d + 1) * T_day)
            load = df['load'].values[idx]
            pv = df['pv'].values[idx]
            price = df['use_price'].values[idx]
            feedin = df['inject_price'].values[idx]

            # --- no battery cost
            net_no_batt = load - pv
            import_no_batt = np.maximum(net_no_batt, 0)
            export_no_batt = np.maximum(-net_no_batt, 0)
            cost_no_batt = (
                np.sum(import_no_batt * price * params['dt']) -
                np.sum(export_no_batt * feedin * params['dt']) +
                np.sum(import_no_batt * df['grid_fee'].values[idx] * params['dt']) +
                params['demand_charge'] * np.max(import_no_batt)
            )

            # --- with battery
            results = run_daily_optimisation(size, load, pv, price, feedin, params)
            cost_with_batt = results['objective']

            total_cost_no_batt += cost_no_batt
            total_cost_with_batt += cost_with_batt

            # store daily summary
            daily_records.append({
                "day": d + 1,
                "size_kw": size,
                "cost_no_batt": cost_no_batt,
                "cost_with_batt": cost_with_batt,
                "savings": cost_no_batt - cost_with_batt,
                "peak_import_no_batt": np.max(import_no_batt),
                "peak_import_with_batt": results['Peak']
            })

            # temporarily store detailed results for this size
            daily_results_temp[d+1] = results

            # update progress bar
            step += 1
            progress.progress(step / total_steps)
            status_text.text(f"Battery size: {size} kW — Day {d+1} of {n_days}")

        # --- end of size, check if it's the best
        annual_savings = total_cost_no_batt - total_cost_with_batt
        if annual_savings > best_savings:
            best_savings = annual_savings
            best_size = size
            detailed_results = daily_results_temp.copy()

        # store summary
        summary_records.append({
            "size_kw": size,
            "annual_cost_no_batt": total_cost_no_batt,
            "annual_cost_with_batt": total_cost_with_batt,
            "annual_savings": annual_savings
        })

    status_text.text(f"✅ Optimisation complete — Best size: {best_size} kW")
    progress.empty()

    summary_df = pd.DataFrame(summary_records)
    daily_df = pd.DataFrame(daily_records)

    return summary_df, daily_df, detailed_results, best_size



def plot_results(T, dt, load_profile, pv_profile, results):
    ts = np.arange(T) * dt
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(ts, load_profile, label='Load (kW)')
    ax[0].plot(ts, pv_profile, label='PV (kW)')
    ax[0].plot(ts, load_profile - pv_profile, label='Net before battery (kW)')
    ax[0].legend(); ax[0].set_ylabel('kW')

    ax[1].plot(ts, results['P'], label='Battery power (kW) — +charging, -discharging')
    ax[1].legend(); ax[1].set_ylabel('kW')

    ax[2].plot(ts, results['Import'], label='Import (kW)')
    ax[2].plot(ts, results['Export'], label='Export (kW)')
    ax[2].plot(ts, results['SoC'], label='SoC (kWh)')
    ax[2].legend(); ax[2].set_ylabel('kW / kWh'); ax[2].set_xlabel('Hour')
    return fig

# -----------------------
# Streamlit app
# -----------------------

st.set_page_config(layout="wide", page_title="BESS optimisation prototype")
st.title("Behind-the-meter BESS — prototype")

params = get_user_inputs()
T = int(params['ptu_count'])
load_arr = np.zeros(T)
pv_arr = np.zeros(T)
price_arr = np.zeros(T)
feedin_arr = np.zeros(T)
grid_fee_arr = np.zeros(T)
dt = params['dt']

# --- Unified data input (upload or generate) ---
st.header("1. Input data")

if 'df' not in st.session_state:
    st.session_state['df'] = None

choice = st.radio("Choose data source:", ["Upload CSV", "Generate synthetic"])

if choice == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload yearly CSV (expected 365 × PTUs rows). Required columns: load, pv, use_price, inject_price, grid_fee",
        type=["csv"]
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state['df'] = df
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif choice == "Generate synthetic":
    col1, col2 = st.columns(2)
    with col1:
        pv_max_kw = st.number_input("PV max power (kW)", value=80.0)
        base_load = st.number_input("Base load (kW)", value=20.0)
        daytime_peak = st.number_input("Daytime peak (kW)", value=60.0)
    with col2:
        evening_peak = st.number_input("Evening peak (kW)", value=80.0)
        avg_price = st.number_input("DA average price (€/kWh)", value=0.20)

    if st.button("Generate synthetic full-year data"):
        load_year, pv_year, price_year, feedin_year = generate_full_year_profiles(
            T, pv_max_kw, base_load, daytime_peak, evening_peak, avg_price,
            params['use_fixed_feedin'], params['fixed_feedin']
        )
        st.session_state['df'] = pd.DataFrame({
            'load': load_year,
            'pv': pv_year,
            'use_price': price_year,
            'inject_price': feedin_year,
            'grid_fee': np.full(365*T, params['grid_energy_fee'])
        })
        st.success("Synthetic full-year data generated.")

# --- Preview first day ---
df = st.session_state.get('df', None)
if df is not None:
    first_day = df.iloc[:T]
    st.subheader("Preview: first day")
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(first_day['load'].values, label='Load')
    ax.plot(first_day['pv'].values, label='PV')
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

run_type = st.radio("Optimisation horizon", ["First day only", "Full year"])
if run_type == "First day only":
    df_run = pd.DataFrame({
        'load': load_arr,
        'pv': pv_arr,
        'use_price': price_arr,
        'inject_price': feedin_arr,
        'grid_fee': grid_fee_arr
    })
else:
    df_run = df


# --- Battery size sweep (compact inputs on main page) ---
st.header("2. Battery size sweep (compact)")
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    min_size = st.number_input("Min size (kW)", value=10, step=1, min_value=1)
with col_b:
    max_size = st.number_input("Max size (kW)", value=50, step=1, min_value=1)
with col_c:
    step_size = st.number_input("Step (kW)", value=10, step=1, min_value=1)

battery_sizes = []
if step_size > 0 and max_size >= min_size:
    battery_sizes = list(range(int(min_size), int(max_size) + 1, int(step_size)))
else:
    st.warning("Please ensure Max >= Min and Step > 0")

if battery_sizes:
    st.write(f"Battery sizes to test ({len(battery_sizes)}): {battery_sizes}")
    if len(battery_sizes) > 10 and run_type == "Full year":
        st.warning(f"⚠️ Too many battery sizes selected ({len(battery_sizes)}). Please choose ≤ 10.")

st.markdown("---")
st.info("Use the 'Solve day optimisation' button below to run the single-day LP. The battery size sweep is recorded above for future multi-size runs.")
st.header("Run optimisation")
if st.button("Run Optimisation"):
    # --- Prepare DF for optimisation ---
    if df is not None:
        load_arr = df['load'].values[:T]
        pv_arr = df['pv'].values[:T]
        price_arr = df['use_price'].values[:T] if 'use_price' in df.columns else np.full(T, 0.20)
        feedin_arr = df['inject_price'].values[:T] if 'inject_price' in df.columns else np.full(T, params['fixed_feedin'])
        grid_fee_arr = df['grid_fee'].values[:T] if 'grid_fee' in df.columns else np.full(T, params['grid_energy_fee'])
    else:
        load_arr, pv_arr, price_arr, feedin_arr = generate_mock_profiles(
            T, 80.0, 20.0, 60.0, 80.0, 0.20, params['use_fixed_feedin'], params['fixed_feedin']
        )
        grid_fee_arr = np.full(T, params['grid_energy_fee'])

    # Wrap single-day as a mini DF if "First day only"
    if run_type == "First day only":
        df_run = pd.DataFrame({
            'load': load_arr,
            'pv': pv_arr,
            'use_price': price_arr,
            'inject_price': feedin_arr,
            'grid_fee': grid_fee_arr
        })
    else:
        df_run = df

    # --- Run yearly optimisation (1 day or full year) ---
    summary_df, daily_df, detailed_results, best_size = run_yearly_optimisation(battery_sizes, df_run, params)
    st.subheader("Battery size sweep results")
    st.dataframe(summary_df)
    st.markdown(f"**Best battery size:** {best_size} kW → savings: {summary_df['annual_savings'].max():.2f} €")

    # --- Select day to display ---
    day_to_show = st.number_input("Select day to display", 1)
    daily_display = daily_df[(daily_df['day'] == day_to_show) & (daily_df['size_kw'] == best_size)].iloc[0]

    # --- Show summary metrics ---
    st.subheader("Summary results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost without battery (€)", f"{daily_display['cost_no_batt']:.2f}")
    col2.metric("Total cost with battery (€)", f"{daily_display['cost_with_batt']:.2f}")
    col3.metric("Savings (€)", f"{daily_display['savings']:.2f}")

    st.write(f"Peak import (kW): {daily_display['peak_import_with_batt']:.2f}")

    # --- Plot results for selected day ---
    ts = np.arange(T) * params['dt']
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(ts, df_run['load'].values[:T], label='Load (kW)')
    ax[0].plot(ts, df_run['pv'].values[:T], label='PV (kW)')
    ax[0].plot(ts, df_run['load'].values[:T] - df_run['pv'].values[:T], label='Net before battery')
    ax[0].legend(); ax[0].set_ylabel('kW')

    # Battery power / SoC from daily_df stored in run_yearly_optimisation
    # If you don’t yet store P/SoC in daily_df, you can still call run_daily_optimisation for the selected day:
    results_day = detailed_results[day_to_show]


    ax[1].plot(ts, results_day['P'], label='Battery power (+charging, -discharging)')
    ax[1].legend(); ax[1].set_ylabel('kW')

    ax[2].plot(ts, results_day['Import'], label='Import')
    ax[2].plot(ts, results_day['Export'], label='Export')
    ax[2].plot(ts, results_day['SoC'], label='SoC (kWh)')
    ax[2].legend(); ax[2].set_ylabel('kW / kWh'); ax[2].set_xlabel('Hour')

    st.pyplot(fig)

    # --- Day-ahead price plot ---
    fig_price, ax_price = plt.subplots(figsize=(12, 3))
    ax_price.plot(ts, df_run['use_price'].values[:T], label='DA Price', color='tab:orange')
    ax_price.set_xlabel("Hour of day"); ax_price.set_ylabel("Price (€/kWh)")
    ax_price.set_title("Day-ahead energy price"); ax_price.grid(True); ax_price.legend()
    st.pyplot(fig_price)

    # --- Show results table ---
    results_df = pd.DataFrame({
        't': np.arange(T),
        'load_kW': df_run['load'].values[:T],
        'pv_kW': df_run['pv'].values[:T],
        'P_batt_kW': results_day['P'],
        'Import_kW': results_day['Import'],
        'Export_kW': results_day['Export'],
        'SoC_kWh': results_day['SoC'],
        'price_eur_kWh': df_run['use_price'].values[:T],
        'feedin_eur_kWh': df_run['inject_price'].values[:T],
    })
    st.subheader("Results table (first 50 rows)")
    st.dataframe(results_df.head(50))

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download results CSV", csv, "bess_day_results.csv", "text/csv")
