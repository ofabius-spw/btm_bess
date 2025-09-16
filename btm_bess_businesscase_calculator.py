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
    eta_ch = st.sidebar.number_input("Charging efficiency (η_ch)", value=0.96, min_value=0.5, max_value=1.0, step=0.01)
    eta_dis = st.sidebar.number_input("Discharging efficiency (η_dis)", value=0.96, min_value=0.5, max_value=1.0, step=0.01)

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

    st.sidebar.subheader("Data / upload")
    uploaded = st.sidebar.file_uploader("Upload CSV with columns for load (kW) and pv (kW)", type=["csv"])

    return dict(ptu_count=ptu_count, dt=dt, max_power_kw=max_power_kw, duration_hrs=duration_hrs, capacity_kwh=capacity_kwh,
                eta_ch=eta_ch, eta_dis=eta_dis, soc_init=soc_init, soc_min=soc_min, end_of_day_soc_enforced=end_of_day_soc_enforced,
                import_cap=import_cap, export_cap=export_cap, grid_energy_fee=grid_energy_fee, demand_charge=demand_charge,
                use_fixed_feedin=use_fixed_feedin, fixed_feedin=fixed_feedin, epsilon=epsilon, uploaded=uploaded)

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
        elif h > 20 or h < 6:
            load_profile[i] = base_load

    price_profile = avg_price + 0.05 * np.sin(2 * np.pi * (hours - 14) / 24)
    feedin_profile = np.full(T, fixed_feedin) if use_fixed_feedin else price_profile * 0.5

    return load_profile, pv_profile, price_profile, feedin_profile

def build_and_solve_lp(T, dt, load_profile, pv_profile, price_profile, feedin_profile, params):
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
    obj_terms = []
    for t in range(T):
        obj_terms.append(Import[t] * price_profile[t] * dt)
        obj_terms.append(- Export[t] * feedin_profile[t] * dt)
        obj_terms.append(Import[t] * params['grid_energy_fee'] * dt)
    obj_terms.append(params['demand_charge'] * Peak)

    prob += lpSum(obj_terms)
    prob.solve()

    results = {
        'status': LpStatus[prob.status],
        'objective': value(prob.objective),
        'P': np.array([v.varValue for v in P]),
        'Import': np.array([v.varValue for v in Import]),
        'Export': np.array([v.varValue for v in Export]),
        'SoC': np.array([v.varValue for v in SoC]),
        'Peak': Peak.varValue,
    }
    return results

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
# -----------------------
# Scenario Sweeper Tab
# -----------------------

def scenario_sweeper_tab(params, load_profile, pv_profile, price_profile, feedin_profile):
    st.header("Scenario Sweeper")

    st.subheader("Select parameters to sweep")
    sweep_options = ['Battery max power', 'PV max power', 'Base load level', 'Grid energy fee', 'DA average price']
    sweep_selected = st.multiselect("Select parameters to sweep", options=sweep_options)

    sweep_ranges = {}
    default_steps = 5  # default number of steps

    # For each selected parameter, get min, max, step in columns
    for param in sweep_selected:
        if param == 'Battery max power':
            default_min, default_max = 20, 100
            use_float = False
        elif param == 'PV max power':
            default_min, default_max = 20, 120
            use_float = False
        elif param == 'Base load level':
            default_min, default_max = 20, 50
            use_float = False
        elif param == 'Grid energy fee':
            default_min, default_max = 0.0, 0.1
            use_float = True
        elif param == 'DA average price':
            default_min, default_max = 0.15, 0.25
            use_float = True
        else:
            default_min, default_max = 0, 1
            use_float = True

        col1, col2, col3 = st.columns(3)
        with col1:
            if use_float:
                min_val = st.number_input(f"{param} min", value=float(default_min), format="%.2f")
            else:
                min_val = st.number_input(f"{param} min", value=int(default_min))
        with col2:
            if use_float:
                max_val = st.number_input(f"{param} max", value=float(default_max), format="%.2f")
            else:
                max_val = st.number_input(f"{param} max", value=int(default_max))
        with col3:
            if use_float:
                step_val = st.number_input(f"{param} step", value=round((max_val - min_val)/default_steps, 2), format="%.2f")
            else:
                step_val = st.number_input(f"{param} step", value=(int(max_val) - int(min_val))//default_steps)

        # Check that (max-min) is multiple of step
        if step_val <= 0 or (max_val - min_val)/step_val != int((max_val - min_val)/step_val):
            st.warning(f"For {param}, the difference between min and max must be a multiple of the step.")
            return

        sweep_ranges[param] = (min_val, max_val, step_val)
    # Calculate all combinations
    sweep_values = {}
    for k, (vmin, vmax, step) in sweep_ranges.items():
        if k in ['Grid energy fee', 'DA average price']:
            sweep_values_key = np.round(np.arange(vmin, vmax + 0.0001, step), 2)
        else:
            sweep_values_key = np.arange(int(vmin), int(vmax) + 1, int(step))
        sweep_values[k] = sweep_values_key

    keys, values = zip(*sweep_values.items()) if sweep_values else ([], [])
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Limit number of combinations
    if len(combinations) > 1000:
        st.warning("Please select less than 1000 parameter combinations. Even I have my limits.")
        return
    elif len(combinations) > 100:
        st.warning("With more than 100 parameter combinations, this might be a little slow...")
    if len(combinations) == 0:
        st.info("Select at least one parameter to sweep.")
        return

    # Run sweep for each combination
    results_list = []
    for combo in combinations:
        local_params = params.copy()
        for k, v in combo.items():
            local_params[k] = v

        base_load = combo.get('base_load', np.mean(load_profile))
        avg_price = combo.get('avg_price', np.mean(price_profile))
        grid_fee = combo.get('grid_energy_fee', params['grid_energy_fee'])

        # generate PV if pv_max_kw changed
        pv_max = combo.get('pv_max_kw', None)
        if pv_max is not None:
            load_mock, pv_mock, price_mock, feedin_mock = generate_mock_profiles(
                len(load_profile), pv_max, base_load, 60, 80, avg_price,
                local_params['use_fixed_feedin'], local_params['fixed_feedin'])
        else:
            pv_mock = pv_profile
            price_mock = price_profile

        local_params['grid_energy_fee'] = grid_fee
        res = build_and_solve_lp(len(load_profile), params['dt'],
                                load_profile, pv_mock, price_mock, feedin_profile,
                                local_params)
        total_cost = res['objective']

        row = combo.copy()
        row['Total cost (€)'] = total_cost
        results_list.append(row)

    df_results = pd.DataFrame(results_list)


    # Plot costs vs one chosen parameter
    st.subheader("Plot: Total cost vs parameter")
    param_options = list(sweep_ranges.keys())
    if param_options:
        x_param = st.selectbox("Choose parameter for x-axis", options=param_options)
        # Dropdowns to fix other parameters using actual scenario values
        fixed_params = {k: st.selectbox(f"Fix {k.replace('_', ' ').title()}", options=sorted(df_results[k].unique())) 
                        for k in param_options if k != x_param}

        df_plot = df_results.copy()
        for k, v in fixed_params.items():
            df_plot = df_plot[df_plot[k] == v]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_plot[x_param], df_plot['Total cost (€)'], marker='o')
        ax.set_xlabel(x_param.replace('_', ' ').title())
        ax.set_ylabel('Total cost (€)')
        ax.set_title(f'Total cost vs {x_param.replace("_", " ").title()}')
        st.pyplot(fig)

    # Move table to the end
    st.subheader("Scenario results table")
    st.dataframe(df_results)



st.set_page_config(layout="wide", page_title="BESS day optimisation (LP)")
st.title("Behind-the-meter BESS — single-day LP prototype")
tab1, tab2 = st.tabs(["Single-day Optimisation", "Scenario Sweeper"])


params = get_user_inputs()
T = int(params['ptu_count'])
dt = params['dt']

if params['uploaded'] is None:
    pv_max_kw = st.sidebar.number_input("Default PV max power (kW)", value=80.0)
    base_load = st.sidebar.number_input("Default base load (kW)", value=20.0)
    daytime_peak = st.sidebar.number_input("Default daytime peak (kW)", value=60.0)
    evening_peak = st.sidebar.number_input("Default evening peak (kW)", value=80.0)
    avg_price = st.sidebar.number_input("Default DA average price (€/kWh)", value=0.20)
    load_profile, pv_profile, price_profile, feedin_profile = generate_mock_profiles(T, pv_max_kw, base_load, daytime_peak, evening_peak, avg_price, params['use_fixed_feedin'], params['fixed_feedin'])
else:
    df = pd.read_csv(params['uploaded'])
    load_col = st.sidebar.selectbox("Select load column (kW)", options=df.columns)
    pv_col = st.sidebar.selectbox("Select PV column (kW)", options=df.columns)
    load_profile = np.tile(df[load_col].values, int(np.ceil(T/len(df))))[:T]
    pv_profile = np.tile(df[pv_col].values, int(np.ceil(T/len(df))))[:T]
    price_profile = df['price'].values[:T] if 'price' in df.columns else np.full(T, 0.20)
    if params['use_fixed_feedin']:
        feedin_profile = np.full(T, params['fixed_feedin'])
    else:
        feedin_profile = df['feedin'].values[:T] if 'feedin' in df.columns else price_profile*0.5

with tab1:
    st.header("Run optimisation")
    if st.button("Solve day optimisation"):
        results = build_and_solve_lp(T, dt, load_profile, pv_profile, price_profile, feedin_profile, params)

        net_no_batt = load_profile - pv_profile
        import_no_batt = np.maximum(net_no_batt, 0)
        export_no_batt = np.maximum(-net_no_batt, 0)

        cost_no_batt = np.sum(import_no_batt * price_profile * dt) - np.sum(export_no_batt * feedin_profile * dt) + np.sum(import_no_batt * params['grid_energy_fee'] * dt) + params['demand_charge'] * np.max(import_no_batt)

        st.subheader("Summary results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total cost without battery (€)", f"{cost_no_batt:.2f}")
        col2.metric("Total cost with battery (€)", f"{results['objective']:.2f}")
        col3.metric("Savings (€)", f"{(cost_no_batt - results['objective']):.2f}")

        st.write(f"Peak import (kW): {results['Peak']:.2f}")

        fig = plot_results(T, dt, load_profile, pv_profile, results)
        st.pyplot(fig)

        results_df = pd.DataFrame({
            't': np.arange(T),
            'load_kW': load_profile,
            'pv_kW': pv_profile,
            'P_batt_kW': results['P'],  # only P
            'Import_kW': results['Import'],
            'Export_kW': results['Export'],
            'SoC_kWh': results['SoC'],
            'price_eur_kWh': price_profile,
            'feedin_eur_kWh': feedin_profile,
        })

        # After plotting the battery / load / PV results
        st.subheader("Day-ahead energy price profile")
        fig_price, ax_price = plt.subplots(figsize=(12, 3))
        hours = np.arange(T) * dt
        ax_price.plot(hours, price_profile, label='DA Price (€/kWh)', color='tab:orange')
        ax_price.set_xlabel("Hour of day")
        ax_price.set_ylabel("Price (€/kWh)")
        ax_price.set_title("Day-ahead energy price")
        ax_price.grid(True)
        ax_price.legend()
        st.pyplot(fig_price)

        st.subheader("Results table (first 50 rows)")
        st.dataframe(results_df.head(50))

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", csv, "bess_day_results.csv", "text/csv")
    else:
        st.info("Press 'Solve day optimisation' to run the LP.")

    st.markdown("---")
    st.markdown("**Notes:** This prototype uses a single continuous decision variable per PTU (battery power), auxiliaries for linear SoC treatment, and a small epsilon penalty to discourage simultaneous charge/discharge.")

with tab2:
    scenario_sweeper_tab(params, load_profile, pv_profile, price_profile, feedin_profile)
