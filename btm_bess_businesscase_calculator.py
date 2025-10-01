import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import copy
import json
from datetime import datetime
import pytz
import io

from plotting import (
    plot_daily_profiles,
    plot_yearly_overview,
    plot_cost_comparison,
    plot_battery_behavior,
    plot_idle_hours,
    plot_daily_savings
)


# -----------------------
# Helper functions
# -----------------------

def get_user_inputs():
    st.sidebar.header("Simulation settings")
    st.sidebar.write("Changing settings here will need a re-optimisation after.")
    st.sidebar.subheader("Battery specs")
    price_eur_per_mw = float(st.sidebar.number_input("Battery CapEx (k€/mw)", value=700, step=10, format="%d"))*1000
    project_horizon_yrs = float(st.sidebar.number_input("Project horizon (yrs)", value=10, step=1, format="%d"))

    st.sidebar.subheader("Operational settings")
    soc_init_pct = st.sidebar.number_input("Start SoC (%)", value=40, step=1, min_value=0, max_value=100, format="%d")
    st.sidebar.write("40% is roughly optimal for default settings for load, PV and duck curve prices.")
    import_cap = st.sidebar.number_input("Import cap (mw)", value=50., step=0.1, format="%2d")
    export_cap = st.sidebar.number_input("Export cap (mw) ", value=50., step=0.1, format="%2d")
    demand_charge = st.sidebar.number_input("Demand (peak) charge (k€/mw/yr) - currently not implemented", value=0, max_value=0, format="%d")*1000

    with st.sidebar.expander("Fixed inputs", expanded=False):
        st.write("These are the inputs we decided not to change for the experiments. They are implemented, so adjusting them will show the effect.")
        duration_hrs = st.number_input("Cycle duration (h)", value=2.0, step=0.5)
        eta = st.number_input("Charging/discharging efficiency ", value=0.95, min_value=0.5, max_value=1.0, step=0.01)
        soc_min_pct = st.number_input("Minimum SoC (%)", value=5, min_value=0, max_value=40, step=1, format="%d")
        soc_max_pct = st.number_input("Maximum SoC (%)", value=95, min_value=60, max_value=100, step=1, format="%d")
        end_of_day_soc_enforced = st.checkbox("Enforce end-of-day SoC = start SoC", value=True)
        dt = st.number_input("PTU duration in hours (only 0.25 is tested)", value=0.25, disabled=True)
    
    ptu_count = 24./float(dt)
    # Keep uploader out of sidebar per your request (we'll use main page uploader).
    return dict(
        ptu_count=ptu_count, dt=dt, 
        duration_hrs=duration_hrs, 
        eta=eta,
        soc_init_pct=soc_init_pct, soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
        end_of_day_soc_enforced=end_of_day_soc_enforced,project_horizon_yrs=project_horizon_yrs,
        import_cap=import_cap, export_cap=export_cap, demand_charge=demand_charge, price_eur_per_mw=price_eur_per_mw
    )


def generate_mock_profiles(T, base_load, daytime_peak, evening_peak, avg_price, feedin_discount):
    hours = (np.arange(T) / T) * 24.0
    mu, sigma = 13.0, 3.0
    pv_profile = np.maximum(0, np.exp(-0.5 * ((hours - mu) / sigma) ** 2))

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

    # --- Duck curve price profile ---
    morning_peak = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    midday_dip   = -np.exp(-0.5 * ((hours - 13) / 2.0) ** 2)
    evening_peak = np.exp(-0.5 * ((hours - 19) / 1.5) ** 2)

    duck_profile = morning_peak + midday_dip + evening_peak
    duck_profile -= duck_profile.mean()          # zero-mean to preserve avg_price

    price_profile = avg_price * (1 + duck_profile)

    feedin_profile = price_profile - feedin_discount

    return load_profile, pv_profile, price_profile, feedin_profile

def generate_full_year_profiles(T,  base_load, daytime_peak, evening_peak, avg_price, feedin_discount):
    # single-day
    load_year, pv_year, price_year, feedin_year = generate_mock_profiles(
        T, base_load, daytime_peak, evening_peak, avg_price, feedin_discount
    )

    # # tile for 365 days
    # load_year = np.tile(day_load, 365)
    # pv_year = np.tile(day_pv, 365)
    # price_year = np.tile(day_price, 365)
    # feedin_year = np.tile(day_feedin, 365)

    return load_year, pv_year, price_year, feedin_year


def build_and_solve_lp(T, dt, load_profile, pv_profile, price_profile, feedin_profile, grid_fee_per_mwh, params, penalty=1e-4):
    prob = LpProblem("BESS_day_opt_linear", LpMinimize)

    # Single battery power variable per PTU
    P = [LpVariable(f"P_{t}", lowBound=-params['max_power_mw'], upBound=params['max_power_mw']) for t in range(T)]

    # Net grid variable split into positive and negative parts for cost calculation
    G_import = [LpVariable(f"G_import_{t}", lowBound=0, upBound=params['import_cap']) for t in range(T)]
    G_export = [LpVariable(f"G_export_{t}", lowBound=0, upBound=params['export_cap']) for t in range(T)]
    
    # Signed net grid variable (for balance constraint)
    G = [LpVariable(f"G_{t}") for t in range(T)]
    for t in range(T):
        prob += G[t] == G_import[t] - G_export[t]

    # Peak import for demand charges
    Peak = LpVariable("PeakImport", lowBound=0)

    # SoC
    SoC = [LpVariable(f"SoC_{t}", lowBound=params['soc_min'], upBound=params['soc_max']) for t in range(T)]

    # SoC dynamics with correct efficiency depending on charge/discharge
    P_plus  = [LpVariable(f"P_plus_{t}", lowBound=0, upBound=params['max_power_mw']) for t in range(T)]
    P_minus = [LpVariable(f"P_minus_{t}", lowBound=0, upBound=params['max_power_mw']) for t in range(T)]

    for t in range(T):
        # link to main P variable
        prob += P[t] == P_plus[t] - P_minus[t]

        # SoC dynamics
        if t == 0:
            prob += SoC[0] == params['soc_init'] + params['eta'] * P_plus[0] * dt - (1 / params['eta']) * P_minus[0] * dt
        else:
            prob += SoC[t] == SoC[t-1] + params['eta'] * P_plus[t] * dt - (1 / params['eta']) * P_minus[t] * dt

    if params['end_of_day_soc_enforced']:
        prob += SoC[T-1] == params['soc_init']


    # Grid constraints
    for t in range(T):
        prob += G[t] == load_profile[t] - pv_profile[t] + P[t]
        prob += Peak >= G_import[t]

    # Objective with tiny linear penalty to discourage simultaneous import/export
    obj_terms = [
        lpSum(
            G_import[t] * price_profile[t] * dt        # import cost
            - G_export[t] * feedin_profile[t] * dt    # export revenue
            + G_import[t] * grid_fee_per_mwh[t] * dt # import fee
            + penalty * (G_import[t] + G_export[t])  # tiny penalty
        )
        for t in range(T)
    ]
    prob += lpSum(obj_terms)

    # Solve
    prob.solve()

    # Collect results
    results = {
        'P': [v.varValue for v in P],
        'SoC': [v.varValue for v in SoC],
        'G_import': [v.varValue for v in G_import],
        'G_export': [v.varValue for v in G_export],
        'G': [v.varValue for v in G],
        'Peak': Peak.varValue,
        'objective': value(prob.objective)
    }

    return results

def run_daily_optimisation(size_mw, load, pv, price, feedin, grid_fee_per_mwh, params):
    """
    Run the single-day optimisation for a given battery size.
    """
    # Override battery sizing
    params_copy = params.copy()
    params_copy['max_power_mw'] = size_mw
    params_copy['capacity_mwh'] = size_mw * params_copy['duration_hrs']

    # Recompute SoC values in MWh from percentages
    params_copy['soc_init'] = params_copy['soc_init_pct'] / 100.0 * params_copy['capacity_mwh']
    params_copy['soc_min']  = params_copy['soc_min_pct']  / 100.0 * params_copy['capacity_mwh']
    params_copy['soc_max']  = params_copy['soc_max_pct']  / 100.0 * params_copy['capacity_mwh']


    results = build_and_solve_lp(
        T=len(load),
        dt=params_copy['dt'],
        load_profile=load,
        pv_profile=pv,
        price_profile=price,
        feedin_profile=feedin,
        grid_fee_per_mwh=grid_fee_per_mwh,
        params=params_copy
    )


    return results

def run_yearly_optimisation(sizes, df, params):
    """
    Run optimisation for each battery size across all days.
    Returns:
        summary_df: per-size annualised summary (operating + capex write-off)
        daily_df: per-day detailed results (one row per day per size; costs are daily)
        detailed_results: dict(day -> results) for the best size
        best_size: size (MW) with highest annual savings
    Notes:
        - costs returned in summary_df are annualised (EUR / year).
        - daily_df contains daily numbers (EUR per day) as produced during the simulation.
    """
    T_day = int(params['ptu_count'])
    n_days = int(len(df) // T_day)
    if n_days <= 0:
        raise ValueError("Input dataframe too short for the configured PTU count")

    # If user runs only a single day for testing (n_days < 365) we scale daily totals to annual:
    annual_multiplier = 365.0 / float(n_days)

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
        # accumulate daily operating costs (these are sums over the days present in df)
        total_operating_cost_no_batt = 0.0
        total_operating_cost_with_batt = 0.0
        daily_results_temp = {}  # store per-day detailed results for this size

        for d in range(n_days):
            idx = slice(d * T_day, (d + 1) * T_day)
            load = df['load'].values[idx]
            pv = df['pv'].values[idx]
            price = df['use_price'].values[idx]
            feedin = df['inject_price'].values[idx]
            grid_fee_per_mwh = df['grid_fee'].values[idx]

            # --- no battery cost for this day (daily EUR)
            net_no_batt = load - pv
            import_no_batt = np.maximum(net_no_batt, 0.0)
            export_no_batt = np.maximum(-net_no_batt, 0.0)
            cost_no_batt = (
                np.sum(import_no_batt * price * params['dt']) -
                np.sum(export_no_batt * feedin * params['dt']) +
                np.sum(import_no_batt * df['grid_fee'].values[idx] * params['dt'])
                # demand charge handling is intentionally left out here; add appropriately if desired
            )

            # --- with battery (daily EUR) - result objective should be the day's operating cost
            results = run_daily_optimisation(size, load, pv, price, feedin, grid_fee_per_mwh, params)
            cost_with_batt = results['objective']

            # accumulate daily totals
            total_operating_cost_no_batt += cost_no_batt
            total_operating_cost_with_batt += cost_with_batt

            # store daily summary (note: costs here are per-day)
            daily_records.append({
                "day": d + 1,
                "size_mw": size,
                "cost_no_batt": cost_no_batt,
                "cost_with_batt": cost_with_batt,
                "savings": cost_no_batt - cost_with_batt,
                "peak_import_no_batt": np.max(import_no_batt),
                "peak_import_with_batt": np.max(results['G_import'])
            })

            # temporarily store detailed results for this size
            daily_results_temp[d+1] = results

            # update progress
            step += 1
            progress.progress(step / total_steps)
            status_text.text(f"Battery size: {size} MW — Day {d+1} of {n_days}")

        # --- annualise the operating totals (convert sum of days -> EUR/year)
        annual_cost_no_batt = total_operating_cost_no_batt * annual_multiplier
        annual_operating_cost_with_batt = total_operating_cost_with_batt * annual_multiplier

        # --- capex (total and annualised write-off)
        capex_total = params['price_eur_per_mw'] * size                 # EUR (price per MW * MW)
        writeoff_annual = capex_total / float(params['project_horizon_yrs'])  # EUR / year

        # --- final annual cost including capex write-off
        annual_cost_with_batt_incl_capex = annual_operating_cost_with_batt + writeoff_annual

        # --- savings (EUR / year)
        annual_savings = annual_cost_no_batt - annual_cost_with_batt_incl_capex

        # keep best size by annual savings
        if annual_savings > best_savings:
            best_savings = annual_savings
            best_size = size
            detailed_results = copy.deepcopy(daily_results_temp)

        # record summary for this size (all annualised except capex_total which is total upfront)
        summary_records.append({
            "size_mw": size,
            "annual_cost_no_batt": annual_cost_no_batt,
            "annual_operating_cost_with_batt": annual_operating_cost_with_batt,
            "annual_cost_with_batt_incl_capex": annual_cost_with_batt_incl_capex,
            "annual_savings": annual_savings,
            "capex_total_eur": capex_total,
            "writeoff_annual_eur": writeoff_annual
        })

    status_text.text(f"✅ Optimisation complete — Best size: {best_size} MW")
    progress.empty()

    summary_df = pd.DataFrame(summary_records)
    daily_df = pd.DataFrame(daily_records)
    return summary_df, daily_df, detailed_results, best_size

def summarise_generation_settings():
    """
    Return a one-line human-readable summary of the synthetic data generation settings
    that were used (reads from st.session_state['generation_settings']).
    If no generated settings are present (i.e. user uploaded a file), returns 'uploaded file'.
    """
    g = st.session_state.get('generation_settings')
    if not g:
        return "uploaded file"

    # format numbers nicely
    def fmt(x, nd=2):
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    parts = [
        f"Base load {fmt(g['base_load_mw'])} MW",
        f"Day peak {fmt(g['daytime_peak_mw'])} MW",
        f"Evening peak {fmt(g['evening_peak_mw'])} MW",
        f"Avg price {fmt(g['avg_price_eur_mwh'])} €/MWh",
        f"Feedin discount {fmt(g['feedin_discount_eur_mwh'])} €/MWh",
        f"Grid fee {fmt(g['grid_fee_eur_mwh'])} €/MWh",
        f"PTUs {g['ptu_count']} (dt={fmt(g['ptu_duration_hr'], nd=3)} h)"
    ]
    return "; ".join(parts)

# -----------------------
# Streamlit app
# -----------------------
summary_df = None
st.set_page_config(layout="wide", page_title="BTM BESS optimiser and business case")
st.title("BTM BESS optimiser and business case")
st.write("Otto Fabius - Sympower - September 2025")

params = get_user_inputs()
T = int(params['ptu_count'])
load_arr = np.zeros(T)
pv_arr = np.zeros(T)
price_arr = np.zeros(T)
feedin_arr = np.zeros(T)
grid_fee_arr = np.zeros(T)
dt = params['dt']

# --- Unified data input (upload or generate) ---
with st.expander("1. Input data", expanded=True):

    uploaded = st.file_uploader(
        "Upload yearly CSV (expected 365 × PTUs rows). Required columns: load, pv, use_price, inject_price, grid_fee",
        type=["csv"]
    )

    # --- Generate synthetic data automatically if no upload or scenario selected ---
    if uploaded is None:
        # --- Scenario selector with buttons ---
        st.write("### Choose a scenario for synthetic data generation")
        st.write("Load will be normalised to 1 MW average.")

        if 'scenario' not in st.session_state:
            st.session_state['scenario'] = "Profitable arbitrage"

        # Predefined scenarios
        scenario_params = {
            "Profitable arbitrage": {
                "base_load_mw": 1.0,
                "daytime_peak_mw": 1.0,
                "evening_peak_mw": 1.0,
                "pv_max_mw": 1.0,
                "avg_price_eur_mwh": 70,
                "feedin_discount_eur_mwh": 10,
                "grid_fee_eur_mwh": 5
            },
            "Unviable (Load matches PV)": {
                "base_load_mw": 2.0,
                "daytime_peak_mw": 4.0,
                "evening_peak_mw": 2.0,
                "pv_max_mw": 4.0,
                "avg_price_eur_mwh": 60,
                "feedin_discount_eur_mwh": 5,
                "grid_fee_eur_mwh": 8
            },
            "Viable for optimal BESS size": {
                "base_load_mw": 1.,
                "daytime_peak_mw": 1.,
                "evening_peak_mw": 1.,
                "pv_max_mw": 4.0,
                "avg_price_eur_mwh": 70,
                "feedin_discount_eur_mwh": 10,
                "grid_fee_eur_mwh": 30
            }
        }

        scenario_texts = {
            "Profitable arbitrage": "Profitable arbitrage settings have a year round duck curve and a decently high price volatility. Any case like this, where profits from arbitrage outweigh write-offs, will result in the largest battery size being the most profitable.",
            "Unviable (Load matches PV)": "Load matches pv settings show that BTM BESS is worse when the load profile is not very different the pv production. When the price variances below are set to 10 (a mild duck-curve), this scenario doesnt have a profitable business case",
            "Viable for optimal BESS size": "Set PV_multiplier to 1, and the price variances to 25 each, in the next section. In this scenario day-ahead spreads alone are not volatile enough to provide a good business case, but there is a significant grid fee per MWh.  In cases like these, there will be an optimal battery size that will maximise self consumption. With these settings, the optimum is around 1.1 MW for standard capex of 700k & battery depth of 2 hrs,",
            "Custom": "Adjust the parameters below to create your own scenario."
        }

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Unviable (Load matches PV)"):
                st.session_state.update(scenario_params["Unviable (Load matches PV)"])
                st.session_state['scenario'] = "Unviable (Load matches PV)"

        with col2:
            if st.button("Viable for optimal BESS size"):
                st.session_state.update(scenario_params["Viable for optimal BESS size"])
                st.session_state['scenario'] = "Viable for optimal BESS size"

        with col3:
            if st.button("Profitable arbitrage"):
                st.session_state.update(scenario_params["Profitable arbitrage"])
                st.session_state['scenario'] = "Profitable arbitrage"

        with col4:
            if st.button("Custom"):
                st.session_state['scenario'] = "Custom"
        st.write(scenario_texts[st.session_state['scenario']])
        # Determine defaults for expander
        defaults = scenario_params.get(st.session_state['scenario'], {})

        # Only expand parameters for Custom scenario
        expanded_state = st.session_state.get('scenario') == "Custom"

        with st.expander("Show / adjust scenario parameters", expanded=(st.session_state['scenario'] == "Custom")):
            # --- Input parameters (number_inputs) ---
            col1, col2, col3 = st.columns(3)
            with col1:
                base_load = st.number_input("Base CI& load (MW)", value=defaults.get("base_load_mw", 1.0))
                grid_energy_fee = st.number_input("Grid energy fee (€/MWh)", value=defaults.get("grid_fee_eur_mwh", 10))
            with col2: 
                daytime_peak = st.number_input("Daytime peak load (MW)", value=defaults.get("daytime_peak_mw", 1.0))
                evening_peak = st.number_input("Evening peak load (MW)", value=defaults.get("evening_peak_mw", 1.0))
            with col3:
                feedin_discount = st.number_input("Difference between cons and prod price (€/MWh)", value=defaults.get("feedin_discount_eur_mwh", 10))
                avg_price = st.number_input("DA average price (€/MWh)", value=defaults.get("avg_price_eur_mwh", 80))



        load_year, pv_year, price_year, feedin_year = generate_full_year_profiles(
            T, base_load, daytime_peak, evening_peak, avg_price, feedin_discount
        )
        df = pd.DataFrame({
            'load': load_year,
            'pv': pv_year,
            'use_price': price_year,
            'inject_price': feedin_year,
            'grid_fee': np.full(T, grid_energy_fee)
        })
        st.info(f"Synthetic data generated for scenario: {st.session_state['scenario']}")

        # Save the generation settings for later use
        st.session_state['generation_settings'] = {
            "base_load_mw": float(base_load),
            "grid_fee_eur_mwh": float(grid_energy_fee),
            "daytime_peak_mw": float(daytime_peak),
            "evening_peak_mw": float(evening_peak),
            "feedin_discount_eur_mwh": float(feedin_discount),
            "avg_price_eur_mwh": float(avg_price),
            "ptu_count": int(T),
            "ptu_duration_hr": float(dt)
        }
    else:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # normalise load
    df['load'] = df['load']/np.mean(np.abs(df['load']))
    
    # calculate mean and variance assuming normal distribution
    mean_cons_price = np.mean(df['use_price'])
    mean_prod_price = np.mean(df['inject_price'])
    
    std_cons_price = np.std(df['use_price'])
    std_prod_price = np.std(df['inject_price'])

    col0, col1, col2, col3 = st.columns(4)
    with col0:
        disable_load = st.checkbox("Run business case with a load of 0. This setting will not affect pv generation", value=False)
    with col1:
        # normalise pv and multiply by input factor
        pv_multiplier = st.number_input("PV Multiplier: MWp installed capacity per MW of average load", value=0.5, step=0.1)  # Multiply magnitude of PV generation by this factor
        st.write(f"Adjust pv profiles. The input will likely be changed to KWp installed per MW of average load. Currently it is just a msimple multiplier.")
        df['pv'] = df['pv']#/np.mean(np.abs(df['pv']))
        df['pv'] *= pv_multiplier
    with col2:
        # Add variance to dayahead prices
        new_cons_price_std = st.number_input("Consumption price variance (std dev in €/MWh)", value=std_cons_price, step=1.0, min_value=std_cons_price/10)
        st.write(f"The current dayahead load prices have mean {mean_cons_price:.2f} €/MWh and standard deviation {std_cons_price:.2f} €/MWh. Select a new value for the std below to adjust the variance around the yearly mean.")
    with col3:
        new_prod_price_std = st.number_input("Production price variance (std dev in €/MWh)", value=std_prod_price, step=1.0, min_value=std_prod_price/10)       
        st.write(f"The current dayahead production prices have mean {mean_prod_price:.2f} €/MWh and standard deviation {std_prod_price:.2f} €/MWh. Select a new value for the std below to adjust the variance around the yearly mean.")

    #renormalise the values by adjusting their current deviation from the mean to the new std dev
    df['use_price'] = mean_cons_price + (df['use_price'] - mean_cons_price) * (new_cons_price_std / std_cons_price)
    df['inject_price'] = mean_prod_price + (df['inject_price'] - mean_prod_price) * (new_prod_price_std / std_prod_price)

    # check if all production prices are lower than consumption prices
    if np.any(df['inject_price'] > df['use_price']):
        st.warning("⚠️ Unexpectedly, some PTUs have higher feed-in prices than consumption prices. The model implementation does not handle this correctly." \
        "Please check your input data and/or your variance settings (or continue at your own peril...)")
    if disable_load:
        df['load'] = 0.0
if uploaded:
    st.write("### Yearly overview of uploaded data")

    # Number of PTUs per day (e.g., 15-min intervals → 96 PTUs/day)
    PTU_per_day = 96

    col1, col2 = st.columns(2)
    fig_daily, fig_hourly = plot_yearly_overview(df, PTU_per_day)

    with col1:
        st.pyplot(fig_daily)

    with col2:
        st.pyplot(fig_hourly)





st.subheader(f"Inspect daily profiles")

# day selector
selected_day = st.number_input(
    "Select day to view",
    min_value=0,
    max_value=(len(df)//T)-1,
    value=0,
    step=1,
    key='selected_day'  # automatically stored in st.session_state['selected_day']
)
col1, col2 = st.columns(2)

fig_profile, fig_price = plot_daily_profiles(df, T, selected_day)

with col1:
    st.pyplot(fig_profile)

with col2:
    st.pyplot(fig_price)


# --- Battery size sweep (compact inputs on main page) ---
with st.expander("Optimisation", expanded=True):
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

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        min_size = st.number_input("Min size (mw)", value=1., step=.1, min_value=0.)
    with col_b:
        max_size = st.number_input("Max size (mw)", value=3., step=0.1, min_value=0.)
    with col_c:
        n_steps = st.number_input("Steps", value=1, step=1, min_value=1, max_value=100)


    battery_sizes = []
    if max_size >= min_size:
        battery_sizes = [round(x, 2) for x in np.linspace(min_size, max_size, n_steps)]
    else:
        st.warning("Please ensure Max >= Min and Step > 0")

    if battery_sizes:
        st.write(f"Battery sizes to test ({len(battery_sizes)}): {[f'{x:.1f}' for x in battery_sizes]}")
        if len(battery_sizes) > 10 and run_type == "Full year":
            st.warning(f"⚠️ Too many battery sizes selected ({len(battery_sizes)}). Please choose ≤ 10.")


    # Run optimisation display
    st.markdown("---")
    st.write("\n")
    st.markdown("""
        <div style="font-size:24px; font-weight:bold; text-align:left;">
        🚀 Run Optimisation 🚀
        </div>
        """, unsafe_allow_html=True)
    if st.button("(re-)Run Optimisation"):

        # --- Prepare DF for optimisation ---
        load_arr = df['load'].values[:T]
        pv_arr = df['pv'].values[:T]
        price_arr = df['use_price'].values[:T] if 'use_price' in df.columns else np.full(T, 0.20)
        feedin_arr = df['inject_price'].values[:T] if 'inject_price' in df.columns else np.full(T, feedin_discount)
        grid_fee_arr = df['grid_fee'].values[:T] if 'grid_fee' in df.columns else np.full(T, grid_energy_fee)

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


        # with st.expander("Results exploration", expanded=True):
if summary_df is not None:        
    with st.expander("Battery size sweep results (click to show)", expanded=False):
        if len(battery_sizes) > 1:
            # --- Cost plot ---
            fig = plot_cost_comparison(summary_df)
            st.pyplot(fig)

        else:
            st.dataframe(summary_df)
            st.markdown(
                f"**Battery size:** {best_size} MW → "
                f"savings excl. capex: {summary_df['annual_operating_cost_with_batt'].max()/1000:.2f} k€, "
                f"savings incl. capex: {summary_df['annual_cost_with_batt_incl_capex'].max()/1000:.2f} k€"
            )

    # --- Select day to display ---
    day_to_show = 1
    daily_display = daily_df[(daily_df['day'] == day_to_show) & 
                        (daily_df['size_mw'].round(2) == round(best_size, 2))].iloc[0]


# -----------------------
# Summary metrics
# -----------------------

if summary_df is not None and best_size is not None:
    st.subheader("Summary results")

    # Pick the best row (by size)
    best_row_mask = summary_df['size_mw'].round(3) == round(best_size, 3)
    best_row = summary_df[best_row_mask].iloc[0]

    # Extract values
    cost_no_batt = float(best_row['annual_cost_no_batt'])
    op_cost_with_batt = float(best_row['annual_operating_cost_with_batt'])
    total_cost_with_batt = float(best_row['annual_cost_with_batt_incl_capex'])
    capex_total = float(best_row['capex_total_eur'])
    writeoff_annual = float(best_row['writeoff_annual_eur'])

    # Savings
    savings_incl_capex = cost_no_batt - total_cost_with_batt
    savings_excl_capex = cost_no_batt - op_cost_with_batt

    # Payback (years)
    payback_years = capex_total / savings_excl_capex if savings_excl_capex > 0 else "N/A"

    # 10% of net savings (excl capex)
    rev_10pct = 0.10 * savings_excl_capex

    # Peak import
    peak_import_with_batt = daily_df[daily_df['size_mw'].round(2) == round(best_size,2)]['peak_import_with_batt'].max()
    peak_import_without_batt = daily_df[daily_df['size_mw'].round(2) == round(best_size,2)]['cost_no_batt'].max()  # optional

    # --- Calculate metrics from detailed results ---
    battery_power_positive = []
    idle_low50_per_day = []
    idle_high50_per_day = []

    for day_results in detailed_results.values():
        P = np.array(day_results['P'])
        SoC = np.array(day_results['SoC'])

        # Calculate cycles
        battery_power_positive.append(np.sum(P[P>0]))

        # Calculate idle hours
        idle_mask = np.abs(P) < 0.1 * best_size
        soc_pct = SoC / (best_size * params['duration_hrs']) * 100
        idle_low50_per_day.append(np.sum(idle_mask & (soc_pct < 50)) * params['dt'])
        idle_high50_per_day.append(np.sum(idle_mask & (soc_pct >= 50)) * params['dt'])

    cycles_per_day = np.sum(battery_power_positive) * params['dt'] / (best_size * params['duration_hrs']) / len(detailed_results) 

    # Display metrics in 4 columns
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Yearly cost without battery use (k€)", f"{cost_no_batt/1000:.2f}")
    col1.metric("Capital cost (k€)", f"{capex_total/1000:.2f}")

    col2.metric("Cost excl write-off (k€)", f"{op_cost_with_batt/1000:.2f}")
    col2.metric("Write-off cost (k€)", f"{writeoff_annual/1000:.2f}")

    col3.metric("Years until payback", f"{payback_years:.1f} yrs" if isinstance(payback_years, float) else "N/A")
    col3.metric("Peak import with battery (MW)", f"{peak_import_with_batt:.2f}")

    col4.metric("Gross savings excl capex (k€)", f"{savings_excl_capex/1000:.2f}")
    col4.metric("Average number of (full) cycles per day:", f"{cycles_per_day:.2f}")


    with st.expander(f"View battery behavior for {best_size} MW (click to show)", expanded=False):

        # --- Plot results for selected day ---
        results_day = detailed_results[day_to_show]
        fig = plot_battery_behavior(results_day, df_run, T, params, best_size, day_to_show)
        st.pyplot(fig)

        # --- Plot idle hours ---
        fig_idle = plot_idle_hours(detailed_results, best_size, params)
        st.pyplot(fig_idle)
    

    if run_type == "Full year":
        # --- Daily summary Plot ---
        fig_daily = plot_daily_savings(daily_df, best_size)
        st.pyplot(fig_daily)


    # --- CET timestamp for filename ---
    cet = pytz.timezone('CET')
    dt_str = datetime.now(cet).strftime("%Y%m%d_%H%M%S")

    # --- Convert parameters to JSON string ---
    params_json = json.dumps(params)

    results_df = pd.DataFrame({
        't': np.arange(T),
        'load_mw': df_run['load'].values[:T],
        'pv_mw': df_run['pv'].values[:T],
        'P_batt_mw': results_day['P'],
        'Import_mw': results_day['G_import'],
        'Export_mw': results_day['G_export'],
        'SoC_mwh': results_day['SoC'],
        'price_eur_mwh': df_run['use_price'].values[:T],
        'feedin_eur_mwh': df_run['inject_price'].values[:T],
    })
    
    # --- Create CSV with parameters as first row ---
    output = io.StringIO()
    output.write(f"# Simulation parameters: {params_json}\n")
    results_df.to_csv(output, index=False)
    csv_data = output.getvalue().encode('utf-8')

    # --- 1. Experiment summary CSV ---
    best_summary_row = summary_df[summary_df['size_mw'] == best_size].iloc[0]

    # --- Compute ROI and profit ---
    roi_years = best_summary_row['capex_total_eur'] / best_summary_row['annual_savings'] if best_summary_row['annual_savings'] > 0 else np.nan
    profit_per_year = best_summary_row['annual_savings']

    # --- Compute average idle hours ---
    idle_total_per_day = []
    idle_low_per_day = []
    idle_high_per_day = []
    # get the selected max power of the battery   
    # best_size = best_summary_row['size_mw'] 
    for day_results in detailed_results.values():
        P = np.array(day_results['P'])
        SoC = np.array(day_results['SoC'])
        idle_mask = np.abs(P) < 0.1 * best_summary_row['size_mw']  # idle if abs(power) < 10% max power
        soc_pct = SoC / (best_size * params['duration_hrs']) * 100  # % of capacity

        idle_total_per_day.append(np.sum(idle_mask) * params['dt'])
        idle_low_per_day.append(np.sum(idle_mask & (soc_pct < 30)) * params['dt'])
        idle_high_per_day.append(np.sum(idle_mask & (soc_pct > 70)) * params['dt'])

    avg_idle_total = np.mean(idle_total_per_day)
    avg_idle_low = np.mean(idle_low_per_day)
    avg_idle_high = np.mean(idle_high_per_day)

    # --- Create experiment summary DataFrame ---
    best_row_mask = summary_df['size_mw'].round(3) == round(best_size, 3)
    if best_row_mask.any():
        best_row = summary_df[best_row_mask].iloc[0]
    else:
        # fallback to first row (should not normally happen)
        best_row = summary_df.iloc[0]

    # Extract annual numbers (these are the same names used in your summary_records)
    cost_no_bess = float(best_row['annual_cost_no_batt'])
    op_cost_with_bess = float(best_row['annual_operating_cost_with_batt'])
    total_cost_with_bess = float(best_row['annual_cost_with_batt_incl_capex'])
    savings_incl_capex = float(best_row['annual_savings'])
    capex_total = float(best_row['capex_total_eur'])
    capex_annual = float(best_row['writeoff_annual_eur'])

    # Compute savings excluding capex and derived metrics
    savings_excl_capex = cost_no_bess - op_cost_with_bess
    if savings_excl_capex > 0:
        payback_years = capex_total / savings_excl_capex
    else:
        payback_years = "N/A"

    rev_10pct = 0.10 * savings_excl_capex

    # Peak import
    peak_import_with_batt = daily_df[daily_df['size_mw'].round(2) == round(best_size,2)]['peak_import_with_batt'].max()
    peak_import_without_batt = load_arr.max()

    # Build export DataFrame with new columns and order
    exp_summary_df = pd.DataFrame([{
        "Input filename": f"{uploaded.name if uploaded else 'generated'} ",
        "Data generation settings": summarise_generation_settings(), # TODO: need to adjust this function!
        "MWp per MW of average load": pv_multiplier,
        "Import cap (MW)": params["import_cap"],
        "Battery CapEx (k€/MW)": capex_total / best_size / 1000,
        "Demand (peak) charge (k€/MW/yr)": params["demand_charge"],
        "Best BESS size (MW)": best_size,
        "Total savings per MW average load excl capex (EUR/MW/yr)": savings_excl_capex,
        "Initial BESS investment (€)": capex_total,
        "CapEx annual write-off (EUR/yr)": capex_annual,
        "Total cost without BESS (EUR/yr)": cost_no_bess,
        "Total cost with BESS - excl capex (EUR/yr)": op_cost_with_bess,
        "Total cost with BESS - incl capex (EUR/yr)": total_cost_with_bess,
        "Total savings incl capex (EUR/yr)": savings_incl_capex,
        "Total savings excl capex (EUR/yr)": savings_excl_capex,
        "Total savings per MW of BESS incl capex (EUR/MW/year)": savings_incl_capex / best_size,
        "ROI (years)": payback_years,
        "Annual peak load with BESS (MW)": peak_import_with_batt,
        "Peak import costs (EUR/yr)": peak_import_with_batt * params["demand_charge"] * 1000,  # EUR/yr
        "Pct self-consumption (if PV)": 'not implemented yet',
        "Idle time >50% SOC (%)": np.mean(idle_high50_per_day), # take the mean
        "Idle time <50% SOC": np.mean(idle_low50_per_day),
        "Average full cycles per day": cycles_per_day
    }])


    # --- Download button for single-row experiment summary ---
    exp_csv = exp_summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download experiment summary (CSV)",
        exp_csv,
        f"experiment_summary_{dt_str}.csv",
        "text/csv"
    )


