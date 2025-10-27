import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions (from your script)
# -----------------------------
def execute_bid_strategy_for_period(hour, strategy_df):
    for _, row in strategy_df.iterrows():
        if row["Start (h)"] <= hour < row["End (h)"]:
            return row["Direction"]
    return "UP"  # fallback

def generate_mock_data(ptus_per_day, num_days, battery_p_mw):
    num_ptus_total = ptus_per_day * num_days
    load = np.ones(num_ptus_total) * battery_p_mw * 2
    cleared_price_up = np.random.uniform(100, 1000, size=num_ptus_total)
    cleared_price_down = np.random.uniform(100, 1000, size=num_ptus_total)
    return load, cleared_price_up, cleared_price_down

def cap_deliverable_up(bid_mw, load_prev):
    return min(bid_mw, load_prev)

def calculate_ptu_revenue(delivered_mwh, cleared_price):
    return delivered_mwh * cleared_price

def execute_activation(soc, bid_power, deliverable_mw, ptu_hours, battery, bid_direction, cleared_price, bid_price):
    activated = bid_price <= cleared_price
    requested_energy = deliverable_mw * ptu_hours
    delivered_energy = 0.0

    if activated:
        if bid_direction == "UP":
            energy_available = soc * battery["E_capacity"]
            delivered_energy = min(energy_available, requested_energy)
            soc -= delivered_energy / battery["E_capacity"]
        elif bid_direction == "DOWN":
            energy_available = (battery["soc_max"] - soc) * battery["E_capacity"]
            delivered_energy = min(energy_available, requested_energy)
            soc += delivered_energy / battery["E_capacity"]

        undelivered_energy = requested_energy - delivered_energy
    else:
        undelivered_energy = 0

    cycles_increment = delivered_energy / battery["E_capacity"]
    return activated, delivered_energy, soc, cycles_increment, undelivered_energy

def plot_first_day_prices(ptus_per_day, ptu_hours, cleared_price_up, cleared_price_down, strategy_df, bid_price_up, bid_price_down):
    time_hours = np.arange(ptus_per_day) * ptu_hours
    bid_directions = np.array([execute_bid_strategy_for_period((i*ptu_hours)%24, strategy_df) for i in range(ptus_per_day)])
    bids = np.array([bid_price_up if d=="UP" else bid_price_down for d in bid_directions])
    cleared_prices = np.array([cleared_price_up[i] if bid_directions[i]=="UP" else cleared_price_down[i] for i in range(ptus_per_day)])

    plt.figure(figsize=(12,5))
    plt.plot(time_hours, cleared_prices, label="Cleared Price", color="gray", alpha=0.6)
    plt.plot(time_hours, bids, color='green', linestyle='--', linewidth=2, label="Our Bid")
    plt.xlabel("Time [hours]")
    plt.ylabel("Price [€/MWh]")
    plt.title("First Day: Cleared Prices vs Our Bids")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

def plot_soc(soctrace):
    plt.figure(figsize=(12,4))
    plt.plot(soctrace, label="SoC", color='blue')
    plt.xlabel("PTU")
    plt.ylabel("State of Charge")
    plt.title("Battery SoC over time")
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔋 Battery Simulation for C&I Site (No Grid Injection)")

# Sidebar
st.sidebar.header("Simulation Settings")

# Main simulation controls
num_days = st.sidebar.number_input("Number of Days", 1, 30, 1)
ptu_hours = st.sidebar.selectbox("PTU Duration (hours)", [0.25, 0.5, 1.0], index=0)
soc_protection = st.sidebar.checkbox("Enable SoC Protection", True)

# Battery specs
st.sidebar.subheader("Battery Parameters")
battery_p_mw = st.sidebar.number_input("Power (MW)", 0.1, 10.0, 1.0, 0.1)
battery_depth_hours = st.sidebar.number_input("Depth (hours)", 0.5, 10.0, 2.0, 0.5)
battery_initial_soc = st.sidebar.slider("Initial SoC", 0.0, 1.0, 0.5)
battery_soc_min = st.sidebar.slider("Min SoC", 0.0, 1.0, 0.0)
battery_soc_max = st.sidebar.slider("Max SoC", 0.0, 1.0, 1.0)

# Market parameters
st.sidebar.subheader("Market Parameters")
imbalance_price = st.sidebar.number_input("Imbalance Price (€/MWh)", 0, 5000, 1000)

# Run button
run_sim = st.sidebar.button("Run Simulation")

# -----------------------------
# Editable Bid Strategy
# -----------------------------
with st.expander("⚙️ Advanced Bid Strategy Settings", expanded=False):
    st.write("Define the hours and directions for your bid strategy:")
    default_strategy = pd.DataFrame({
        "Start (h)": [0, 6, 12, 17],
        "End (h)": [6, 12, 17, 24],
        "Direction": ["DOWN", "UP", "DOWN", "UP"]
    })
    strategy_df = st.data_editor(default_strategy, num_rows="dynamic", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        bid_price_up = st.number_input("Bid Price UP (€/MWh)", 0, 2000, 500)
    with col2:
        bid_price_down = st.number_input("Bid Price DOWN (€/MWh)", 0, 2000, 500)

# -----------------------------
# Run Simulation Logic
# -----------------------------
if run_sim:
    ptus_per_day = int(24 / ptu_hours)
    battery = {
        "P_mw": battery_p_mw,
        "depth_hours": battery_depth_hours,
        "soc_min": battery_soc_min,
        "soc_max": battery_soc_max,
        "initial_soc": battery_initial_soc,
        "E_capacity": battery_p_mw * battery_depth_hours
    }

    load, cleared_price_up, cleared_price_down = generate_mock_data(ptus_per_day, num_days, battery_p_mw)

    soc = battery_initial_soc
    revenues, cycles, activations, undelivered = [], [], [], []
    soctrace = []

    for day in range(num_days):
        day_rev, day_cyc, day_act, day_undel = 0, 0, 0, 0
        for ptu in range(ptus_per_day):
            idx = day * ptus_per_day + ptu
            hour = (ptu * ptu_hours) % 24
            bid_dir = execute_bid_strategy_for_period(hour, strategy_df)
            bid_price = bid_price_up if bid_dir == "UP" else bid_price_down
            cleared_price = cleared_price_up[idx] if bid_dir == "UP" else cleared_price_down[idx]
            load_prev = load[idx - 1] if idx > 0 else load[idx]
            deliverable_mw = cap_deliverable_up(battery_p_mw, load_prev) if bid_dir == "UP" else battery_p_mw

            activated, delivered_mwh, soc, cyc_inc, undel_mwh = execute_activation(
                soc, battery_p_mw, deliverable_mw, ptu_hours, battery, bid_dir, cleared_price, bid_price
            )
            if activated:
                day_act += 1
            day_rev += calculate_ptu_revenue(delivered_mwh, cleared_price)
            day_cyc += cyc_inc
            if undel_mwh > 0:
                day_undel += 1
            soctrace.append(soc)

        revenues.append(day_rev)
        cycles.append(day_cyc)
        activations.append(day_act)
        undelivered.append(day_undel)

    # -----------------------------
    # Display Results
    # -----------------------------
    tab1, tab2 = st.tabs(["📊 Overview", "📈 Plots"])

    with tab1:
        st.subheader("Simulation Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue (€)", f"{sum(revenues):,.0f}")
        col2.metric("Total Activations", f"{sum(activations)}")
        col3.metric("Total Cycles", f"{sum(cycles):.2f}")
        col4.metric("Undelivered PTUs", f"{sum(undelivered)}")

    with tab2:
        st.subheader("First Day: Cleared Prices vs Bids")
        plot_first_day_prices(ptus_per_day, ptu_hours, cleared_price_up, cleared_price_down, strategy_df, bid_price_up, bid_price_down)
        st.subheader("Battery SoC Over Time")
        plot_soc(soctrace)
