import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# -----------------------------
# Helper functions (from your script)
# -----------------------------
def execute_bid_strategy_for_period(hour, strategy_df):
    """
    Returns the bid direction, price, and rest SoC for a given hour based on strategy.
    Returns: tuple (direction, bid_price, rest_soc)
    """
    for _, row in strategy_df.iterrows():
        if row["Start (h)"] <= hour < row["End (h)"]:
            return row["Direction"], row["Price (€/MWh)"], row["Rest SoC"]
    return "UP", 500, 0.5  # fallback defaults

def generate_mock_data(ptus_per_day, num_days, battery_p_mw, ptu_hours):
    num_ptus_total = ptus_per_day * num_days

    # Generate timestamps starting from Jan 1, 2024
    start_date = pd.Timestamp("2024-01-01 00:00:00")
    ptu_duration = pd.Timedelta(hours=ptu_hours)
    timestamps = pd.date_range(start=start_date, periods=num_ptus_total, freq=ptu_duration)

    load = np.ones(num_ptus_total) * battery_p_mw * 2
    cleared_price_up = np.random.uniform(100, 1000, size=num_ptus_total)
    cleared_price_down = np.random.uniform(100, 1000, size=num_ptus_total)
    return timestamps, load, cleared_price_up, cleared_price_down

def load_and_validate_csv(uploaded_file):
    """
    Load and validate CSV file with required columns.
    Returns: DataFrame with validated data or None if validation fails
    """
    try:
        # Reset file pointer to beginning in case file was read before
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        # Check for required columns
        required_cols = ['timestamp', 'load_kw', 'cleared_price_up', 'cleared_price_down']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: timestamp, load_kw, cleared_price_up, cleared_price_down")
            return None

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Validate numeric columns
        numeric_cols = ['load_kw', 'cleared_price_up', 'cleared_price_down']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    st.error(f"Column '{col}' must contain numeric values")
                    return None

        # Check for negative load (load should always be positive)
        if (df['load_kw'] < 0).any():
            st.warning("Warning: Negative load values detected. Please verify your data.")

        # Convert load from kW to MW for internal calculations
        df['load_mw'] = df['load_kw'] / 1000

        return df

    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def cap_deliverable_up(bid_mw, load_prev):
    """
    Legacy function - caps bid size by load.
    Note: This is no longer used in main simulation.
    Bid sizing is now handled explicitly in forecast limiting logic.
    """
    return min(bid_mw, load_prev)

def forecast_load(load_array, ptu_idx, ptus_per_day, lookback_days):
    """
    Forecast load for a PTU based on average of same PTU in previous N days.

    Args:
        load_array: Full array of load values (MW)
        ptu_idx: Index of current PTU
        ptus_per_day: Number of PTUs per day
        lookback_days: Number of past days to average

    Returns:
        Forecasted load (MW), or actual load if insufficient history
    """
    # Calculate which PTU of the day this is
    ptu_of_day = ptu_idx % ptus_per_day

    # Collect historical values for this same PTU across previous days
    historical_values = []
    for day_offset in range(1, lookback_days + 1):
        hist_idx = ptu_idx - (day_offset * ptus_per_day)
        if hist_idx >= 0:
            historical_values.append(load_array[hist_idx])

    # If we have historical data, return the average; otherwise return current load
    if len(historical_values) > 0:
        return np.mean(historical_values)
    else:
        # Not enough history, return actual load (no limiting)
        return load_array[ptu_idx]

def calculate_ptu_revenue(delivered_mwh, cleared_price, direction):
    """
    Calculate revenue for a PTU activation.

    UP regulation (discharge): revenue = delivered × price
    DOWN regulation (charge): revenue = -(delivered × price)

    Examples:
    - UP 3 MWh @ -30 €/MWh = 3 × (-30) = -90€ (debit)
    - UP 3 MWh @ +30 €/MWh = 3 × 30 = +90€ (profit)
    - DOWN 3 MWh @ -30 €/MWh = -(3 × (-30)) = +90€ (profit)
    - DOWN 3 MWh @ +30 €/MWh = -(3 × 30) = -90€ (debit)
    """
    base_revenue = delivered_mwh * cleared_price

    if direction == "UP":
        return base_revenue
    else:  # DOWN
        return -base_revenue

def execute_activation(soc, bid_power, deliverable_mw, ptu_hours, battery, bid_direction, cleared_price, bid_price, baseline_load_mw, actual_load_mw):
    """
    Execute battery activation based on TSO baseline comparison.

    UP regulation: Decrease consumption (battery discharges)
    DOWN regulation: Increase consumption (battery charges)

    baseline_load_mw: C&I load from last non-activated PTU (MW)
    actual_load_mw: Current PTU's actual C&I load (MW)
    """
    activated = bid_price <= cleared_price

    # Calculate the activation target energy
    activation_target_energy = deliverable_mw * ptu_hours

    if activated:
        # Convert loads to energy
        baseline_energy = baseline_load_mw * ptu_hours
        actual_energy = actual_load_mw * ptu_hours

        if bid_direction == "UP":
            # UP: Decrease consumption (battery discharges)
            # Expected net = Baseline - Target
            # Battery must deliver = Actual - Expected net
            # Formula: Battery = Actual - (Baseline - Target) = Target - (Baseline - Actual)
            requested_energy = activation_target_energy - (baseline_energy - actual_energy)

            # Constraint 1: Non-negative
            requested_energy = max(0, requested_energy)

            # Store the original market obligation (before physical constraints)
            market_obligation = requested_energy

            # Constraint 2: No grid injection - cap by actual load (not baseline)
            # Battery discharges during current PTU, so must not exceed current site load
            # This limits what we CAN deliver, not what we MUST deliver
            max_discharge_energy = actual_energy
            deliverable_energy = min(requested_energy, max_discharge_energy)

            # Constraint 3: SoC limit - check what battery can deliver
            energy_available = soc * battery["E_capacity"]
            delivered_energy = min(energy_available, deliverable_energy)

            # Update SoC
            soc -= delivered_energy / battery["E_capacity"]

            # Calculate underdelivery against market obligation, not physical constraint
            requested_energy = market_obligation
            undelivered_energy = market_obligation - delivered_energy

            # Determine underdelivery reason (priority: load limit, then SoC limit)
            if undelivered_energy > 0.001:  # Small tolerance for floating point
                if deliverable_energy < market_obligation - 0.001:
                    underdelivery_reason = "load_limit"
                elif delivered_energy < deliverable_energy - 0.001:
                    underdelivery_reason = "soc_limit_low"
                else:
                    underdelivery_reason = "other"
            else:
                underdelivery_reason = None

        elif bid_direction == "DOWN":
            # DOWN: Increase consumption (battery charges)
            # Expected net = Baseline + Target
            # Battery must deliver = Expected net - Actual
            # Formula: Battery = (Baseline + Target) - Actual = Target + (Baseline - Actual)
            requested_energy = activation_target_energy + (baseline_energy - actual_energy)

            # Constraint 1: Non-negative
            requested_energy = max(0, requested_energy)

            # Constraint 2: SoC limit - check what battery can store
            energy_available = (battery["soc_max"] - soc) * battery["E_capacity"]
            delivered_energy = min(energy_available, requested_energy)

            # Update SoC
            soc += delivered_energy / battery["E_capacity"]

            # Calculate underdelivery and determine reason for DOWN direction
            undelivered_energy = requested_energy - delivered_energy
            if undelivered_energy > 0.001:  # Small tolerance for floating point
                underdelivery_reason = "soc_limit_high"
            else:
                underdelivery_reason = None
    else:
        delivered_energy = 0.0
        undelivered_energy = 0
        underdelivery_reason = None

    cycles_increment = delivered_energy / battery["E_capacity"]
    return activated, delivered_energy, soc, cycles_increment, undelivered_energy, underdelivery_reason

def recover_rest_soc(soc, rest_soc, battery, ptu_hours, load_mw):
    """
    Move the battery SoC toward the target rest_soc when not activated.
    Respects battery power limits, SoC bounds (0 to 1), and no-grid-injection constraint.

    Returns: (energy_moved, new_soc, cycles_increment)
    """
    if abs(soc - rest_soc) < 0.001:  # Already at rest SoC (within tolerance)
        return 0.0, soc, 0.0

    # Calculate energy needed to reach rest_soc
    energy_needed = (rest_soc - soc) * battery["E_capacity"]

    # Calculate max energy we can move in this PTU based on power limit
    max_energy_per_ptu = battery["P_mw"] * ptu_hours

    # Limit energy movement to what's possible in this PTU
    if energy_needed > 0:  # Need to charge
        # Check how much we can charge (respect soc_max = 1)
        max_chargeable = (1.0 - soc) * battery["E_capacity"]
        energy_to_move = min(energy_needed, max_energy_per_ptu, max_chargeable)
    else:  # Need to discharge
        # Check how much we can discharge (respect soc_min = 0)
        max_dischargeable = soc * battery["E_capacity"]

        # CRITICAL: Cap discharge by site load to prevent grid injection
        max_discharge_by_load = load_mw * ptu_hours

        energy_to_move = max(energy_needed, -max_energy_per_ptu, -max_dischargeable, -max_discharge_by_load)

    # Update SoC
    new_soc = soc + (energy_to_move / battery["E_capacity"])

    # Calculate cycles (absolute value since we count both charge and discharge)
    cycles_increment = abs(energy_to_move) / battery["E_capacity"]

    return abs(energy_to_move), new_soc, cycles_increment

def plot_first_day_prices(ptus_per_day, ptu_hours, cleared_price_up, cleared_price_down, strategy_df):
    time_hours = np.arange(ptus_per_day) * ptu_hours

    # Get direction and Price for each PTU
    strategy_results = [execute_bid_strategy_for_period((i*ptu_hours)%24, strategy_df) for i in range(ptus_per_day)]
    bid_directions = np.array([direction for direction, _, _ in strategy_results])
    bid_prices = np.array([price for _, price, _ in strategy_results])

    # Get the relevant cleared prices based on direction
    cleared_prices = np.array([cleared_price_up[i] if bid_directions[i]=="UP" else cleared_price_down[i] for i in range(ptus_per_day)])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_hours, cleared_prices, label="Cleared Price", color="gray", alpha=0.6)
    ax.plot(time_hours, bid_prices, color='green', linestyle='--', linewidth=2, label="Our Bid")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Price [€/MWh]")
    ax.set_title("First Day: Cleared Prices vs Our Bids")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_soc(soctrace):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(soctrace, label="SoC", color='blue')
    ax.set_xlabel("PTU")
    ax.set_ylabel("State of Charge")
    ax.set_title("Battery SoC over time")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

def plot_revenue_analysis(ptu_df, aggregation_level, selected_period=None, time_range_filter=None):
    """
    Plot revenue analysis with flexible time aggregation.

    Args:
        ptu_df: DataFrame with PTU-level results
        aggregation_level: 'ptu', 'daily', or 'monthly'
        selected_period: day number (for ptu), year-month period (for daily), or None (for monthly)
        time_range_filter: tuple (start_hour, end_hour) for PTU-level filtering, or None
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Ensure timestamp is datetime
    ptu_df['timestamp'] = pd.to_datetime(ptu_df['timestamp'])

    # Add calendar month columns
    ptu_df['year_month'] = ptu_df['timestamp'].dt.to_period('M')
    ptu_df['month_label'] = ptu_df['timestamp'].dt.strftime('%b %Y')
    ptu_df['date'] = ptu_df['timestamp'].dt.date

    if aggregation_level == 'ptu':
        # Filter for selected day
        day_data = ptu_df[ptu_df['day'] == selected_period].copy()

        # Apply time range filter if provided
        if time_range_filter is not None:
            start_hour, end_hour = time_range_filter
            day_data = day_data[(day_data['hour'] >= start_hour) & (day_data['hour'] < end_hour)]

        x_values = day_data['hour'].values
        x_label = 'Hour of Day'

        # Get date for title
        date_str = pd.to_datetime(day_data['timestamp'].iloc[0]).strftime('%Y-%m-%d')
        if time_range_filter is not None:
            title = f'Revenue Analysis - {date_str} ({start_hour:02d}:00-{end_hour:02d}:00)'
        else:
            title = f'Revenue Analysis - {date_str} (Day {selected_period})'

        gross_revenue = day_data['ptu_revenue'].values
        penalties = day_data['ptu_penalty'].values
        net_revenue = gross_revenue - penalties

    elif aggregation_level == 'daily':
        # Filter for selected month and aggregate by day
        month_data = ptu_df[ptu_df['year_month'] == selected_period].copy()

        daily_agg = month_data.groupby('date').agg({
            'ptu_revenue': 'sum',
            'ptu_penalty': 'sum'
        }).reset_index()

        # Get day of month for x-axis
        daily_agg['day_of_month'] = pd.to_datetime(daily_agg['date']).dt.day

        x_values = daily_agg['day_of_month'].values
        x_label = 'Day of Month'

        # Get month label for title
        month_label = month_data['month_label'].iloc[0]
        title = f'Revenue Analysis - {month_label} (Daily Aggregates)'

        gross_revenue = daily_agg['ptu_revenue'].values
        penalties = daily_agg['ptu_penalty'].values
        net_revenue = gross_revenue - penalties

    else:  # monthly
        # Aggregate by calendar month
        monthly_agg = ptu_df.groupby(['year_month', 'month_label']).agg({
            'ptu_revenue': 'sum',
            'ptu_penalty': 'sum'
        }).reset_index()

        x_values = monthly_agg['month_label'].values
        x_label = 'Month'
        title = 'Revenue Analysis - Monthly Aggregates'

        gross_revenue = monthly_agg['ptu_revenue'].values
        penalties = monthly_agg['ptu_penalty'].values
        net_revenue = gross_revenue - penalties

    # Create bar width
    bar_width = 0.35
    x_pos = np.arange(len(x_values))

    # Plot bars
    ax.bar(x_pos - bar_width/2, gross_revenue, bar_width,
           label='Gross Revenue', color='green', alpha=0.7)
    ax.bar(x_pos + bar_width/2, penalties, bar_width,
           label='Penalties', color='red', alpha=0.7)

    # Plot net revenue line
    ax.plot(x_pos, net_revenue, color='blue', marker='o',
            linewidth=2, markersize=6, label='Net Revenue', zorder=5)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel('Revenue (€)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_bid_strategy(strategy_df):
    """
    Plot the Prices throughout the day with different colors for UP and DOWN.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Iterate through each row in the strategy
    for _, row in strategy_df.iterrows():
        start_h = row["Start (h)"]
        end_h = row["End (h)"]
        direction = row["Direction"]
        bid_price = row["Price (€/MWh)"]

        # Choose color based on direction
        color = 'green' if direction == "UP" else 'red'

        # Plot horizontal line for this period
        ax.plot([start_h, end_h], [bid_price, bid_price],
                color=color, linewidth=3, label=direction if start_h == 0 or direction != strategy_df.iloc[_-1]["Direction"] else "")

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title("Bidding Strategy Throughout the Day")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1000)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 25, 2))

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔋 Battery Simulation for C&I Site (No Grid Injection)")

# Sidebar
st.sidebar.header("Simulation Settings")

# Main simulation controls
num_days = 7
ptu_hours = 0.25 #st.sidebar.selectbox("PTU Duration (hours)", [0.25, 0.5, 1.0], index=0)
soc_protection = st.sidebar.checkbox("Enable SoC Protection", True)

# Battery specs
st.sidebar.subheader("Battery Parameters")
battery_p_mw = st.sidebar.number_input("Power (MW)", 0.1, 10.0, 1.0, 0.1)
battery_depth_hours = st.sidebar.number_input("Depth (hours)", 0.5, 10.0, 2.0, 0.5)
battery_initial_soc = 0.5 # st.sidebar.slider("Initial SoC", 0.0, 1.0, 0.5)
battery_soc_min = 0. #st.sidebar.slider("Min SoC", 0.0, 1.0, 0.0)
battery_soc_max = 1. #st.sidebar.slider("Max SoC", 0.0, 1.0, 1.0)
max_cycles_per_day = st.sidebar.number_input("Max Cycles per Day", 0.0, 10.0, 10.0, 0.1, help="Maximum battery cycles allowed per day (1 cycle = full discharge + full charge equivalent)")

# Market parameters
st.sidebar.subheader("Underdelivery Penalty Settings")
imbalance_price = st.sidebar.number_input("Imbalance Price (€/MWh)", 0, 5000, 200)
small_penalty = st.sidebar.number_input("Small Penalty (€)", 0, 1000, 25, help="Fixed penalty per underdelivery event")

# -----------------------------
# Data Input (Main App)
# -----------------------------
st.write("User guide available at https://github.com/ofabius-spw/btm_bess/blob/main/USER_GUIDE_btm_mfrr.md")
uploaded_file = st.file_uploader(
    "Upload CSV (optional)",
    type=['csv'],
    help="Upload CSV with columns: timestamp, load_kw, cleared_price_up, cleared_price_down. Leave empty to use mock data."
)

# -----------------------------
# Bidding Strategy and Data Visualization
# -----------------------------
with st.expander("⚙️ Define the hours, directions, Prices, and target Rest SoC ", expanded=False):
    default_strategy = pd.DataFrame({
        "Start (h)": [0, 6, 12, 17],
        "End (h)": [6, 12, 17, 24],
        "Direction": ["DOWN", "UP", "DOWN", "UP"],
        "Price (€/MWh)": [400, 500, 450, 550],
        "Rest SoC": [0.5, 0.5, 0.5, 0.5],
        "% of max power": [1.0, 1.0, 1.0, 1.0]
    })
    strategy_df = st.data_editor(default_strategy, num_rows="dynamic", use_container_width=True)

    # Load forecast settings
    limit_bids_to_forecast = st.checkbox(
        "Limit UP bids to load forecast",
        False,
        help="If enabled, UP bids will be limited to forecasted load to avoid grid injection risk"
    )
    if limit_bids_to_forecast:
        forecast_lookback_days = st.number_input(
            "Forecast lookback period (days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
            help="Number of past days to average for load forecast"
        )

# Integrated visualization in expander
with st.expander("📊 Strategy and Market Data Visualization", expanded=False):
    # Load data if uploaded
    if uploaded_file is not None:
        preview_df = load_and_validate_csv(uploaded_file)
        if preview_df is not None:
            # Calculate number of days
            ptus_per_day_preview = int(24 / ptu_hours)
            num_days_preview = int(np.ceil(len(preview_df) / ptus_per_day_preview))
        else:
            preview_df = None
            num_days_preview = 1
    else:
        preview_df = None
        num_days_preview = 1

    # Dropdown for display options (always show if file uploaded)
    if preview_df is not None:
        display_option = st.selectbox(
            "Market Data to Display",
            options=["Average Daily Prices", "Select a Specific Day", "None"],
            index=0
        )

        if display_option == "Average Daily Prices":
            show_average = True
            selected_day = None
        elif display_option == "Select a Specific Day":
            show_average = False
            selected_day = st.number_input(
                "Select Day",
                min_value=1,
                max_value=num_days_preview,
                value=1,
                step=1
            )
        else:  # "None"
            show_average = False
            selected_day = None
    else:
        show_average = False
        selected_day = None

    # Create integrated plot
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot bid strategy prices on left y-axis
    for _, row in strategy_df.iterrows():
        start_h = row["Start (h)"]
        end_h = row["End (h)"]
        direction = row["Direction"]
        bid_price = row["Price (€/MWh)"]

        color = 'green' if direction == "UP" else 'red'
        linestyle = '-'
        alpha = 0.7

        ax1.plot([start_h, end_h], [bid_price, bid_price],
                color=color, linewidth=3, linestyle=linestyle, alpha=alpha,
                label=f'Bid {direction}' if start_h == 0 or direction != strategy_df.iloc[_-1]["Direction"] else "")

    # If data uploaded, overlay cleared prices
    if preview_df is not None:
        if show_average:
            # Calculate average prices per PTU across all days with std
            preview_df['ptu_of_day'] = preview_df.index % ptus_per_day_preview
            avg_prices = preview_df.groupby('ptu_of_day').agg({
                'cleared_price_up': 'mean',
                'cleared_price_down': 'mean',
                'load_mw': ['mean', 'std']
            }).reset_index()

            # Flatten column names
            avg_prices.columns = ['ptu_of_day', 'cleared_price_up', 'cleared_price_down', 'load_mw_mean', 'load_mw_std']

            # Create time axis
            avg_prices['hour_of_day'] = avg_prices['ptu_of_day'] * ptu_hours

            # Plot average cleared prices
            ax1.plot(avg_prices['hour_of_day'], avg_prices['cleared_price_up'],
                    label='Avg Cleared Price UP', color='darkgreen', linewidth=2, linestyle=':', alpha=0.8)
            ax1.plot(avg_prices['hour_of_day'], avg_prices['cleared_price_down'],
                    label='Avg Cleared Price DOWN', color='darkred', linewidth=2, linestyle=':', alpha=0.8)

            day_data = avg_prices  # For load plotting below

        elif selected_day is not None:
            # Extract data for selected day
            start_idx = (selected_day - 1) * ptus_per_day_preview
            end_idx = min(start_idx + ptus_per_day_preview, len(preview_df))
            day_data = preview_df.iloc[start_idx:end_idx].copy()

            # Create time axis (hours of day)
            day_data['hour_of_day'] = [(i * ptu_hours) for i in range(len(day_data))]

            # Plot cleared prices for specific day
            ax1.plot(day_data['hour_of_day'], day_data['cleared_price_up'],
                    label='Cleared Price UP', color='darkgreen', linewidth=2, linestyle=':', alpha=0.8)
            ax1.plot(day_data['hour_of_day'], day_data['cleared_price_down'],
                    label='Cleared Price DOWN', color='darkred', linewidth=2, linestyle=':', alpha=0.8)
        else:
            day_data = None
    else:
        day_data = None

    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Price (€/MWh)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 1000)
    ax1.set_xticks(range(0, 25, 2))
    ax1.legend(loc='upper left')

    # Add load on right y-axis if data available
    if day_data is not None:
        ax2 = ax1.twinx()

        if show_average:
            # Plot average load with confidence intervals
            load_mean_kw = day_data['load_mw_mean'] * 1000
            load_std_kw = day_data['load_mw_std'] * 1000

            # Plot mean load (solid line)
            ax2.plot(day_data['hour_of_day'], load_mean_kw,
                    label='Avg Load',
                    color='blue', linewidth=2, linestyle='-')

            # Plot upper confidence bound (mean + 2*std)
            ax2.plot(day_data['hour_of_day'], load_mean_kw + 2*load_std_kw,
                    label='95% Confidence Interval',
                    color='blue', linewidth=1.5, linestyle=':', alpha=0.7)

            # Plot lower confidence bound (mean - 2*std)
            ax2.plot(day_data['hour_of_day'], load_mean_kw - 2*load_std_kw,
                    color='blue', linewidth=1.5, linestyle=':', alpha=0.7)
        else:
            # Plot single day load (no confidence intervals)
            ax2.plot(day_data['hour_of_day'], day_data['load_mw'] * 1000,
                    label='Load',
                    color='blue', linewidth=2, linestyle='--')

        ax2.set_ylabel('Load (kW)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.legend(loc='upper right')

        if show_average:
            title = 'Bidding Strategy & Average Market Data (All Days)'
        else:
            title = f'Bidding Strategy & Market Data (Day {selected_day})'
    else:
        title = 'Bidding Strategy'

    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

# Run Simulation button
run_sim = st.button("🚀 Run Simulation", type="primary")

# Initialize session state for results persistence
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

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

    # Load data from CSV or generate mock data
    if uploaded_file is not None:
        data_df = load_and_validate_csv(uploaded_file)
        if data_df is None:
            st.stop()  # Stop execution if CSV validation fails

        # Extract arrays and timestamps from DataFrame
        timestamps = data_df['timestamp'].values
        load = data_df['load_mw'].values
        cleared_price_up = data_df['cleared_price_up'].values
        cleared_price_down = data_df['cleared_price_down'].values

        # Update num_days based on actual data
        num_ptus_total = len(data_df)
        num_days = int(np.ceil(num_ptus_total / ptus_per_day))

    else:
        # Use mock data
        timestamps, load, cleared_price_up, cleared_price_down = generate_mock_data(ptus_per_day, num_days, battery_p_mw, ptu_hours)
        st.info(f"Using mock data: {num_days} days")

    soc = battery_initial_soc
    revenues, cycles, activations, undelivered = [], [], [], []
    soctrace = []
    skipped_due_to_cycles = []  # Track PTUs skipped due to cycle limit
    undelivered_energy_mwh = []  # Track actual undelivered energy in MWh
    penalties = []  # Track penalty costs

    # Track underdelivery by reason
    undel_cycle_limit = []  # Underdelivered MWh due to cycle limit
    undel_load_limit = []   # Underdelivered MWh due to load constraint (grid injection prevention)
    undel_soc_low = []      # Underdelivered MWh due to low SoC (UP direction)
    undel_soc_high = []     # Underdelivered MWh due to high SoC (DOWN direction)

    # PTU-level results storage (will build list of dicts, then convert to DataFrame)
    ptu_results_list = []

    # Track baseline: index of last non-activated PTU
    last_non_activated_idx = 0

    # Create progress bar and status
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Update frequency: update every N days or at minimum every 10% progress
    update_frequency = max(1, num_days // 20)  # Update at least 20 times

    for day in range(num_days):
        # Update progress bar periodically
        if day % update_frequency == 0 or day == num_days - 1:
            progress = (day + 1) / num_days
            progress_bar.progress(progress)
            progress_text.text(f"Simulating day {day + 1} of {num_days}... ({progress*100:.0f}%)")
            time.sleep(0.01)  # Small delay to force UI update
        day_rev, day_cyc, day_act, day_undel, day_skipped = 0, 0, 0, 0, 0
        day_undel_energy, day_penalty = 0.0, 0.0
        day_undel_cycle, day_undel_load, day_undel_soc_low, day_undel_soc_high = 0.0, 0.0, 0.0, 0.0
        for ptu in range(ptus_per_day):
            idx = day * ptus_per_day + ptu
            hour = (ptu * ptu_hours) % 24

            # Get bid direction, price, and rest SoC from strategy
            bid_dir, bid_price, rest_soc = execute_bid_strategy_for_period(hour, strategy_df)

            cleared_price = cleared_price_up[idx] if bid_dir == "UP" else cleared_price_down[idx]

            # Baseline load: from last PTU that had NO activation
            baseline_load = load[last_non_activated_idx]
            actual_load = load[idx]

            # Calculate load forecast if enabled
            if limit_bids_to_forecast:
                forecasted_load = forecast_load(load, idx, ptus_per_day, forecast_lookback_days)
            else:
                forecasted_load = None

            if limit_bids_to_forecast and bid_dir == "UP":
                # Limit deliverable to forecasted load
                deliverable_mw = min(battery_p_mw, forecasted_load)
            else:
                # Bid full battery power (grid injection protection is handled in execute_activation)
                deliverable_mw = battery_p_mw

            # Check if cycle limit allows activation
            cycle_limit_reached = day_cyc >= max_cycles_per_day

            if cycle_limit_reached:
                # Skip activation - cycle limit reached
                activated = False
                delivered_mwh = 0
                cyc_inc = 0
                undel_mwh = 0
                undel_reason = None
                day_skipped += 1

                # Check if bid would have been accepted to determine if this counts as cycle_limit underdelivery
                bid_accepted = bid_price <= cleared_price
                if bid_accepted:
                    # Calculate what would have been delivered if not cycle limited
                    potential_undel = deliverable_mw * ptu_hours
                    day_undel_cycle += potential_undel
                    undel_reason = "cycle_limit"
            else:
                # Execute activation with baseline comparison
                activated, delivered_mwh, soc, cyc_inc, undel_mwh, undel_reason = execute_activation(
                    soc, battery_p_mw, deliverable_mw, ptu_hours, battery, bid_dir, cleared_price, bid_price, baseline_load, actual_load
                )

            # Update baseline: if not activated, this becomes the new baseline PTU
            if not activated:
                last_non_activated_idx = idx

            # SoC Protection: recover rest_soc when not activated
            if not activated and soc_protection and not cycle_limit_reached:
                # Use current PTU's load to enforce no-grid-injection constraint
                current_load = load[idx]
                energy_moved, soc, recovery_cyc = recover_rest_soc(soc, rest_soc, battery, ptu_hours, current_load)

                # Only add recovery cycles if it doesn't exceed limit
                if day_cyc + recovery_cyc <= max_cycles_per_day:
                    day_cyc += recovery_cyc
                else:
                    # Partial or no recovery due to cycle limit
                    remaining_cycles = max_cycles_per_day - day_cyc
                    if remaining_cycles > 0:
                        # Allow partial recovery
                        day_cyc += remaining_cycles
                    day_skipped += 1

            if activated:
                day_act += 1
            day_rev += calculate_ptu_revenue(delivered_mwh, cleared_price, bid_dir)
            day_cyc += cyc_inc

            # Track undelivered energy by reason and calculate penalty
            ptu_penalty = 0.0
            if undel_mwh > 0:
                day_undel += 1
                day_undel_energy += undel_mwh

                # Track by reason (excluding cycle_limit which is handled above)
                if undel_reason == "load_limit":
                    day_undel_load += undel_mwh
                elif undel_reason == "soc_limit_low":
                    day_undel_soc_low += undel_mwh
                elif undel_reason == "soc_limit_high":
                    day_undel_soc_high += undel_mwh

                # Calculate penalty: small fixed penalty + undelivered energy × imbalance price
                ptu_penalty = small_penalty + (undel_mwh * imbalance_price)
                day_penalty += ptu_penalty

            soctrace.append(soc)

            # Calculate PTU revenue
            ptu_revenue = calculate_ptu_revenue(delivered_mwh, cleared_price, bid_dir)

            # Store PTU-level results
            ptu_results_list.append({
                'timestamp': timestamps[idx],
                'day': day + 1,
                'ptu': ptu,
                'hour': hour,
                'bid_direction': bid_dir,
                'bid_price': bid_price,
                'cleared_price': cleared_price,
                'activated': activated,
                'baseline_load_mw': baseline_load,
                'actual_load_mw': actual_load,
                'forecasted_load_mw': forecasted_load if forecasted_load is not None else actual_load,
                'deliverable_mw': deliverable_mw,
                'delivered_mwh': delivered_mwh,
                'undelivered_mwh': undel_mwh,
                'soc': soc,
                'ptu_revenue': ptu_revenue,
                'ptu_penalty': ptu_penalty,
                'cycle_increment': cyc_inc,
                'skipped_due_to_cycle_limit': cycle_limit_reached,
                'underdelivery_reason': undel_reason
            })

        # Apply penalty to revenue (net revenue = gross revenue - penalties)
        net_revenue = day_rev - day_penalty

        revenues.append(net_revenue)
        cycles.append(day_cyc)
        activations.append(day_act)
        undelivered.append(day_undel)
        skipped_due_to_cycles.append(day_skipped)
        undelivered_energy_mwh.append(day_undel_energy)
        penalties.append(day_penalty)
        undel_cycle_limit.append(day_undel_cycle)
        undel_load_limit.append(day_undel_load)
        undel_soc_low.append(day_undel_soc_low)
        undel_soc_high.append(day_undel_soc_high)

    # Clear progress indicators and show completion
    progress_bar.empty()
    progress_text.empty()

    # Create PTU-level results DataFrame
    ptu_df = pd.DataFrame(ptu_results_list)

    # Calculate aggregate metrics for underdelivery by reason
    total_undel_cycle_limit = sum(undel_cycle_limit)
    total_undel_load_limit = sum(undel_load_limit)
    total_undel_soc_low = sum(undel_soc_low)
    total_undel_soc_high = sum(undel_soc_high)

    # Store all results in session state
    st.session_state.simulation_results = {
        'ptu_df': ptu_df,
        'revenues': revenues,
        'cycles': cycles,
        'activations': activations,
        'undelivered': undelivered,
        'skipped_due_to_cycles': skipped_due_to_cycles,
        'undelivered_energy_mwh': undelivered_energy_mwh,
        'penalties': penalties,
        'soctrace': soctrace,
        'num_days': num_days,
        'ptus_per_day': ptus_per_day,
        'ptu_hours': ptu_hours,
        'cleared_price_up': cleared_price_up,
        'cleared_price_down': cleared_price_down,
        'strategy_df': strategy_df,
        # Underdelivery by reason
        'total_undel_cycle_limit': total_undel_cycle_limit,
        'total_undel_load_limit': total_undel_load_limit,
        'total_undel_soc_low': total_undel_soc_low,
        'total_undel_soc_high': total_undel_soc_high
    }

# -----------------------------
# Display Results (use session state if available)
# -----------------------------
if st.session_state.simulation_results is not None:
    # Retrieve results from session state
    results = st.session_state.simulation_results
    ptu_df = results['ptu_df']
    revenues = results['revenues']
    cycles = results['cycles']
    activations = results['activations']
    undelivered = results['undelivered']
    skipped_due_to_cycles = results['skipped_due_to_cycles']
    undelivered_energy_mwh = results['undelivered_energy_mwh']
    penalties = results['penalties']
    soctrace = results['soctrace']
    num_days = results['num_days']
    ptus_per_day = results['ptus_per_day']
    ptu_hours = results['ptu_hours']
    cleared_price_up = results['cleared_price_up']
    cleared_price_down = results['cleared_price_down']
    strategy_df = results['strategy_df']
    # Underdelivery by reason
    total_undel_cycle_limit = results['total_undel_cycle_limit']
    total_undel_load_limit = results['total_undel_load_limit']
    total_undel_soc_low = results['total_undel_soc_low']
    total_undel_soc_high = results['total_undel_soc_high']

    # Overview Section (always expanded)
    with st.expander("📊 Overview", expanded=True):
        st.subheader("Simulation Summary")

        # Calculate revenue metrics
        total_gross_revenue = ptu_df['ptu_revenue'].sum()  # Revenue before penalties
        total_penalties = sum(penalties)
        total_net_revenue = sum(revenues)  # Already net (gross - penalties)

        # Annualize all metrics (convert to per-year values)
        gross_revenue = (total_gross_revenue / num_days) * 365
        penalties = (total_penalties / num_days) * 365
        net_revenue = (total_net_revenue / num_days) * 365

        # First row: Revenue metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Revenue (€/yr)", f"{gross_revenue:,.0f}", help="Gross revenue (before penalties) annualized from simulation period")
        col2.metric("Penalties (€/yr)", f"{penalties:,.0f}", help="Total penalties from underdelivery, annualized")
        col3.metric("Net Revenue (€/yr)", f"{net_revenue:,.0f}", help="Net revenue (after penalties) annualized from simulation period")
        col4.metric("Activations (/day)", f"{sum(activations)/num_days:.1f}", help="Average number of accepted bids executed per day")

        # Second row: Underdelivery diagnosis (annualized)
        undel_cycle_limit = (total_undel_cycle_limit / num_days) * 365
        undel_load_limit = (total_undel_load_limit / num_days) * 365
        undel_soc_low = (total_undel_soc_low / num_days) * 365
        undel_soc_high = (total_undel_soc_high / num_days) * 365

        st.markdown("##### Underdelivery Diagnosis")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Cycle Limit (MWh/yr)", f"{undel_cycle_limit:.2f}", help="Energy not delivered because daily cycle limit was reached, annualized")
        col6.metric("Load Limit (MWh/yr)", f"{undel_load_limit:.2f}", help="Energy not delivered due to grid injection constraint (load too low), annualized")
        col7.metric("Low SoC (MWh/yr)", f"{undel_soc_low:.2f}", help="Energy not delivered due to insufficient battery charge (UP regulation), annualized")
        col8.metric("High SoC (MWh/yr)", f"{undel_soc_high:.2f}", help="Energy not delivered because battery was full (DOWN regulation), annualized")

    # Operations Section
    with st.expander("📈 Operations", expanded=False):
        st.subheader("First Day: Cleared Prices vs Bids")
        plot_first_day_prices(ptus_per_day, ptu_hours, cleared_price_up, cleared_price_down, strategy_df)
        st.subheader("Battery SoC Over Time")
        plot_soc(soctrace)

    # Revenues Section
    with st.expander("💰 Revenues", expanded=True):
        st.subheader("Revenue Analysis")

        # Time aggregation selector
        aggregation_options = {
            "Monthly Aggregates": "monthly",
            "Daily Aggregates (select month)": "daily",
            "PTU-level (select day)": "ptu"
        }

        selected_aggregation_label = st.selectbox(
            "Time Aggregation Level",
            options=list(aggregation_options.keys()),
            index=0  # Default to monthly
        )

        aggregation_level = aggregation_options[selected_aggregation_label]

        # Conditional period selectors
        selected_period = None
        selected_time_range = None

        if aggregation_level == "ptu":
            # Select specific date using calendar picker
            ptu_df_temp = ptu_df.copy()
            ptu_df_temp['timestamp'] = pd.to_datetime(ptu_df_temp['timestamp'])
            ptu_df_temp['date'] = ptu_df_temp['timestamp'].dt.date

            # Get min/max dates for bounds
            min_date = ptu_df_temp['date'].min()
            max_date = ptu_df_temp['date'].max()

            selected_date = st.date_input(
                "Select Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="Pick a date from the calendar or type it manually (YYYY-MM-DD)"
            )

            # Add time range selector
            time_range = st.slider(
                "Select Time Range (hours)",
                min_value=0,
                max_value=24,
                value=(0, 24),
                step=1,
                help="Select the hour range to display. Default shows full day (0-24h)"
            )

            # Get corresponding day number for plotting
            date_mapping = ptu_df_temp.groupby('date')['day'].first()
            if selected_date in date_mapping.index:
                selected_period = date_mapping[selected_date]
                # Store time range for filtering in plot
                selected_time_range = time_range
            else:
                st.error(f"No data available for {selected_date}. Please select a date within the simulation period.")
                selected_period = None
                selected_time_range = None
        elif aggregation_level == "daily":
            # Select specific month - extract available months from data
            ptu_df_temp = ptu_df.copy()
            ptu_df_temp['timestamp'] = pd.to_datetime(ptu_df_temp['timestamp'])
            ptu_df_temp['year_month'] = ptu_df_temp['timestamp'].dt.to_period('M')
            ptu_df_temp['month_label'] = ptu_df_temp['timestamp'].dt.strftime('%b %Y')

            # Get unique months
            available_months = ptu_df_temp.groupby('year_month')['month_label'].first().reset_index()

            selected_period = st.selectbox(
                "Select Month",
                options=available_months['year_month'].tolist(),
                format_func=lambda x: available_months[available_months['year_month'] == x]['month_label'].iloc[0],
                index=0
            )

        # Plot revenue analysis (only if valid period selected, or if monthly which doesn't need a period)
        if selected_period is not None or aggregation_level == "monthly":
            # Pass time range for PTU-level, None for others
            time_range_filter = selected_time_range if aggregation_level == "ptu" else None
            plot_revenue_analysis(ptu_df, aggregation_level, selected_period, time_range_filter)
