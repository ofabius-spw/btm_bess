# plotting.py
"""
Plotting functions for BTM BESS optimization visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_profiles(df: pd.DataFrame, T: int, selected_day: int):
    """
    Plot load, PV, and price profiles for a selected day.

    Args:
        df: DataFrame with load, pv, use_price, inject_price columns
        T: Number of PTUs per day
        selected_day: Day index to plot

    Returns:
        tuple: (fig_profile, fig_price) - two matplotlib figures
    """
    day_df = df.iloc[selected_day*T : (selected_day+1)*T]
    ts = np.arange(T) * 24 / T

    # Load and PV plot
    fig_profile, ax_profile = plt.subplots(figsize=(10, 3))
    ax_profile.plot(ts, day_df['load'].values, label='Load')
    ax_profile.plot(ts, day_df['pv'].values, label='PV')
    ax_profile.set_xlabel("Hour of day")
    ax_profile.set_ylabel("Power (MW)")
    ax_profile.set_title("Load and PV profiles")
    ax_profile.grid(True)
    ax_profile.legend()

    # Price plot
    fig_price, ax_price = plt.subplots(figsize=(10, 3))
    ax_price.plot(ts, day_df['use_price'], label='DA Price', color='tab:orange')
    ax_price.plot(ts, day_df['inject_price'], label='Feed-in Price', color='tab:green')
    ax_price.set_xlabel("Hour of day")
    ax_price.set_ylabel("Price (€/MWh)")
    ax_price.set_title("Energy prices")
    ax_price.grid(True)
    ax_price.legend()

    return fig_profile, fig_price


def plot_yearly_overview(df: pd.DataFrame, PTU_per_day: int = 96):
    """
    Plot yearly average load profile (daily and hourly views).

    Args:
        df: DataFrame with load and pv columns
        PTU_per_day: Number of PTUs per day

    Returns:
        tuple: (fig_daily, fig_hourly) - two matplotlib figures
    """
    # Truncate to full days
    n_days = len(df) // PTU_per_day
    df_trunc = df.iloc[:n_days * PTU_per_day].copy()

    # PTU within the day
    df_trunc['ptu'] = df_trunc.index % PTU_per_day

    # Compute average and variance per PTU
    avg_load = df_trunc.groupby('ptu').apply(lambda x: (x['load'] + x['pv']).mean())
    std_load = df_trunc.groupby('ptu').apply(lambda x: (x['load'] + x['pv']).std())
    hours = np.arange(PTU_per_day) * 24 / PTU_per_day

    # Reshape for daily view
    load_reshaped = (df_trunc['load'] + df_trunc['pv']).values.reshape(n_days, PTU_per_day)
    avg_daily = pd.Series(load_reshaped.mean(axis=1), index=np.arange(1, n_days + 1))
    std_daily = pd.Series(load_reshaped.std(axis=1), index=np.arange(1, n_days + 1))

    # Daily profile
    fig_daily, ax_daily = plt.subplots(figsize=(10, 3))
    ax_daily.plot(avg_daily.index, avg_daily.values, label='Average Load', color='blue')
    ax_daily.fill_between(avg_daily.index,
                          avg_daily.values - np.sqrt(std_daily.values),
                          avg_daily.values + np.sqrt(std_daily.values),
                          color='blue', alpha=0.2, label='±1 Std Dev')
    ax_daily.set_xlabel('Day of Year')
    ax_daily.set_ylabel('Load')
    ax_daily.set_title('Yearly net load profile averaged over the day')
    ax_daily.legend()
    ax_daily.grid(True)

    # Hourly profile
    fig_hourly, ax_hourly = plt.subplots(figsize=(10, 3))
    ax_hourly.plot(hours, avg_load.values, label='Average net load', color='blue')
    ax_hourly.fill_between(hours,
                           avg_load.values - std_load.values,
                           avg_load.values + std_load.values,
                           color='blue', alpha=0.2, label='±1 Std Dev')
    ax_hourly.set_xlabel('Hour of Day')
    ax_hourly.set_ylabel('Load')
    ax_hourly.set_title('Net load profile averaged over the year with Variance')
    ax_hourly.legend()
    ax_hourly.grid(True)
    plt.xticks(np.arange(0, 25, 2))

    return fig_daily, fig_hourly


def plot_cost_comparison(summary_df: pd.DataFrame):
    """
    Plot annual costs vs battery size.

    Args:
        summary_df: DataFrame with cost columns and size_mw

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary_df['size_mw'], summary_df['annual_cost_no_batt']/1000,
            label='No battery (k€)', linestyle="--", marker='o')
    ax.plot(summary_df['size_mw'], summary_df['annual_operating_cost_with_batt']/1000,
            label='With battery excl. capex (k€)', marker='o')
    ax.plot(summary_df['size_mw'], summary_df['annual_cost_with_batt_incl_capex']/1000,
            label='With battery incl. capex (k€)', marker='o')
    ax.set_xlabel("Battery size (MW)")
    ax.set_ylabel("Annual cost (k€)")
    ax.set_title("Annual costs vs battery size")
    ax.grid(True)
    ax.legend()
    return fig


def plot_battery_behavior(results_day: dict, df_run: pd.DataFrame, T: int,
                          params: dict, best_size: float, day_to_show: int):
    """
    Plot detailed battery behavior including power, SoC, import/export.

    Args:
        results_day: Results dictionary from optimization
        df_run: DataFrame with load, pv, use_price data
        T: Number of PTUs
        params: Parameters dict with dt
        best_size: Battery size in MW
        day_to_show: Day number being shown

    Returns:
        matplotlib.figure.Figure
    """
    ts = np.arange(T) * params['dt']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = {
        'Load (MW)': 'tab:blue',
        'PV (MW)': 'tab:green',
        'Net before battery': 'tab:orange',
        'Battery power (+charging, -discharging)': 'tab:red',
        'Import': 'tab:purple',
        'Export': 'tab:brown',
        'SoC (MWh)': 'black'
    }

    # First subplot: battery power, SoC, net before battery
    ax1.plot(ts, df_run['load'].values[:T] - df_run['pv'].values[:T],
            label='Net before battery', color=colors['Net before battery'])
    ax1.plot(ts, results_day['P'],
            label='Battery power (+charging, -discharging)',
            color=colors['Battery power (+charging, -discharging)'])
    ax1.plot(ts, results_day['SoC'],
            label='SoC (MWh)', color=colors['SoC (MWh)'])
    ax1.plot(ts, df_run['use_price'].values[:T]/100, label='Price (100€/MWh)', color='tab:gray')
    ax1.set_ylabel('MW / MWh')
    ax1.set_title(f"Day {day_to_show} profiles (Battery size: {best_size} MW)")
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax1.grid(True)

    # Second subplot: load, PV, battery power, import/export
    ax2.plot(ts, df_run['load'].values[:T],
            label='Load (MW)', color=colors['Load (MW)'])
    ax2.plot(ts, df_run['pv'].values[:T],
            label='PV (MW)', color=colors['PV (MW)'])
    ax2.plot(ts, results_day['P'],
            label='Battery power (+charging, -discharging)',
            color=colors['Battery power (+charging, -discharging)'])
    ax2.plot(ts, results_day['G_import'],
            label='Import', color=colors['Import'])
    ax2.plot(ts, results_day['G_export'],
            label='Export', color=colors['Export'])

    # Add vertical dotted lines at every whole hour
    for hour in range(int(ts[-1]) + 1):
        ax1.axvline(x=hour, color='gray', linestyle=':', linewidth=0.8)
        ax2.axvline(x=hour, color='gray', linestyle=':', linewidth=0.8)

    ax2.set_xlabel('Hour')
    ax2.set_ylabel('MW')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax2.grid(True)

    return fig


def plot_idle_hours(detailed_results: dict, best_size: float, params: dict):
    """
    Plot daily idle hours split by SoC level.

    Args:
        detailed_results: Dict of day -> optimization results
        best_size: Battery size in MW
        params: Parameters dict with dt and duration_hrs

    Returns:
        matplotlib.figure.Figure
    """
    idle_low50_per_day = []
    idle_high50_per_day = []
    days = list(detailed_results.keys())
    max_power = best_size

    for day_results in detailed_results.values():
        P = np.array(day_results['P'])
        SoC = np.array(day_results['SoC'])
        idle_mask = np.abs(P) < 0.1 * max_power
        soc_pct = SoC / (best_size * params['duration_hrs']) * 100
        idle_low50_per_day.append(np.sum(idle_mask & (soc_pct < 50)) * params['dt'])
        idle_high50_per_day.append(np.sum(idle_mask & (soc_pct >= 50)) * params['dt'])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(days, idle_low50_per_day, color='tab:orange', label='SOC < 50%')
    ax.bar(days, idle_high50_per_day, bottom=idle_low50_per_day,
           color='tab:blue', label='SOC ≥ 50%')
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Idle hours per day")
    ax.set_title(f"Daily idle (<10% of max power) hours for best battery size ({best_size} MW)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    return fig


def plot_daily_savings(daily_df: pd.DataFrame, best_size: float):
    """
    Plot savings per day for the best battery size.

    Args:
        daily_df: DataFrame with daily results
        best_size: Battery size in MW

    Returns:
        matplotlib.figure.Figure
    """
    best_daily_df = daily_df[daily_df['size_mw'].round(2) == round(best_size, 2)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Daily savings (EUR)")
    ax.set_title(f"Savings per day for best battery size ({best_size} MW)")
    ax.plot(best_daily_df['savings'])
    return fig
