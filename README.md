# BTM BESS Business Case Calculator

A Streamlit-based tool for optimizing Behind-The-Meter (BTM) Battery Energy Storage System (BESS) sizing and analyzing business case viability.

**Author:** Otto Fabius - Sympower
**Last Updated:** September 2025

---

## Overview

This tool helps energy professionals and researchers evaluate the economic viability of BTM battery storage systems by:
- Optimizing battery dispatch schedules using linear programming
- Evaluating different battery sizes across multiple scenarios
- Calculating financial metrics including ROI, payback period, and annual savings
- Visualizing battery behavior, load profiles, and cost comparisons

The optimizer considers day-ahead energy prices, grid fees, demand charges, PV generation, and operational constraints to determine optimal battery charging/discharging schedules.

---

## Key Features

### For Users

Access the tool here
https://ofabius-spw-btm-bess-btm-bess-businesscase-calculator-nh9szq.streamlit.app/

- **Multiple data input options:**
  - Upload real CSV data (load, PV, prices, grid fees)
  - Generate synthetic profiles from predefined scenarios
  - Customize scenario parameters

- **Comprehensive optimization:**
  - Single-day or full-year optimization
  - Battery size sweep with customizable ranges
  - Linear programming solver for optimal dispatch
  - Configurable operational constraints (SoC limits, efficiency, import/export caps)

- **Financial analysis:**
  - CapEx and OpEx calculations
  - Annual cost/savings projections
  - ROI and payback period metrics
  - Demand charge impact analysis

- **Rich visualizations:**
  - Daily load and price profiles
  - Yearly overview with variance bands
  - Battery behavior analysis (power, SoC, import/export)
  - Cost comparison across battery sizes
  - Idle hours analysis by SoC level
  - Daily savings breakdown

- **Export capabilities:**
  - Downloadable experiment summary CSV
  - All key metrics and parameters included

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd btm_bess
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install streamlit pandas numpy matplotlib pulp pytz
```

---

## Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run btm_bess_businesscase_calculator.py
```

The app will open in your default web browser (typically at `http://localhost:8501`).

### Basic Workflow

1. **Configure simulation settings** (sidebar):
   - Battery specifications (CapEx, project horizon, cycle duration)
   - Operational settings (SoC limits, import/export caps, demand charges)
   - Advanced settings available in expander

2. **Input data** (Section 1):
   - **Option A:** Upload CSV file with columns: `load`, `pv`, `use_price`, `inject_price`, `grid_fee`
   - **Option B:** Select a predefined scenario for synthetic data generation
     - Profitable arbitrage
     - Unviable (Load matches PV)
     - Viable for optimal BESS size
     - Custom (adjust all parameters)

3. **Adjust data parameters:**
   - PV multiplier (MWp per MW average load)
   - Price variance (consumption and production)
   - Optional: disable load for PV-only analysis

4. **Inspect profiles:**
   - View yearly overview (if uploaded data)
   - Explore daily profiles with day selector

5. **Run optimization:**
   - Choose horizon: First day only or Full year
   - Set battery size range (min, max, steps)
   - Click "Run Optimisation"

6. **Analyze results:**
   - View summary metrics (costs, savings, payback)
   - Explore battery behavior for specific days
   - Examine cost comparison across sizes
   - Download experiment summary

### CSV Upload Format

Required columns:
- `load` (MW) - electricity demand per PTU
- `pv` (MW) - PV generation per PTU
- `use_price` (€/MWh) - consumption price
- `inject_price` (€/MWh) - feed-in tariff
- `grid_fee` (€/MWh) - grid usage fee

Expected format: One row per Program Time Unit (PTU), typically 96 rows/day for 15-minute intervals, 365 days for full year (35,040 rows).

---

## Understanding the Results

### Key Metrics

- **Yearly cost without battery use**: Baseline annual energy costs
- **Capital cost**: Total upfront battery investment
- **Cost excl write-off**: Annual operating costs with battery (energy only)
- **Write-off cost**: Annual CapEx amortization
- **Years until payback**: Time to recover initial investment
- **Peak import with battery**: Maximum grid import power (relevant for demand charges)
- **Gross savings excl capex**: Annual operational savings before CapEx recovery
- **Average cycles per day**: Battery utilization metric

### Scenario Interpretations

**Profitable Arbitrage:**
- High price volatility justifies arbitrage
- Larger batteries = more savings (linear relationship)
- Best when day-ahead spreads are significant

**Unviable (Load Matches PV):**
- Load profile closely follows PV generation
- Limited opportunity for battery value
- May show negative ROI

**Viable for Optimal BESS Size:**
- Grid fees or moderate price spreads create value
- Optimal battery size exists (not "bigger is better")
- Best for self-consumption maximization

---

## For Developers

### Architecture

The application follows a modular design:

```
btm_bess/
├── btm_bess_businesscase_calculator.py  # Main Streamlit app
├── plotting.py                          # Visualization functions
├── da_schedule_with_reserve.csv         # Sample data file
└── README.md                            # This file
```

### Code Structure

**btm_bess_businesscase_calculator.py:**
- `main()` - Streamlit app entry point
- `get_user_inputs()` - Sidebar configuration collection
- `generate_mock_profiles()` - Synthetic data generation (duck curve)
- `build_and_solve_lp()` - Linear programming optimization model
- `run_daily_optimisation()` - Single-day optimization wrapper
- `run_yearly_optimisation()` - Multi-day optimization with progress tracking
- `render_*()` functions - Modular UI sections

**plotting.py:**
- `plot_daily_profiles()` - Load/PV and price plots
- `plot_yearly_overview()` - Aggregated yearly visualizations
- `plot_cost_comparison()` - Battery size sweep results
- `plot_battery_behavior()` - Detailed dispatch visualization
- `plot_idle_hours()` - Battery utilization analysis
- `plot_daily_savings()` - Time-series savings plot

### Optimization Model

The tool uses PuLP for linear programming optimization with:

**Decision Variables:**
- Battery power (charge/discharge) per PTU
- State of Charge (SoC) per PTU
- Grid import/export per PTU
- PV curtailment (if enabled)
- Peak import (for demand charges)

**Constraints:**
- Power balance: Load = PV - Curtailment + Battery + Grid
- SoC dynamics with round-trip efficiency
- SoC limits (min/max)
- Battery power limits
- Import/export caps
- Optional: End-of-day SoC = Start SoC
- Optional: No grid export (curtailment mode)

**Objective:**
Minimize total daily cost = Import costs - Export revenue + Grid fees + Demand charges

### Key Parameters

Stored in `params` dict:
- `ptu_count`, `dt` - Time discretization
- `duration_hrs` - Battery energy/power ratio
- `eta` - Round-trip efficiency
- `soc_init_pct`, `soc_min_pct`, `soc_max_pct` - State of charge limits
- `import_cap`, `export_cap` - Grid connection limits
- `demand_charge` - Peak power charge (€/MW/year)
- `price_eur_per_mw` - Battery CapEx
- `project_horizon_yrs` - Amortization period
- `apply_curtailment` - No export mode

### Session State Management

The app uses Streamlit's session state to persist:
- `scenario` - Selected scenario name
- `generation_settings` - Parameters used for synthetic data
- `optimization_results` - Full results dict (summary, daily data, detailed results)
- `selected_day`, `battery_behavior_day` - UI state for day selectors

### Adding New Features

**New Scenario:**
1. Add entry to `scenario_params` dict in `render_data_input_section()`
2. Add description to `scenario_texts` dict
3. Add button in columns layout

**New Constraint:**
1. Modify `build_and_solve_lp()` to add constraint to `prob`
2. Add parameter to `params` dict in `get_user_inputs()`
3. Update UI in sidebar

**New Visualization:**
1. Create function in `plotting.py` following existing pattern
2. Import in main file
3. Call from `render_results_summary()` or other render function

**New Metric:**
1. Calculate in `run_yearly_optimisation()` or `render_results_summary()`
2. Add to export DataFrame in `render_results_summary()`
3. Display using `st.metric()` or add to plots

### Testing

Current implementation focuses on:
- PTU duration = 0.25h (15-minute intervals)
- Linear efficiency model (constant η)
- Single battery system
- Day-ahead pricing only (no intraday/imbalance)

When testing new features:
- Verify with single-day optimization first
- Check edge cases (zero PV, zero load, price inversions)
- Validate SoC stays within bounds
- Ensure objective value matches manual calculation

### Performance Considerations

- **Full year optimization**: 365 days × N battery sizes × T PTUs (computational intensive)
- Limit battery size sweep to ≤10 sizes for full year
- Progress bar provides user feedback
- Results cached in session state to avoid re-computation

### Dependencies

Core:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `pulp` - Linear programming solver
- `pytz` - Timezone handling

---

## Limitations & Future Enhancements

### Current Limitations

- Single battery only (no multiple systems)
- No battery degradation modeling
- Day-ahead prices only (no intraday/imbalance markets)
- Constant efficiency (no SoC-dependent losses)
- Perfect foresight (optimization with known prices)
- No ancillary services revenue modeling

### Potential Enhancements

- Battery degradation (cycle counting, calendar aging)
- Multi-market optimization (DA + intraday + FCR)
- Rolling horizon optimization (realistic forecasting)
- Stochastic optimization (price uncertainty)
- Multiple battery systems
- DC-coupled vs AC-coupled PV+battery modeling
- Time-of-use tariff structures
- Seasonal storage analysis
- Sensitivity analysis automation

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

Please ensure:
- Code follows existing style conventions
- Functions include docstrings
- Complex logic is commented
- New features include appropriate visualizations

---

## License

[Specify license here]

---

## Contact

For questions or feedback, please contact:
- **Author:** Otto Fabius
- **Organization:** Sympower

---

## Acknowledgments

Built using open-source tools:
- Streamlit for web interface
- PuLP for optimization
- Matplotlib for visualizations
- NumPy/Pandas for data processing
