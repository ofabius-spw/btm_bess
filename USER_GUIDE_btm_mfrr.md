# BTM BESS mFRR Bidding Simulator - User Guide

## General Introduction

This application simulates a Behind-The-Meter (BTM) Battery Energy Storage System (BESS) participating in the manual Frequency Restoration Reserve (mFRR) market. The simulator helps evaluate bidding strategies and revenue potential for batteries installed at Commercial & Industrial (C&I) sites.

**Key Features:**
- Simulate battery operations over multiple days with 15-minute PTUs
- Define custom bidding strategies with time-varying prices and directions
- Visualize market data and bidding strategy
- Calculate net revenues including penalties for underdelivery
- Enforce realistic constraints (SoC limits, cycle limits, no grid injection)

---

## Inputs

### 1. Data Input (CSV Upload - Optional)

Upload a CSV file with the following columns:
- **`timestamp`**: Date and time (e.g., "2024-01-01 00:00:00")
- **`load_kw`**: Site load in kilowatts (kW)
- **`cleared_price_up`**: Cleared UP regulation price (€/MWh)
- **`cleared_price_down`**: Cleared DOWN regulation price (€/MWh)

**Note:** If no file is uploaded, the simulator uses mock data.

### 2. Simulation Settings (Sidebar)

**Number of Days:** Duration of simulation (1-30 days)

**PTU Duration:** Fixed at 0.25 hours (15 minutes)

**SoC Protection:** When enabled, battery actively moves toward target "Rest SoC" during non-activated periods

### 3. Battery Parameters (Sidebar)

- **Power (MW):** Maximum charge/discharge rate
- **Depth (hours):** Battery energy capacity = Power × Depth (MWh)
- **Initial SoC:** Starting state of charge (0.0 to 1.0)
- **Min/Max SoC:** Fixed at 0.0 and 1.0 (0% to 100%)
- **Max Cycles per Day:** Maximum battery cycles allowed per day (1 cycle = full discharge + full charge equivalent)

### 4. Underdelivery Penalty Settings (Sidebar)

- **Imbalance Price (€/MWh):** Price used to calculate penalty for undelivered energy
- **Small Penalty (€):** Fixed penalty per underdelivery event

**Penalty Formula:** `Total Penalty = Small Penalty + (Undelivered Energy × Imbalance Price)`

### 5. Bidding Strategy

Define your bidding strategy in the editable table:
- **Start (h):** Start hour of period (0-23)
- **End (h):** End hour of period (1-24)
- **Direction:**
  - `UP` = Decrease consumption (battery discharges)
  - `DOWN` = Increase consumption (battery charges)
- **Price (€/MWh):** Your bid price for this period
- **Rest SoC:** Target state of charge during this period for active SoC management (0.0-1.0)
- **% of max power:** Reserved for future use (currently 1.0)

### 6. Market Data Visualization Options

- **Average Daily Prices:** Shows average prices across all days in uploaded data
- **Select a Specific Day:** View market data for a single day
- **None:** Show only bidding strategy (no market overlay)

---

## Assumptions

### Market & Activation Rules

1. **Baseline Comparison:** TSO compares current load to the last non-activated PTU to determine required battery delivery
2. **Bid Acceptance:** Bid is accepted if `bid_price ≤ cleared_price`
3. **PTU Resolution:** 15-minute intervals (96 PTUs per day)

### Battery Constraints

1. **SoC Limits:** Battery operates between 0% (empty) and 100% (full)
2. **No Grid Injection:** For UP regulation (discharge), battery output is capped by site load
3. **Power Limits:** Battery cannot exceed its rated power capacity
4. **Cycle Limits:** Daily cycle budget enforced - operations stop when limit reached

### Revenue Calculation

**UP Regulation (Discharge):**
```
Revenue = Delivered Energy (MWh) × Cleared Price (€/MWh)
```
- Positive price → Profit
- Negative price → Debit

**DOWN Regulation (Charge):**
```
Revenue = -(Delivered Energy (MWh) × Cleared Price (€/MWh))
```
- Positive price → Debit (you pay to consume)
- Negative price → Profit (you're paid to consume)

### SoC Protection

When enabled:
- During non-activated PTUs, battery moves toward "Rest SoC" at full power
- Recovery respects all constraints (power limits, SoC bounds, no grid injection)
- Recovery cycles count toward daily cycle limit

---

## Simulation Method

### For Each PTU:

1. **Determine Baseline Load:**
   - Use load from the last PTU where there was NO activation
   - This becomes the reference for calculating required battery delivery

2. **Execute Bid Strategy:**
   - Retrieve bid direction, price, and rest SoC for current hour
   - Check if bid is accepted: `bid_price ≤ cleared_price`

3. **Check Cycle Limit:**
   - If daily cycle limit reached, skip activation and recovery
   - Track skipped PTUs

4. **Calculate Required Battery Delivery (if activated):**

   **UP Regulation (Decrease Consumption):**
   ```
   Expected Net Consumption = Baseline - Activation Target
   Battery Must Deliver = Actual Load - Expected Net
                        = Target - (Baseline - Actual)
   ```

   **DOWN Regulation (Increase Consumption):**
   ```
   Expected Net Consumption = Baseline + Activation Target
   Battery Must Deliver = Expected Net - Actual Load
                        = Target + (Baseline - Actual)
   ```

5. **Execute Activation:**
   - Check SoC availability
   - Apply power limits and no-grid-injection constraint
   - Calculate delivered vs. undelivered energy
   - Update SoC and cycle count

6. **SoC Recovery (if not activated and SoC Protection enabled):**
   - Move battery toward Rest SoC at full power
   - Respect all constraints
   - Count recovery cycles

7. **Calculate Revenue and Penalties:**
   - Revenue: Based on direction and cleared price (see formula above)
   - Penalty: Applied for any undelivered energy
   - Net Revenue = Gross Revenue - Penalties

8. **Update Baseline:**
   - If NOT activated this PTU, update baseline to current load

---

## Outputs

### Simulation Summary (Overview Tab)

**First Row:**
- **Net Revenue (€):** Total revenue after penalties
- **Total Activations:** Count of accepted and executed bids
- **Total Cycles:** Battery cycles used across simulation
- **Total Penalties (€):** Sum of all underdelivery penalties

**Second Row:**
- **Revenue/Day (€):** Average daily net revenue
- **Cycle-Limited PTUs:** PTUs skipped due to cycle limit
- **Avg Cycles/Day:** Average daily cycle usage
- **Undelivered MWh:** Total energy not delivered due to constraints

### Plots (Plots Tab)

**1. First Day: Cleared Prices vs Bids**
- Shows your bid prices vs. actual cleared prices for day 1
- Green = UP regulation
- Red = DOWN regulation
- Helps visualize bid acceptance

**2. Battery SoC Over Time**
- State of charge throughout entire simulation
- Range: 0.0 (empty) to 1.0 (full)
- Shows charging/discharging patterns and SoC management

### Strategy Visualization

**Integrated Plot:**
- **Solid lines:** Your bid prices (green=UP, red=DOWN)
- **Dotted lines:** Cleared market prices (if data uploaded)
- **Dashed line (right axis):** Site load profile in kW
- Toggle between average daily prices or specific day view

---

## Tips for Effective Use

1. **Start with Average Prices:** Use "Average Daily Prices" view to understand typical market patterns
2. **Match Peak Hours:** Align UP bids (discharge) with high-price periods
3. **Mind the Cycles:** Monitor cycle usage - hitting the daily limit stops all operations
4. **Watch SoC:** Enable SoC Protection to maintain battery readiness between activations
5. **Test Sensitivity:** Vary bid prices to find optimal balance between acceptance rate and revenue
6. **Check Penalties:** High undelivered energy indicates SoC management issues

---

## Common Issues

**Q: Many undelivered PTUs**
- Battery may be reaching SoC limits (too empty or too full)
- Consider adjusting Rest SoC targets or bidding strategy
- Increase battery capacity (depth) if needed

**Q: SoC stuck at extremes**
- Enable SoC Protection to actively manage state of charge
- Adjust Rest SoC targets to keep battery in usable range
- Review bidding direction balance (too much UP or DOWN)
