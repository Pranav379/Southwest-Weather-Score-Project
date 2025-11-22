# Flight Delay Predictor Dashboard

## Overview

The **Flight Delay Predictor** is an interactive Streamlit web application designed to forecast weather-related flight delays using a heuristic risk scoring model. This dashboard provides travelers and airline operators with real-time risk assessments based on meteorological data and flight characteristics.

## What This Dashboard Brings to Your Project

### Key Features

1. **Real-Time Risk Assessment**: Evaluates weather conditions against historical delay patterns to produce a 0-100 risk score where 0 = best conditions and 100 = worst conditions.

2. **Dual Data Input Methods**:
   - **Mock Flight Search**: Simulate flights with randomized weather scenarios for testing and exploration
   - **CSV Data Analysis**: Load and analyze actual flight data from the `exported_df.csv` dataset

3. **Visual Risk Indicators**: 
   - Gauge charts with color-coded risk zones (green/yellow/red)
   - Status badges indicating delay likelihood
   - Intuitive traffic-light system for quick decision-making

4. **Factor Breakdown**: Displays contributing factors that increase or decrease delay risk, helping users understand the underlying causes.

5. **Debug Mode**: Raw feature values for data transparency and model validation.

### Business Value

- **Travelers**: Make informed decisions about flight bookings and departure times
- **Airlines**: Optimize scheduling and preemptively manage expected delays
- **Operations Teams**: Plan staffing and resource allocation based on predicted weather impacts

## How the Dashboard Was Built

### Technology Stack

- **Framework**: Streamlit (Python web framework)
- **Visualization**: Plotly (interactive charts and gauges)
- **Data Processing**: Pandas
- **Styling**: Custom CSS with Southwest Airlines branding

### Architecture

The dashboard follows a **multi-page state-based architecture**:

1. **Landing Page**: User input selection (search flights or load CSV data)
2. **Selection Page**: Choose from available flights (mock data only)
3. **Results Page**: Display risk analysis, visualizations, and contributing factors

### Risk Scoring Algorithm

The heuristic model calculates delay risk by aggregating weighted factors:

| Factor | Risk Increase | Condition |
|--------|---------------|-----------|
| High Winds | +30 | > 40 km/h |
| Moderate Winds | +15 | 25-40 km/h |
| Heavy Precipitation | +35 | > 15 mm |
| Light Precipitation | +10 | > 0 mm |
| Snowfall | +40 | > 0 mm |
| Low Pressure | +15 | < 1005 hPa |
| Late Departure | +5 | After 6 PM |
| Long Haul Flight | +5 | > 2000 miles |

**Final Score**: Clamped between 0-100 and interpreted as:
- **0-10**: ✅ Very Low Risk
- **10-40**: ⚠️ Moderate Risk
- **40-100**: 🚨 High Risk of Delay

### Data Flow

```
User Input → Flight Schedule Generation → Weather Data Retrieval → Risk Calculation → Visualization
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)

### Local Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/flight-delay-predictor.git
   cd flight-delay-predictor
   ```

2. **Create Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Your Data** (optional)
   - Place `exported_df.csv` in the project root directory
   - If not present, the app will show a warning but still work with mock data

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

6. **Access the Dashboard**
   - Open your browser and go to: `http://localhost:8501`

## Cloud Deployment (Streamlit Cloud)

### Quick Deploy

1. **Prepare Repository**
   ```bash
   git add .
   git commit -m "Initial dashboard commit"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Connect your GitHub account
   - Select your repository and `app.py` file
   - Click "Deploy"

3. **Access Your App**
   - Your dashboard will be available at: `https://yourusername-flight-delay-predictor.streamlit.app`

### Requirements File

Ensure your `requirements.txt` includes:
```
streamlit==1.40.0
plotly==5.18.0
pandas==2.1.3
```

## File Structure

```
flight-delay-predictor/
├── app.py                    # Main Streamlit application
├── exported_df.csv           # Test data (optional)
├── requirements.txt          # Python dependencies
├── dashboard_overview.md     # This file
└── .gitignore               # Git ignore file (excludes __pycache__, etc.)
```

## Usage Guide

### Scenario 1: Search Mock Flight

1. Click the **"Search Mock Flight"** tab
2. Enter a flight number (e.g., `WN1492`)
3. Click **Search** to generate 3 upcoming flights
4. Select a flight to view risk analysis
5. Review the risk score and contributing factors

### Scenario 2: Analyze CSV Data

1. Click the **"Load Test Data"** tab
2. Select a record from your `exported_df.csv` file
3. Click **Analyze** to calculate risk and compare with actual data
4. Use the debug section to inspect raw weather values

### Understanding the Results

- **Weather Delay Risk Score**: 0-100 scale indicating delay probability
- **Status Badge**: Quick visual indicator (Green/Yellow/Red)
- **Contributing Factors**: Lists specific weather/flight conditions affecting the score
- **Actual CSV Score**: (when using CSV data) Shows the original model's prediction for comparison

## Configuration & Customization

### Adjust Risk Weights

Edit the `calculate_risk_score()` function in `app.py` to modify how each factor contributes to the final score:

```python
if weather['wspd'] > 40: 
    risk += 30  # Modify this value to adjust wind impact
```

### Change Styling

Update the `SOUTHWEST_CSS` string to customize:
- Colors (currently using Southwest Airlines branding)
- Font sizes and typography
- Button and card styling

### Modify Risk Thresholds

Change the status thresholds in the result section:
```python
if risk_score <= 10:        # Very Low Risk threshold
    status_color = "#4CAF50"
```

## Troubleshooting

### Issue: CSV file not loading
- **Solution**: Ensure `exported_df.csv` is in the project root and committed to GitHub (if deploying)
- Check the file path in `CSV_FILE_PATH` variable

### Issue: Dashboard loads slowly
- **Solution**: Streamlit Cloud may need time to install dependencies on first run
- Clear browser cache and refresh after a few moments

### Issue: Charts not displaying
- **Solution**: Verify Plotly is installed: `pip install plotly`
- Check browser console for JavaScript errors

### Issue: Mock data not generating
- **Solution**: Ensure random module is available (part of Python standard library)
- Restart the app with `streamlit run app.py`

## Support & Contribution

For issues, feature requests, or contributions:
1. Open an issue on GitHub
2. Submit a pull request with improvements
3. Contact the development team

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Last Updated**: November 2025
**Version**: 1.0
**Status**: Production Ready