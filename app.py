import streamlit as st
import datetime
import os

# ==========================================
# 1. APP CONFIGURATION & SAFE IMPORTS
# ==========================================
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

# Safe Imports
try:
    import plotly.graph_objects as go
    import pandas as pd
    HAS_PLOTTING = True
    HAS_PANDAS = True
except ImportError:
    HAS_PLOTTING = False
    HAS_PANDAS = False

# ==========================================
# 2. DATA LOADING
# ==========================================
# Set the CSV file path - assumes it's in the same directory as app.py
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(script_dir, 'exported_df.csv')

@st.cache_data
def load_data(file_path):
    """Safely loads the CSV data if available - only first 5000 rows for speed."""
    if not HAS_PANDAS:
        st.error("Pandas library is required to load CSV data.")
        return None
    try:
        # Load only first 5000 rows to speed up loading
        df = pd.read_csv(file_path, nrows=50000)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

TEST_DATA_DF = load_data(CSV_FILE_PATH)

# ==========================================
# 3. SOUTHWEST STYLING (CSS) - LEGIBILITY IMPROVED (DARK MODE ELEMENTS)
# ==========================================
SOUTHWEST_CSS = """
<style>
    * {
        box-sizing: border-box;
    }

    /* ===== App Background & Layout ===== */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* ===== Typography ===== */
    h1, h2, h3 {
        color: #304CB2 !important;  /* Southwest Blue */
        font-family: 'Arial', sans-serif;
        font-weight: 800;
    }

    /* ===== Score Box ===== */
    .score-container {
        background: linear-gradient(145deg, #304CB2, #1e327a);
        color: #ffffff;
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(48, 76, 178, 0.2);
    }

    .score-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.85;
        margin-bottom: 15px;
        font-weight: 600;
    }

    .big-score {
        font-size: 3.5rem;
        font-weight: 900;
        color: #FFB612;  /* Southwest Yellow */
        line-height: 1;
        margin: 10px 0;
    }

    /* ===== Divider ===== */
    hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 30px 0;
    }

    /* ===== Expander Bodies (inner content) ===== */
    /* Keep content dark-on-light for readability */
    .stExpander > div > div {
        color: #333333 !important;
        background-color: #ffffff !important;
    }

    /* ===== Buttons ===== */
    button {
        background-color: #304CB2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    button:hover {
        background-color: #1e327a !important;
    }

    /* ===== DataFrame Styling ===== */
    /* Header */
    div.stDataFrame table thead {
        background-color: #304CB2 !important;  /* Southwest Blue */
    }

    div.stDataFrame table thead th {
        color: #FFB612 !important;            /* Southwest Yellow */
        font-weight: 700 !important;
    }

    /* Body rows */
    div.stDataFrame table tbody tr {
        background-color: #333333 !important;  /* Dark background */
    }

    div.stDataFrame table tbody td {
        color: #ffffff !important;             /* White text */
    }

    /* =========================================================
       EXPANDER HEADERS (title + icon) - handle old & new DOM
       ========================================================= */

    /* Old layout: header in first child div */
    div[data-testid="stExpander"] > div:first-child {
        background-color: #1e327a !important;
        border-radius: 8px !important;
    }

    div[data-testid="stExpander"] > div:first-child * {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 600 !important;
    }

    /* New layout: <details data-testid="stExpander"><summary>...</summary> */
    details[data-testid="stExpander"] > summary {
        background-color: #1e327a !important;
        border-radius: 8px !important;
        list-style: none;  /* hide default marker if any */
    }

    details[data-testid="stExpander"] > summary * {
        color: #ffffff !important;
        fill: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Ensure the summary itself (text node) is also white */
    details[data-testid="stExpander"] > summary {
        color: #ffffff !important;
    }
</style>
"""

st.markdown(SOUTHWEST_CSS, unsafe_allow_html=True)

st.markdown(SOUTHWEST_CSS, unsafe_allow_html=True)

# --- FORCE expander headers to be white-on-blue, no matter what ---
NUKE_EXPANDER_CSS = """
<style>
/* Target ALL expander headers via <summary> tag */
summary {
    background-color: #1e327a !important;  /* dark blue bar */
    color: #ffffff !important;             /* white text */
    border-radius: 8px !important;
}

/* Make any child nodes (spans, emojis, icons, etc.) white too */
summary * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* In case Streamlit wraps the header in a div inside summary */
summary div {
    color: #ffffff !important;
}
</style>
"""
st.markdown(NUKE_EXPANDER_CSS, unsafe_allow_html=True)


# ==========================================
# 4. LOGIC & HEURISTIC ENGINE
# ==========================================
def calculate_risk_score(weather, flight_data):
    """Calculates the 'Weather Delay Risk' Score (0-100)."""
    risk = 0.0
    
    if weather['wspd'] > 40:
        risk += 30
    elif weather['wspd'] > 25:
        risk += 15
    
    if weather['prcp'] > 15:
        risk += 35
    elif weather['prcp'] > 0:
        risk += 10
    
    if weather['snow'] > 0:
        risk += 40
    
    if weather['pres'] < 1005:
        risk += 15
    
    if flight_data['dep_time'] > 1800:
        risk += 5
    
    if flight_data['distance'] > 2000:
        risk += 5
    
    return max(0.0, min(100.0, risk))

# ==========================================
# 5. USER INTERFACE FLOW
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'selected_flight' not in st.session_state:
    st.session_state.selected_flight = None

st.title("✈️ Flight Delay Predictor")

# Check if CSV data is available
if TEST_DATA_DF is None:
    st.error("❌ CSV file not found!")
    st.info(f"Looking for: {CSV_FILE_PATH}")
    st.stop()

# --- PAGE 1: LANDING ---
if st.session_state.page == 'landing':
    st.markdown("### 🕒 Analyze Flight Delay Risk")
    st.markdown("Select a flight record from your dataset to get a weather delay prediction.")
    
    st.markdown("---")
    
    # Step 1: Get unique flight numbers
    flight_numbers = []
    for index, row in TEST_DATA_DF.iterrows():
        flight_num = row.get('Flight_Number_Reporting_Airline', 'N/A')
        if flight_num != 'N/A':
            try:
                flight_num = f"WN{int(float(flight_num))}"
            except:
                flight_num = f"WN{flight_num}"
        if flight_num not in flight_numbers:
            flight_numbers.append(flight_num)
    
    selected_flight_num = st.selectbox(
        "📊 Select a flight number:",
        options=flight_numbers,
        key="flight_select"
    )
    
    # Step 2: Get all routes for this flight number
    matching_rows = []
    for index, row in TEST_DATA_DF.iterrows():
        flight_num = row.get('Flight_Number_Reporting_Airline', 'N/A')
        if flight_num != 'N/A':
            try:
                flight_num = f"WN{int(float(flight_num))}"
            except:
                flight_num = f"WN{flight_num}"
        if flight_num == selected_flight_num:
            origin = row.get('Origin', 'N/A')
            dest = row.get('Dest', 'N/A')
            route_label = f"{origin} → {dest}"
            matching_rows.append({
                'label': route_label,
                'index': index
            })
    
    # Remove duplicates but keep index
    unique_routes = []
    seen = set()
    for item in matching_rows:
        if item['label'] not in seen:
            unique_routes.append(item)
            seen.add(item['label'])
    
    route_options = [r['label'] for r in unique_routes]
    
    selected_route = st.selectbox(
        "📍 Select a route:",
        options=route_options,
        key="route_select"
    )
    
    if st.button("Analyze", key="analyze_btn", use_container_width=True):
        # Find the index for this route
        selected_route_obj = next(r for r in unique_routes if r['label'] == selected_route)
        selected_index = selected_route_obj['index']
        selected_row = TEST_DATA_DF.iloc[selected_index]
        
        # Format the data
        flight_num = str(selected_row.get('Flight_Number_Reporting_Airline', 'N/A'))
        if flight_num != 'N/A':
            try:
                flight_num = f"WN{int(float(flight_num))}"
            except:
                flight_num = f"WN{flight_num}"
        
        flight_data = {
            "id": selected_index,
            "source": "CSV",
            "flight_num": flight_num,
            "date": f"Q{selected_row.get('Quarter', 'N/A')} Day {selected_row.get('DayofMonth', 'N/A')}",
            "origin": str(selected_row.get('Origin', 'N/A')),
            "dest": str(selected_row.get('Dest', 'N/A')),
            "distance": float(selected_row.get('Distance', 0)),
            "dep_time": int(selected_row.get('CRSDepTime', 0)),
            "weather_raw": {
                'tavg': float(selected_row.get('tavg', 0)),
                'prcp': float(selected_row.get('prcp', 0)),
                'snow': float(selected_row.get('snow', 0)),
                'wspd': float(selected_row.get('wspd', 0)),
                'pres': float(selected_row.get('pres', 0)),
            },
            "true_weather_score": float(selected_row.get('weatherScore', 0))
        }
        st.session_state.selected_flight = flight_data
        st.session_state.page = 'result'
        st.rerun()

# --- PAGE 2: RESULT ---
elif st.session_state.page == 'result':
    flight = st.session_state.selected_flight
    
    c1, c2 = st.columns([1, 4])
    if c1.button("← Back"):
        st.session_state.page = 'landing'
        st.session_state.selected_flight = None
        st.rerun()
    
    # Data Source Badge
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Analysis for {flight['flight_num']}")
    with col_badge:
        st.success("CSV Data")
    
    weather = flight['weather_raw']
    # Use the actual CSV weather score instead of calculating
    risk_score = flight.get('true_weather_score', 0)
    
    # Display score card
    st.markdown(f"""
    <div class="score-container">
        <div class="score-label">Weather Delay Risk (0=Best, 100=Worst)</div>
        <div class="big-score">{risk_score:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Source: CSV Dataset")
    
    # Status and gauge
    col_gauge, col_status = st.columns([1, 1])
    
    if risk_score <= 10:
        status_color = "#4CAF50"
        status_title = "✅ Very Low Risk"
        status_msg = "Excellent conditions. Expect on-time departure."
    elif risk_score < 40:
        status_color = "#FFB612"
        status_title = "⚠️ Moderate Risk"
        status_msg = "Some weather factors present, but low to moderate delay risk."
    else:
        status_color = "#C60C30"
        status_title = "🚨 High Risk"
        status_msg = "Significant poor weather. Delays likely."
    
    with col_gauge:
        if HAS_PLOTTING:
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': status_color},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 10], 'color': '#e8f5e9'},
                        {'range': [10, 40], 'color': '#fff8e1'},
                        {'range': [40, 100], 'color': '#ffebee'}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col_status:
        st.markdown(f"### {status_title}")
        st.write(status_msg)
    
    st.markdown("---")
    st.markdown("### Contributing Factors")
    
    col_inc, col_dec = st.columns(2)
    
    with col_inc:
        # Note: Content inside expanders remains dark text on white background for this section,
        # as a fully dark mode requires more extensive CSS changes than requested.
        with st.expander("📈 Factors INCREASING Risk", expanded=True):
            risks = []
            if weather['wspd'] > 25:
                risks.append(f"• High Winds ({weather['wspd']:.1f} km/h)")
            if weather['pres'] < 1005:
                risks.append(f"• Low Pressure ({weather['pres']:.1f} hPa)")
            if weather['prcp'] > 0:
                risks.append(f"• Precipitation ({weather['prcp']:.1f} mm)")
            if weather['snow'] > 0:
                risks.append(f"• Snowfall ({weather['snow']:.1f} mm)")
            if flight['distance'] > 2000:
                risks.append("• Long Haul Flight")
            if flight['dep_time'] > 1800:
                risks.append("• Late Evening Departure")
            
            if risks:
                st.markdown("\n".join(risks))
            else:
                st.write("No major risk factors.")
    
    with col_dec:
        with st.expander("📉 Factors DECREASING Risk", expanded=True):
            goods = []
            if 15 < weather['tavg'] < 30:
                goods.append(f"• Mild Temps ({weather['tavg']:.1f}°C)")
            if weather['wspd'] < 15:
                goods.append("• Calm Winds")
            if weather['pres'] >= 1015:
                goods.append("• High Pressure System")
            if weather['prcp'] == 0:
                goods.append("• No Precipitation")
            
            if goods:
                st.markdown("\n".join(goods))
            else:
                st.write("Standard conditions.")
    
    # Flight Details Card
    st.markdown("---")
    with st.expander("✈️ Flight Details", expanded=False):
        if HAS_PANDAS:
            dep_time_str = f"{int(flight['dep_time']):04d}"
            formatted_dep_time = f"{dep_time_str[:2]}:{dep_time_str[2:]}"
            details_data = {
                'Property': ['Flight Number', 'Origin', 'Destination', 'Distance (miles)', 'Departure Time', 'Date', 'Record Index'],
                'Value': [
                    flight['flight_num'],
                    flight['origin'],
                    flight['dest'],
                    f"{int(flight['distance'])}",
                    formatted_dep_time,
                    flight['date'],
                    flight['id']
                ]
            }
            st.dataframe(pd.DataFrame(details_data), use_container_width=True, hide_index=True)
    
    # Debug section
    if st.checkbox("View Raw Weather Data (Debug)"):
        if HAS_PANDAS:
            debug_data = {
                'Feature': ['Temperature (°C)', 'Precipitation (mm)', 'Snow (mm)', 'Wind Speed (km/h)', 'Pressure (hPa)'],
                'Value': [f"{weather['tavg']:.1f}", f"{weather['prcp']:.1f}", f"{weather['snow']:.1f}", f"{weather['wspd']:.1f}", f"{weather['pres']:.1f}"]
            }
            st.dataframe(pd.DataFrame(debug_data), use_container_width=True)