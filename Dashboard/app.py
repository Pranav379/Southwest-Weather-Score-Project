import streamlit as st
import datetime
import os
import pickle

# configure encoders
with open('label_encoders.pkl', 'rb') as file:
    data = pickle.load(file)


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
# FILTER: Only keep flights with Risk > 0
# ==========================================
if TEST_DATA_DF is not None and 'weatherScore' in TEST_DATA_DF.columns:
    TEST_DATA_DF = TEST_DATA_DF[TEST_DATA_DF['weatherScore'] > 0]

# ==========================================
# 3. SOUTHWEST STYLING (CSS) - LEGIBILITY IMPROVED (DARK MODE ELEMENTS)
# ==========================================
SOUTHWEST_CSS = """
<style>
    /* 1. Global Reset */
    * {
        box-sizing: border-box;
    }

    /* 2. Main App Background & Text */
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa !important;
        color: #333333 !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* 3. WIDGET LABELS ( The Fix ) */
    /* Target the container and every possible child element */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] > div,
    [data-testid="stWidgetLabel"] > label,
    [data-testid="stWidgetLabel"] p {
        color: #304CB2 !important; /* Southwest Blue */
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* 4. Selectbox Input Text (The text inside the box) */
    div[data-baseweb="select"] > div {
        color: #333333 !important; /* Dark grey text */
        background-color: #ffffff !important; /* White background */
    }

    /* 5. Headings */
    h1, h2, h3 {
        color: #304CB2 !important;
        font-family: 'Arial', sans-serif;
        font-weight: 800;
    }

    /* 6. Score Box Styling */
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
        color: #ffffff !important; /* Force white text inside blue box */
    }
    .big-score {
        font-size: 3.5rem;
        font-weight: 900;
        color: #FFB612 !important;
        line-height: 1;
        margin: 10px 0;
    }

    /* 7. Buttons */
    button {
        background-color: #304CB2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    button:hover {
        background-color: #1e327a !important;
    }
    
    /* 8. Expander/Details Fixes */
    summary {
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
    
    # --- LIMIT TO 10 ITEMS ---
        if len(flight_numbers) >= 10:
            break
    
    selected_flight_num = st.selectbox(
        "📊 Select a flight number:",
        options=flight_numbers,
        key="flight_select"
    )
    
# Step 2: Get all routes for this flight number
    matching_rows = []
    for index, row in TEST_DATA_DF.iterrows():
        # 1. Identify Flight Number
        row_flight_num = row.get('Flight_Number_Reporting_Airline', 'N/A')
        if row_flight_num != 'N/A':
            try:
                # Normalize flight number to match selection (e.g. 2606.0 -> WN2606)
                f_str = f"WN{int(float(row_flight_num))}"
            except:
                f_str = f"WN{row_flight_num}"
        
        # 2. If this row matches the selected flight...
        if f_str == selected_flight_num:
            raw_origin = row.get('Origin', 'N/A')
            raw_dest = row.get('Dest', 'N/A')
            
            # 3. Try to decode names, but have a fallback!
            try:
                # Attempt to look up the real names (e.g., "DAL", "HOU")
                origin_idx = int(float(raw_origin))
                dest_idx = int(float(raw_dest))
                
                origin_name = data['Origin'].classes_[origin_idx]
                dest_name = data['Origin'].classes_[dest_idx]
                
                route_label = f"{origin_name} → {dest_name}"
                
            except Exception:
                # --- THE FIX: Don't skip! Use raw values if lookup fails ---
                # This ensures the dropdown is NEVER empty if data exists
                route_label = f"Loc {raw_origin} → Loc {raw_dest}"

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
        selected_row = TEST_DATA_DF.loc[selected_index]
        
        # Format the data
        flight_num = str(selected_row.get('Flight_Number_Reporting_Airline', 'N/A'))
        if flight_num != 'N/A':
            try:
                flight_num = f"WN{int(float(flight_num))}"
            except:
                flight_num = f"WN{flight_num}"

        # --- NEW DATE FORMATTING LOGIC ---
        try:
            # 1. Try to grab Month, Day, and Year
            # Use .get() to be safe. Default Year to 2024 if missing.
            mm = int(float(selected_row.get('Month', 0)))
            dd = int(float(selected_row.get('DayofMonth', 0)))
            yy = int(float(selected_row.get('Year', 2024)))
            
            # 2. Convert to "September 10, 2024" format
            date_str = datetime.date(yy, mm, dd).strftime("%B %d, %Y")
        except Exception:
            # Fallback: If 'Month' is missing, stick to the old Quarter format
            date_str = f"Q{selected_row.get('Quarter', 'N/A')} Day {selected_row.get('DayofMonth', 'N/A')}"   
        # -----------------------------------

        flight_data = {
            "id": selected_index,
            "source": "CSV",
            "flight_num": flight_num,
            "date": date_str,  # <--- Uses the new formatted string
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
    
    # Status and gauge
    col_gauge, col_status = st.columns([1, 1])
    
    if risk_score <= 10:
        status_color = "#4CAF50"  # Green
        status_title = "✅ Very Low Risk"
        status_msg = "Excellent conditions. Expect on-time departure."
    elif risk_score <= 30:
        status_color = "#8BC34A"  # Light Green
        status_title = "🟢 Low Risk"
        status_msg = "Good conditions, though minor weather factors are present."
    elif risk_score <= 60:
        status_color = "#FFB612"  # Yellow/Orange
        status_title = "⚠️ Moderate Risk"
        status_msg = "Weather factors present. Potential for minor delays."
    elif risk_score <= 80:
        status_color = "#FF5722"  # Orange/Red
        status_title = "🚨 High Risk"
        status_msg = "Poor weather conditions. Delays are likely."
    else:
        status_color = "#C60C30"  # Deep Red
        status_title = "⛔ Very High Risk"
        status_msg = "Severe weather. Significant delays or cancellations expected."
    
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
                        {'range': [0, 10], 'color': '#e8f5e9'},   # Very Low (Pale Green)
                        {'range': [10, 30], 'color': '#f1f8e9'},  # Low (Very Pale Green)
                        {'range': [30, 60], 'color': '#fff8e1'},  # Moderate (Pale Yellow)
                        {'range': [60, 80], 'color': '#fbe9e7'},  # High (Pale Orange)
                        {'range': [80, 100], 'color': '#ffebee'}  # Very High (Pale Red)
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
        # Note: Content inside expanders remains dark text on white background
        with st.expander("📈 Factors INCREASING Risk", expanded=True):
            risks = []
            
            # LOGIC remains in Metric (to match data), DISPLAY converts to Imperial
            if weather['wspd'] > 25:
                wspd_mph = weather['wspd'] * 0.621371
                risks.append(f"• High Winds ({wspd_mph:.1f} mph)")
            
            if weather['pres'] < 1005:
                pres_in = weather['pres'] * 0.02953
                risks.append(f"• Low Pressure ({pres_in:.1f} inHg)")
            
            if weather['prcp'] > 0:
                prcp_in = weather['prcp'] * 0.03937
                risks.append(f"• Precipitation ({prcp_in:.1f} in)")
            
            if weather['snow'] > 0:
                snow_in = weather['snow'] * 0.03937
                risks.append(f"• Snowfall ({snow_in:.1f} in)")
            
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
            # Logic check remains in Celsius (15-30C is roughly 59-86F)
            if 15 < weather['tavg'] < 30:
                # Convert to Fahrenheit for display
                temp_f = (weather['tavg'] * 9/5) + 32
                goods.append(f"• Mild Temps ({temp_f:.0f}°F)")
            
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
    
# --- FINAL SECTION: DETAILS & WEATHER SIDE-BY-SIDE ---
    st.markdown("---")
    
    # Create two equal columns
    c_details, c_weather = st.columns(2)
    
    # --- LEFT COLUMN: FLIGHT DETAILS ---
    with c_details:
        st.markdown("### ✈️ Flight Details")
        if HAS_PANDAS:
            dep_time_str = f"{int(flight['dep_time']):04d}"
            formatted_dep_time = f"{dep_time_str[:2]}:{dep_time_str[2:]}"
            
            origin_name = data['Origin'].classes_[int(float(flight['origin']))]
            dest_name = data['Origin'].classes_[int(float(flight['dest']))]
            distance_val = f"{int(float(flight['distance']))}"
            
            # Flight Details HTML Table
            table_html = f"""
            <style>
                .details-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .details-table td {{
                    padding: 8px 5px;
                    border-bottom: 1px solid #e0e0e0;
                    font-size: 0.9rem;
                }}
                .details-label {{
                    font-weight: 700;
                    color: #304CB2;
                    width: 40%;
                }}
                .details-value {{
                    color: #333333;
                    text-align: right;
                }}
            </style>
            
            <table class="details-table">
                <tr><td class="details-label">Flight Number</td><td class="details-value">{flight['flight_num']}</td></tr>
                <tr><td class="details-label">Origin</td><td class="details-value">{origin_name}</td></tr>
                <tr><td class="details-label">Destination</td><td class="details-value">{dest_name}</td></tr>
                <tr><td class="details-label">Distance</td><td class="details-value">{distance_val} miles</td></tr>
                <tr><td class="details-label">Departure Time</td><td class="details-value">{formatted_dep_time}</td></tr>
                <tr><td class="details-label">Date</td><td class="details-value">{flight['date']}</td></tr>
            </table>
            """
            st.markdown(table_html, unsafe_allow_html=True)

    # --- RIGHT COLUMN: WEATHER DATA ---
    with c_weather:
        st.markdown("### ☁️ Weather Data")
        
        # Conversions: Imperial Units
        temp_f = (weather['tavg'] * 9/5) + 32          # Whole number
        prcp_in = weather['prcp'] * 0.03937            # 1 decimal
        snow_in = weather['snow'] * 0.03937            # 1 decimal
        wspd_mph = weather['wspd'] * 0.621371          # 1 decimal
        pres_in = weather['pres'] * 0.02953            # 1 decimal
        
        # Weather Data HTML Table (Reuses styles from above)
        weather_html = f"""
        <table class="details-table">
            <tr><td class="details-label">Temperature</td><td class="details-value">{temp_f:.0f} °F</td></tr>
            <tr><td class="details-label">Precipitation</td><td class="details-value">{prcp_in:.1f} in</td></tr>
            <tr><td class="details-label">Snow</td><td class="details-value">{snow_in:.1f} in</td></tr>
            <tr><td class="details-label">Wind Speed</td><td class="details-value">{wspd_mph:.1f} mph</td></tr>
            <tr><td class="details-label">Pressure</td><td class="details-value">{pres_in:.1f} inHg</td></tr>
        </table>
        """
        st.markdown(weather_html, unsafe_allow_html=True)