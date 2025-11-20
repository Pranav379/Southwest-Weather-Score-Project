import streamlit as st
import datetime
import random
import time

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
CSV_FILE_PATH = 'exported_df.csv'

@st.cache_data
def load_data(file_path):
    """Safely loads the CSV data if available."""
    if not HAS_PANDAS:
        st.error("Pandas library is required to load CSV data.")
        return None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

TEST_DATA_DF = load_data(CSV_FILE_PATH)

# ==========================================
# 3. SOUTHWEST STYLING (CSS)
# ==========================================
SOUTHWEST_CSS = """
<style>
    * { box-sizing: border-box; }
    
    /* Main Background */
    [data-testid="stAppViewContainer"] { 
        background-color: #f8f9fa !important; 
        color: #333333 !important; 
    }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    
    /* Typography */
    h1, h2, h3 { 
        color: #304CB2 !important; 
        font-family: 'Arial', sans-serif; 
        font-weight: 800; 
    }
    
    /* Score Box Container */
    .score-container {
        background: linear-gradient(145deg, #304CB2, #1e327a);
        color: white;
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
        color: #FFB612;
        line-height: 1;
        margin: 10px 0;
    }
    
    /* Flight Card */
    .flight-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .flight-card:hover {
        border-color: #304CB2;
        box-shadow: 0 4px 12px rgba(48, 76, 178, 0.1);
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 30px 0;
    }
    
    /* Status Messages */
    .status-low { color: #4CAF50; font-weight: bold; }
    .status-med { color: #FFB612; font-weight: bold; }
    .status-high { color: #C60C30; font-weight: bold; }
    
    /* Expander Styling */
    .stExpander { 
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
    }
    
    /* Table styling */
    [data-testid="stDataFrame"] {
        width: 100% !important;
    }
</style>
"""
st.markdown(SOUTHWEST_CSS, unsafe_allow_html=True)

# ==========================================
# 4. LOGIC & HEURISTIC ENGINE
# ==========================================
def generate_mock_schedule(flight_num):
    """Simulates looking up a schedule based on input."""
    today = datetime.date.today()
    routes = [
        ("DAL", "HOU", 239), ("MDW", "LGA", 733), ("PHX", "LAX", 370),
        ("DEN", "SFO", 967), ("BWI", "MCO", 787), ("ATL", "DAL", 721)
    ]
    schedule = []
    for i in range(1, 4):
        r_origin, r_dest, r_dist = random.choice(routes)
        date_offset = today + datetime.timedelta(days=i)
        dep_h = random.randint(6, 21)
        dep_m = random.choice([0, 15, 30, 45])
        dep_time = dep_h * 100 + dep_m
        schedule.append({
            "id": i,
            "source": "Mock",
            "flight_num": flight_num.upper(),
            "date": date_offset,
            "origin": r_origin,
            "dest": r_dest,
            "distance": r_dist,
            "dep_time": dep_time,
            "weather": None
        })
    return schedule

def get_mock_weather_data():
    """Generates random weather features."""
    return {
        'tavg': random.uniform(-5, 35),
        'prcp': random.choices([0, 5.0, 25.0], weights=[0.6, 0.3, 0.1])[0],
        'snow': random.choices([0, 10.0], weights=[0.9, 0.1])[0],
        'wspd': random.uniform(5, 45),
        'pres': random.uniform(995, 1030)
    }

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

# --- PAGE 1: LANDING ---
if st.session_state.page == 'landing':
    st.markdown("### 🕒 Will you make it on time?")
    
    tab1, tab2 = st.tabs(["Search Mock Flight", "Load Test Data"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            f_num = st.text_input("Flight Number", placeholder="e.g. WN1492", value="WN1492", key="mock_flight_input")
        with col2:
            st.write("")
            st.write("")
            if st.button("Search", key="search_btn", use_container_width=True):
                with st.spinner("Searching..."):
                    time.sleep(0.5)
                st.session_state.schedule = generate_mock_schedule(f_num)
                st.session_state.page = 'selection'
                st.rerun()
    
    with tab2:
        if TEST_DATA_DF is not None:
            data_options = []
            for index, row in TEST_DATA_DF.iterrows():
                option_label = (
                    f"Index {index} | Flight {row.get('Flight_Num', 'N/A')} "
                    f"| {row.get('Origin', 'N/A')}→{row.get('Dest', 'N/A')} "
                    f"| Score: {row.get('weatherScore', 'N/A')}"
                )
                data_options.append(option_label)
            
            selected_label = st.selectbox(
                "Select a record from CSV:",
                options=data_options,
                key="data_select"
            )
            
            if st.button("Analyze", key="analyze_btn", use_container_width=True):
                selected_index = int(selected_label.split(' |')[0].replace('Index ', ''))
                selected_row = TEST_DATA_DF.iloc[selected_index]
                flight_data = {
                    "id": selected_index,
                    "source": "CSV",
                    "flight_num": selected_row.get('Flight_Num', 'N/A'),
                    "date": f"Q{selected_row.get('Quarter', 'N/A')} Day {selected_row.get('DayofMonth', 'N/A')}",
                    "origin": selected_row.get('Origin', 'N/A'),
                    "dest": selected_row.get('Dest', 'N/A'),
                    "distance": selected_row.get('Distance', 0),
                    "dep_time": selected_row.get('CRSDepTime', 0),
                    "weather_raw": {
                        'tavg': selected_row.get('tavg', 0),
                        'prcp': selected_row.get('prcp', 0),
                        'snow': selected_row.get('snow', 0),
                        'wspd': selected_row.get('wspd', 0),
                        'pres': selected_row.get('pres', 0),
                    },
                    "true_weather_score": selected_row.get('weatherScore', 'N/A')
                }
                st.session_state.selected_flight = flight_data
                st.session_state.page = 'result'
                st.rerun()
        else:
            st.warning(f"Could not load data from '{CSV_FILE_PATH}'.")

# --- PAGE 2: SELECTION ---
elif st.session_state.page == 'selection':
    if st.button("← Back"):
        st.session_state.page = 'landing'
        st.rerun()
    
    st.subheader("Select Your Flight")
    for flight in st.session_state.schedule:
        d_time_str = f"{flight['dep_time']:04d}"
        formatted_time = f"{d_time_str[:2]}:{d_time_str[2:]}"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class="flight-card">
                <strong style="color:#304CB2; font-size:1.2em;">{flight['flight_num']}</strong>
                <div>{flight['origin']} ➝ {flight['dest']} ({flight['distance']} mi)</div>
                <div style="font-size:0.9em; color:#666;">📅 {flight['date']} | 🕐 {formatted_time}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("Select", key=f"select_{flight['id']}", use_container_width=True):
                st.session_state.selected_flight = flight
                st.session_state.page = 'result'
                st.rerun()

# --- PAGE 3: RESULT ---
elif st.session_state.page == 'result':
    flight = st.session_state.selected_flight
    
    c1, c2 = st.columns([1, 4])
    if c1.button("← Start Over"):
        st.session_state.page = 'landing'
        st.session_state.selected_flight = None
        st.rerun()
    
    st.markdown(f"## Analysis for {flight['flight_num']}")
    
    # Calculate risk
    if flight.get('source') == "CSV":
        weather = flight['weather_raw']
        risk_score = calculate_risk_score(weather, flight)
        source_label = "Source: CSV Data"
    else:
        weather = get_mock_weather_data()
        risk_score = calculate_risk_score(weather, flight)
        source_label = "Source: Mock Forecast"
    
    # Display score card
    st.markdown(f"""
    <div class="score-container">
        <div class="score-label">Weather Delay Risk (0=Best, 100=Worst)</div>
        <div class="big-score">{risk_score:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption(source_label)
    
    if flight.get('source') == "CSV":
        st.info(f"**Actual CSV Weather Score:** {flight.get('true_weather_score', 'N/A')}")
    
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
    
    # Debug section
    st.markdown("---")
    if st.checkbox("View Raw Data (Debug)"):
        if HAS_PANDAS:
            debug_data = {
                'Feature': ['Temperature (°C)', 'Precipitation (mm)', 'Snow (mm)', 'Wind Speed (km/h)', 'Pressure (hPa)'],
                'Value': [f"{weather['tavg']:.1f}", f"{weather['prcp']:.1f}", f"{weather['snow']:.1f}", f"{weather['wspd']:.1f}", f"{weather['pres']:.1f}"]
            }
            st.dataframe(pd.DataFrame(debug_data), use_container_width=True)