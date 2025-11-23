# import streamlit as st
# import datetime
# import os
# import pickle
# import airportsdata

# # configure encoders
# # Note: This assumes './Dashboard/label_encoders.pkl' exists relative to where the app is run.
# try:
#     with open('./Dashboard/label_encoders.pkl', 'rb') as file:
#         data = pickle.load(file)
# except FileNotFoundError:
#     st.error("Error: label_encoders.pkl not found. Please ensure the file is in the correct path.")
#     data = None # Set data to None if file not found
# except Exception as e:
#     st.error(f"Error loading label encoders: {e}")
#     data = None


# # ==========================================
# # 1. APP CONFIGURATION & SAFE IMPORTS
# # ==========================================
# st.set_page_config(
#     page_title="Flight Delay Predictor ✈️",
#     page_icon="✈️",
#     layout="centered"
# )

# # Safe Imports
# try:
#     import plotly.graph_objects as go
#     import pandas as pd
#     HAS_PLOTTING = True
#     HAS_PANDAS = True
# except ImportError:
#     HAS_PLOTTING = False
#     HAS_PANDAS = False
#     st.error("Pandas or Plotly library is missing. Install them to run the app.")


# # ==========================================
# # 2. DATA LOADING
# # ==========================================
# # Set the CSV file path - assumes it's in the same directory as app.py
# script_dir = os.path.dirname(os.path.abspath(__file__))
# CSV_FILE_PATH = os.path.join(script_dir, 'flight_data.csv.gz')

# @st.cache_data
# def load_data(file_path):
#     """Safely loads the CSV data if available - only first 5000 rows for speed."""
#     if not HAS_PANDAS:
#         return None
#     try:
#         # Load only first 5000 rows to speed up loading
#         df = pd.read_csv(file_path, nrows=5000, compression='gzip')
#         df.columns = df.columns.str.strip()
#         return df
#     except FileNotFoundError:
#         return None
#     except Exception as e:
#         st.error(f"Error loading data: {e}")
#         return None

# TEST_DATA_DF = load_data(CSV_FILE_PATH)
# # ==========================================
# # FILTER: Only keep flights with Risk > 0
# # ==========================================
# if TEST_DATA_DF is not None and 'weatherScore' in TEST_DATA_DF.columns:
#     TEST_DATA_DF = TEST_DATA_DF[TEST_DATA_DF['weatherScore'] > 0]
# elif TEST_DATA_DF is None:
#     st.error("Cannot load flight data. Stopping execution.")
#     st.stop()
# elif data is None:
#     st.error("Cannot load label encoders. Stopping execution.")
#     st.stop()


# # ==========================================
# # 3. SOUTHWEST STYLING (CSS) - THEME UPGRADE
# # ==========================================
# SOUTHWEST_CSS = """
# <style>
#     /* 1. Main Background */
#     [data-testid="stAppViewContainer"] {
#         background-color: #f4f7f6 !important; /* Very light grey-blue */
#         color: #333333 !important;
#     }
    
#     /* 2. Cards (The "Boarding Pass" Look) */
#     .stCard {
#         background-color: #ffffff;
#         padding: 25px;
#         border-radius: 15px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Soft shadow */
#         margin-bottom: 20px;
#         border-top: 5px solid #304CB2; /* Southwest Blue Header Line */
#     }
    
#     /* 3. Headers */
#     h1, h2, h3 {
#         color: #304CB2 !important;
#         font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
#         font-weight: 800;
#     }
    
#     /* 4. Southwest Striping (Decorative Line) */
#     .sw-stripe {
#         height: 6px;
#         width: 100%;
#         background: linear-gradient(90deg, #304CB2 33%, #C60C30 33%, #C60C30 66%, #FFB612 66%);
#         border-radius: 3px;
#         margin: 10px 0 25px 0;
#     }

#     /* 5. Score Box Styling */
#     .score-container {
#         background: linear-gradient(135deg, #304CB2, #1A2C75);
#         color: #ffffff;
#         padding: 30px;
#         border-radius: 20px;
#         text-align: center;
#         box-shadow: 0 8px 20px rgba(48, 76, 178, 0.3);
#         position: relative;
#         overflow: hidden;
#     }
#     .score-label {
#         font-size: 0.85rem;
#         text-transform: uppercase;
#         letter-spacing: 2px;
#         color: #FFB612 !important; 
#         font-weight: 700;
#     }
#     .big-score {
#         font-size: 4rem;
#         font-weight: 900;
#         color: #ffffff !important;
#         margin: 5px 0;
#     }

#     /* 6. Buttons (Rounder & Bolder) */
#     button {
#         background-color: #304CB2 !important;
#         color: white !important;
#         border-radius: 50px !important; /* Pill shape */
#         font-weight: 700 !important;
#         padding: 0.5rem 1rem !important;
#         border: none !important;
#         transition: all 0.3s ease !important;
#     }
#     button:hover {
#         background-color: #253b8c !important;
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(48, 76, 178, 0.3);
#     }
    
#     /* 7. Custom Tables */
#     .details-table td {
#         padding: 12px 5px;
#         border-bottom: 1px solid #f0f0f0;
#         color: #444;
#     }
#     .details-label {
#         font-weight: 700;
#         color: #304CB2;
#         text-transform: uppercase;
#         font-size: 0.75rem;
#         letter-spacing: 0.5px;
#     }
#     .details-value {
#         font-weight: 600;
#         font-size: 1rem;
#         color: #222;
#     }
#     /* ... (keep your existing CSS) ... */
    
#     /* 8. FIX: Force Input Labels to be Visible (Southwest Blue) */
#     .stSelectbox label p {
#         color: #304CB2 !important; /* Force text to Blue */
#         font-size: 1.1rem !important;
#         font-weight: 700 !important;
#     }
    
#     /* Optional: Fix the dropdown box itself to look cleaner */
#     div[data-baseweb="select"] > div {
#         background-color: #ffffff !important; /* White background */
#         border: 1px solid #304CB2 !important; /* Blue border */
#         color: #333 !important; /* Dark text inside */
#     }
# </style>
# """
# st.markdown(SOUTHWEST_CSS, unsafe_allow_html=True)

# # --- FORCE expander headers to be white-on-blue, no matter what ---
# NUKE_EXPANDER_CSS = """
# <style>
# /* Target ALL expander headers via <summary> tag */
# summary {
#     background-color: #1e327a !important;  /* dark blue bar */
#     color: #ffffff !important;             /* white text */
#     border-radius: 8px !important;
# }

# /* Make any child nodes (spans, emojis, icons, etc.) white too */
# summary * {
#     color: #ffffff !important;
#     fill: #ffffff !important;
# }

# /* In case Streamlit wraps the header in a div inside summary */
# summary div {
#     color: #ffffff !important;
# }
# </style>
# """
# st.markdown(NUKE_EXPANDER_CSS, unsafe_allow_html=True)


# # ==========================================
# # 4. LOGIC & HEURISTIC ENGINE
# # ==========================================
# def calculate_risk_score(weather, flight_data):
#     """Calculates the 'Weather Delay Risk' Score (0-100)."""
#     risk = 0.0
    
#     if weather['wspd'] > 40:
#         risk += 30
#     elif weather['wspd'] > 25:
#         risk += 15
    
#     if weather['prcp'] > 15:
#         risk += 35
#     elif weather['prcp'] > 0:
#         risk += 10
    
#     if weather['snow'] > 0:
#         risk += 40
    
#     if weather['pres'] < 1005:
#         risk += 15
    
#     if flight_data['dep_time'] > 1800:
#         risk += 5
    
#     if flight_data['distance'] > 2000:
#         risk += 5
    
#     return max(0.0, min(100.0, risk))

# # ==========================================
# # 5. USER INTERFACE FLOW
# # ==========================================
# if 'page' not in st.session_state:
#     st.session_state.page = 'landing'
# if 'selected_flight' not in st.session_state:
#     st.session_state.selected_flight = None

# # --- LOGO & TITLE SECTION ---
# col_logo, col_text = st.columns([2, 3])
# with col_logo:
#     # Use st.image for better size control in a column
#     st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Southwest_Airlines_logo_2014.svg/320px-Southwest_Airlines_logo_2014.svg.png", width=500)

# st.markdown('<div class="sw-stripe"></div>', unsafe_allow_html=True)

# # UPDATED: Added Plane Emoji ✈️
# st.title("Flight Delay Predictor ✈️")

# # Check if data is available before proceeding
# if TEST_DATA_DF is None:
#     st.error("❌ CSV file not found!")
#     st.info(f"Looking for: {CSV_FILE_PATH}")
#     st.stop()
# if data is None:
#     st.stop() # already handled error above


# # --- PAGE 1: LANDING ---
# if st.session_state.page == 'landing':
    
#     # FIX: Load airport data once for better dropdown display
#     airports = airportsdata.load('IATA') 

#     # UPDATED: Changed <h2> to <div> to bypass global Blue styling
#     st.markdown("""
#         <div style='
#             text-align: left; 
#             color: #000000; 
#             font-size: 20px; 
#             font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
#             font-weight: 800; 
#             margin-bottom: 50px; 
#         '>
#             Enter your flight number to get started!
#         </div>
#     """, unsafe_allow_html=True)
    
    
#     # Step 1: Get unique flight numbers
#     flight_numbers = []
#     for index, row in TEST_DATA_DF.iterrows():
#         flight_num = row.get('Flight_Number_Reporting_Airline', 'N/A')
#         if flight_num != 'N/A':
#             try:
#                 flight_num = f"WN{int(float(flight_num))}"
#             except:
#                 flight_num = f"WN{flight_num}"
#         if flight_num not in flight_numbers:
#             flight_numbers.append(flight_num)
    
#     # --- LIMIT TO 10 ITEMS ---
#         if len(flight_numbers) >= 10:
#             break
    
#     selected_flight_num = st.selectbox(
#         "📊 Select a flight number:",
#         options=flight_numbers,
#         key="flight_select"
#     )
    
# # Step 2: Get all routes for this flight number
#     matching_rows = []
#     for index, row in TEST_DATA_DF.iterrows():
#         # 1. Identify Flight Number
#         row_flight_num = row.get('Flight_Number_Reporting_Airline', 'N/A')
#         if row_flight_num != 'N/A':
#             try:
#                 # Normalize flight number to match selection (e.g. 2606.0 -> WN2606)
#                 f_str = f"WN{int(float(row_flight_num))}"
#             except:
#                 f_str = f"WN{row_flight_num}"
        
#         # 2. If this row matches the selected flight...
#         if f_str == selected_flight_num:
#             raw_origin = row.get('Origin', 'N/A')
#             raw_dest = row.get('Dest', 'N/A')
            
#             # 3. FIX: Try to decode names with full airport details
#             try:
#                 # Get IATA codes using the label encoder
#                 origin_idx = int(float(raw_origin))
#                 dest_idx = int(float(raw_dest))
                
#                 origin_iata = data['Origin'].classes_[origin_idx]
#                 dest_iata = data['Origin'].classes_[dest_idx]
                
#                 # Get full airport info
#                 originInfo = airports.get(origin_iata)
#                 destInfo = airports.get(dest_iata)
                
#                 # Build the user-friendly label
#                 origin_display = f"{originInfo.get('name', origin_iata)} ({origin_iata})" if originInfo else origin_iata
#                 dest_display = f"{destInfo.get('name', dest_iata)} ({dest_iata})" if destInfo else dest_iata
                
#                 route_label = f"{origin_display} → {dest_display}"
                
#             except Exception:
#                 # Fallback if label encoding or IATA lookup fails
#                 # --- The FIX: Don't skip! Use raw values if lookup fails ---
#                 route_label = f"{raw_origin} → {raw_dest}"

#             matching_rows.append({
#                 'label': route_label,
#                 'index': index
#             })
    
#     # Remove duplicates but keep index
#     unique_routes = []
#     seen = set()
#     for item in matching_rows:
#         if item['label'] not in seen:
#             unique_routes.append(item)
#             seen.add(item['label'])
    
#     route_options = [r['label'] for r in unique_routes]
    
#     selected_route = st.selectbox(
#         "📍 Select a route:",
#         options=route_options,
#         key="route_select"
#     )
    
#     if st.button("Analyze", key="analyze_btn", use_container_width=True):
#         # Find the index for this route
#         selected_route_obj = next(r for r in unique_routes if r['label'] == selected_route)
#         selected_index = selected_route_obj['index']
#         selected_row = TEST_DATA_DF.loc[selected_index]
        
#         # Format the data
#         flight_num = str(selected_row.get('Flight_Number_Reporting_Airline', 'N/A'))
#         if flight_num != 'N/A':
#             try:
#                 flight_num = f"WN{int(float(flight_num))}"
#             except:
#                 flight_num = f"WN{flight_num}"

#         # --- NEW DATE FORMATTING LOGIC ---
#         try:
#             # 1. Try to grab Month, Day, and Year
#             # Use .get() to be safe. Default Year to 2024 if missing.
#             mm = int(float(selected_row.get('Month', 0)))
#             dd = int(float(selected_row.get('DayofMonth', 0)))
#             yy = int(float(selected_row.get('Year', 2024)))
            
#             # 2. Convert to "September 10, 2024" format
#             date_str = datetime.date(yy, mm, dd).strftime("%B %d, %Y")
#         except Exception:
#             # Fallback: If 'Month' is missing, stick to the old Quarter format
#             date_str = f"Q{selected_row.get('Quarter', 'N/A')} Day {selected_row.get('DayofMonth', 'N/A')}"    
#         # -----------------------------------

#         def safe_float(value):
#             try:
#                 v = float(value)
#                 return 0 if pd.isna(v) else v
#             except:
#                 return 0

#         flight_data = {
#             "id": selected_index,
#             "source": "CSV",
#             "flight_num": flight_num,
#             "date": date_str,  # <--- Uses the new formatted string
#             "origin": str(selected_row.get('Origin', 'N/A')),
#             "dest": str(selected_row.get('Dest', 'N/A')),
#             "distance": safe_float(selected_row.get('Distance', 0)),
#             "dep_time": int(selected_row.get('CRSDepTime', 0)),
#             "weather_raw": {
#                 'tavg': safe_float(selected_row.get('tavg', 0)),
#                 'prcp': safe_float(selected_row.get('prcp', 0)),
#                 'snow': safe_float(selected_row.get('snow', 0)),
#                 'wspd': safe_float(selected_row.get('wspd', 0)),
#                 'pres': safe_float(selected_row.get('pres', 0)),
#             },
#             "true_weather_score": float(selected_row.get('weatherScore', 0))
#         }
#         st.session_state.selected_flight = flight_data
#         st.session_state.page = 'result'
#         st.rerun()

# # --- PAGE 2: RESULT ---
# elif st.session_state.page == 'result':
#     flight = st.session_state.selected_flight
    
#     c1, c2 = st.columns([1, 4])
#     if c1.button("← Back"):
#         st.session_state.page = 'landing'
#         st.session_state.selected_flight = None
#         st.rerun()

    
#     weather = flight['weather_raw']
#     # Use the actual CSV weather score instead of calculating
#     risk_score = flight.get('true_weather_score', 0)
    
#     # Display score card
#     st.markdown(f"""
#     <div class="score-container">
#         <div class="score-label">Weather Delay Risk (0=Best, 100=Worst)</div>
#         <div class="big-score">{risk_score:.1f}</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Status and gauge
#     col_gauge, col_status = st.columns([1, 1])
    
#     if risk_score <= 10:
#         status_color = "#4CAF50"  # Green
#         status_title = "✅ Very Low Risk"
#         status_msg = "Excellent conditions. Expect on-time departure."
#     elif risk_score <= 30:
#         status_color = "#8BC34A"  # Light Green
#         status_title = "🟢 Low Risk"
#         status_msg = "Good conditions, though minor weather factors are present."
#     elif risk_score <= 60:
#         status_color = "#FFB612"  # Yellow/Orange
#         status_title = "⚠️ Moderate Risk"
#         status_msg = "Weather/time of day factors present. Potential for minor delays."
#     elif risk_score <= 80:
#         status_color = "#FF5722"  # Orange/Red
#         status_title = "🚨 High Risk"
#         status_msg = "Delays are likely."
#     else:
#         status_color = "#C60C30"  # Deep Red
#         status_title = "⛔ Very High Risk"
#         status_msg = "Severe weather. Significant delays or cancellations expected."
    
#     with col_gauge:
#         if HAS_PLOTTING:
#             fig = go.Figure(go.Indicator(
#                 mode="gauge",
#                 value=risk_score,
#                 domain={'x': [0, 1], 'y': [0, 1]},
#                 gauge={
#                     'axis': {
#                         'range': [0, 100],
#                         'tickmode': 'array',
#                         # Added 25 and 75 to the values and labels
#                         'tickvals': [0, 25, 50, 75, 100],
#                         'ticktext': ['0', '25', '50', '75', '100'],
#                         'tickfont': {'size': 14, 'color': '#000000'},
#                     },
#                     'bar': {'color': status_color},
#                     'bgcolor': "white",
#                     'steps': [
#                         {'range': [0, 10], 'color': '#e8f5e9'},
#                         {'range': [10, 30], 'color': '#f1f8e9'},
#                         {'range': [30, 60], 'color': '#fff8e1'},
#                         {'range': [60, 80], 'color': '#fbe9e7'},
#                         {'range': [80, 100], 'color': '#ffebee'}
#                     ]
#                 }
#             ))
            
#             # Margins kept wide so numbers don't get cut off
#             fig.update_layout(
#                 height=250, 
#                 margin=dict(l=40, r=40, t=20, b=20), 
#                 paper_bgcolor="rgba(0,0,0,0)"
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
    
#     with col_status:
#         st.markdown(f"### {status_title}")
#         st.write(status_msg)
    
#     st.markdown("---")
#     st.markdown("### Contributing Factors")
    
#     col_inc, col_dec = st.columns(2)
    
#     with col_inc:
#         # Note: Content inside expanders remains dark text on white background
#         with st.expander("📈 Factors INCREASING Risk", expanded=True):
#             risks = []
            
#             # LOGIC remains in Metric, DISPLAY converts to Imperial
#             if weather['wspd'] > 25:
#                 wspd_mph = weather['wspd'] * 0.621371
#                 risks.append(f"• High Winds ({wspd_mph:.1f} mph)")
            
#             if weather['pres'] < 1005:
#                 pres_in = weather['pres'] * 0.02953
#                 risks.append(f"• Low Pressure ({pres_in:.1f} inHg)")
            
#             if weather['prcp'] > 0:
#                 prcp_in = weather['prcp'] * 0.03937
#                 risks.append(f"• Precipitation ({prcp_in:.1f} in)")
            
#             if weather['snow'] > 0:
#                 snow_in = weather['snow'] * 0.03937
#                 risks.append(f"• Snowfall ({snow_in:.1f} in)")
            
#             if flight['distance'] > 2000:
#                 risks.append("• Long Haul Flight")
#             if flight['dep_time'] > 1800:
#                 risks.append("• Late Evening Departure")
            
#             if risks:
#                 # Use double newlines to force separate lines
#                 st.markdown("\n\n".join(risks))
#             else:
#                 st.write("No major risk factors.")
    
#     with col_dec:
#         with st.expander("📉 Factors DECREASING Risk", expanded=True):
#             goods = []
#             # Logic check remains in Celsius
#             if 15 < weather['tavg'] < 30:
#                 # Convert to Fahrenheit
#                 temp_f = (weather['tavg'] * 9/5) + 32
#                 # Display as whole number
#                 goods.append(f"• Mild Temps ({temp_f:.0f}°F)")
            
#             if weather['wspd'] < 15:
#                 goods.append("• Calm Winds")
#             if weather['pres'] >= 1015:
#                 goods.append("• High Pressure System")
#             if weather['prcp'] == 0:
#                 goods.append("• No Precipitation")
            
#             if goods:
#                 # Use double newlines to force separate lines
#                 st.markdown("\n\n".join(goods))
#             else:
#                 st.write("Standard conditions.")
    
# # --- FINAL SECTION: CARDS UI ---
#     st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
#     # 1. PREPARE DATA (Do this before columns so both cards can use it)
#     if HAS_PANDAS:
#         # Decode Airport Names
#         origin_iata = flight['origin']
#         dest_iata = flight['dest']
        
#         # load airport data
#         airports = airportsdata.load('IATA') 
        
#         # --- FIX: Use .get() and provide safe fallback ---
#         originInfo = airports.get(origin_iata)
#         destInfo = airports.get(dest_iata)

#         # format origin and destination
#         if originInfo is None:
#             originDisplay = f"{origin_iata} (Info Missing)"
#         else:
#             originDisplay = originInfo['name'] + " (" + origin_iata + ")"
            
#         if destInfo is None:
#             destDisplay = f"{dest_iata} (Info Missing)"
#         else:
#             destDisplay = destInfo['name'] + " (" + dest_iata + ")"
#         # ------------------------------------------------

#         # Format Times & Distances
#         dep_time_str = f"{int(flight['dep_time']):04d}"
#         formatted_dep_time = f"{dep_time_str[:2]}:{dep_time_str[2:]}"
#         distance_val = f"{int(float(flight['distance']))}"
        
#         # Format Weather Values
#         temp_f = (weather['tavg'] * 9/5) + 32
#         prcp_in = weather['prcp'] * 0.03937
#         snow_in = weather['snow'] * 0.03937
#         wspd_mph = weather['wspd'] * 0.621371
#         pres_in = weather['pres'] * 0.02953

#     # 2. CREATE COLUMNS
#     c_details, c_weather = st.columns(2)
    
#     # --- LEFT CARD: FLIGHT DETAILS ---
#     with c_details:
#         # BUILD THE FLIGHT CARD HTML
#         flight_card_html = f"""
#         <div class="stCard">
#             <h3>✈️ Flight Details</h3>
#             <table class="details-table" style="width:100%">
#                 <tr><td class="details-label">Flight No.</td><td class="details-value">{flight['flight_num']}</td></tr>
#                 <tr><td class="details-label">Route</td><td class="details-value">{originDisplay} ➝ {destDisplay}</td></tr>
#                 <tr><td class="details-label">Distance</td><td class="details-value">{distance_val} mi</td></tr>
#                 <tr><td class="details-label">Departs</td><td class="details-value">{formatted_dep_time}</td></tr>
#                 <tr><td class="details-label">Date</td><td class="details-value">{flight['date']}</td></tr>
#             </table>
#         </div>
#         """
#         st.markdown(flight_card_html, unsafe_allow_html=True)

#     # --- RIGHT CARD: WEATHER REPORT ---
#     with c_weather:
#         # BUILD THE WEATHER CARD HTML (Now uses {origin_name} dynamically)
#         weather_card_html = f"""
#         <div class="stCard" style="border-top: 5px solid #FFB612;">
#             <h3>☁️ Weather at {originDisplay}</h3>
#             <table class="details-table" style="width:100%">
#                 <tr><td class="details-label">Temp</td><td class="details-value">{temp_f:.0f} °F</td></tr>
#                 <tr><td class="details-label">Wind</td><td class="details-value">{f'0 mph' if wspd_mph == 0 else f'{wspd_mph:.1f} mph'}</td></tr>
#                 <tr><td class="details-label">Precip</td><td class="details-value">{f'0 in' if prcp_in == 0 else f'{prcp_in:.1f} in'}</td></tr>
#                 <tr><td class="details-label">Pressure</td><td class="details-value">{f'0 inHg' if pres_in == 0 else f'{pres_in:.1f} inHg'}</td></tr>
#                 <tr><td class="details-label">Snow</td><td class="details-value">{f'0 in' if snow_in == 0 else f'{snow_in:.1f} in'}</td></tr>
#             </table>
#         </div>
#         """
#         st.markdown(weather_card_html, unsafe_allow_html=True)

#################################################################################################################

import streamlit as st
import datetime
import os
import pickle
import airportsdata

# ==========================================
# 0. CONFIGURE LABEL ENCODERS
# ==========================================
try:
    with open('./Dashboard/label_encoders.pkl', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    st.error("Error: label_encoders.pkl not found. Please ensure the file is in the correct path.")
    data = None
except Exception as e:
    st.error(f"Error loading label encoders: {e}")
    data = None

# ==========================================
# 1. APP CONFIGURATION & SAFE IMPORTS
# ==========================================
st.set_page_config(
    page_title="Flight Delay Predictor ✈️",
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
    st.error("Pandas or Plotly library is missing. Install them to run the app.")

# ==========================================
# 2. DATA LOADING
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(script_dir, 'flight_data.csv.gz')

@st.cache_data
@st.cache_data
def load_data(file_path):
    if not HAS_PANDAS:
        return None
        
    TARGET_YEARS = [2015, 2016, 2017, 2018, 2019, 2023, 2024]
    
    TARGET_ROWS_PER_YEAR = 5
    
    collected = {yr: [] for yr in TARGET_YEARS}
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        # Read CSV in moderately large chunks
        chunks = pd.read_csv(
            file_path,
            chunksize=50_000,
        )
    
        for chunk in chunks:
            chunk.columns = chunk.columns.str.strip()
    
            # Loop for each target year
            for yr in TARGET_YEARS:
                if len(collected[yr]) >= TARGET_ROWS_PER_YEAR:
                    continue
    
                slice_yr = chunk[chunk["Year"] == yr]
    
                if len(slice_yr) > 0:
                    take_n = min(200, len(slice_yr))
                    collected[yr].append(
                        slice_yr.sample(take_n, replace=False, random_state=42)
                    )
    
            # Break if enough total data collected
            total_rows = sum(len(pd.concat(collected[y])) if collected[y] else 0 for y in TARGET_YEARS)
            if total_rows >= 1000:   # enough for downstream sampling
                break
    
        # Combine all into a single dataframe
        frames = []
        for yr in TARGET_YEARS:
            if collected[yr]:
                frames.append(pd.concat(collected[yr]))
    
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            # fallback: read a small amount
            df = pd.read_csv(file_path, chunksize = 1000)
    
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

TEST_DATA_DF = load_data(CSV_FILE_PATH)

# Filter only flights with Risk > 0
if TEST_DATA_DF is not None and 'weatherScore' in TEST_DATA_DF.columns:
    TEST_DATA_DF = TEST_DATA_DF[TEST_DATA_DF['weatherScore'] > 0]
elif TEST_DATA_DF is None:
    st.error("Cannot load flight data. Stopping execution.")
    st.stop()
elif data is None:
    st.error("Cannot load label encoders. Stopping execution.")
    st.stop()

# ==========================================
# 3. SOUTHWEST STYLING (CSS)
# ==========================================
SOUTHWEST_CSS = """<style>
[data-testid="stAppViewContainer"] {background-color: #f4f7f6 !important; color: #333 !important;}
.stCard {background-color: #fff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 20px; border-top: 5px solid #304CB2;}
h1,h2,h3 {color: #304CB2 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 800;}
.sw-stripe {height: 6px; width: 100%; background: linear-gradient(90deg, #304CB2 33%, #C60C30 33%, #C60C30 66%, #FFB612 66%); border-radius: 3px; margin: 10px 0 25px 0;}
.score-container {background: linear-gradient(135deg, #304CB2, #1A2C75); color: #fff; padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 8px 20px rgba(48,76,178,0.3); position: relative; overflow: hidden;}
.score-label {font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; color: #FFB612 !important; font-weight: 700;}
.big-score {font-size: 4rem; font-weight: 900; color: #fff !important; margin: 5px 0;}
button {background-color: #304CB2 !important; color: white !important; border-radius: 50px !important; font-weight: 700 !important; padding: 0.5rem 1rem !important; border: none !important; transition: all 0.3s ease !important;}
button:hover {background-color: #253b8c !important; transform: translateY(-2px); box-shadow: 0 5px 15px rgba(48,76,178,0.3);}
.details-table td {padding: 12px 5px; border-bottom: 1px solid #f0f0f0; color: #444;}
.details-label {font-weight: 700; color: #304CB2; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px;}
.details-value {font-weight: 600; font-size: 1rem; color: #222;}
.stSelectbox label p {color: #304CB2 !important; font-size: 1.1rem !important; font-weight: 700 !important;}
div[data-baseweb="select"] > div {background-color: #fff !important; border: 1px solid #304CB2 !important; color: #333 !important;}
</style>"""
st.markdown(SOUTHWEST_CSS, unsafe_allow_html=True)

NUKE_EXPANDER_CSS = """<style>
summary {background-color: #1e327a !important; color: #fff !important; border-radius: 8px !important;}
summary * {color: #fff !important; fill: #fff !important;}
summary div {color: #fff !important;}
</style>"""
st.markdown(NUKE_EXPANDER_CSS, unsafe_allow_html=True)

# ==========================================
# 4. LOGIC & HEURISTIC ENGINE
# ==========================================
def calculate_risk_score(weather, flight_data):
    risk = 0.0
    if weather['wspd'] > 40: risk += 30
    elif weather['wspd'] > 25: risk += 15
    if weather['prcp'] > 15: risk += 35
    elif weather['prcp'] > 0: risk += 10
    if weather['snow'] > 0: risk += 40
    if weather['pres'] < 1005: risk += 15
    if flight_data['dep_time'] > 1800: risk += 5
    if flight_data['distance'] > 2000: risk += 5
    return max(0.0, min(100.0, risk))

# ==========================================
# 5. USER INTERFACE FLOW
# ==========================================
if 'page' not in st.session_state: st.session_state.page = 'landing'
if 'selected_flight' not in st.session_state: st.session_state.selected_flight = None

# --- LOGO & TITLE ---
col_logo, col_text = st.columns([2, 3])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Southwest_Airlines_logo_2014.svg/320px-Southwest_Airlines_logo_2014.svg.png", width=500)
st.markdown('<div class="sw-stripe"></div>', unsafe_allow_html=True)
st.title("Flight Delay Predictor ✈️")

# --- LANDING PAGE ---
if st.session_state.page == 'landing':
    airports = airportsdata.load('IATA')
    st.markdown("""<div style='text-align:left;color:#000;font-size:20px;font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;font-weight:800;margin-bottom:50px;'>Enter your flight number to get started!</div>""", unsafe_allow_html=True)

    def sample_flights_by_score(df):
        ranges = {'0-10': (0, 10, 5), '10-30': (10, 30, 4), '30-60': (30, 60, 2), '60-80': (60, 80, 2), '80-100': (80, 100, 1)}
        
        selected = []
        used_year_month = set()  # Track (Year, Month) to ensure uniqueness
        filtered = ['WN2933', 'WN2759', 'WN1889', 'WN28', 'WN2606', 'WN1582', 'WN1065', 'WN448']  # Flights to exclude
    
        for low, high, n in ranges.values():
            df_range = df[(df['weatherScore'] > low) & (df['weatherScore'] <= high)]
            if df_range.empty:
                continue
            
            # Filter out flights with duplicate Year-Month
            df_range = df_range[~df_range.apply(lambda row: (row['Year'], row['Month']) in used_year_month, axis=1)]
            
            # Pick up to n flights
            sample_count = min(n, len(df_range))
            if sample_count == 0:
                continue
            
            sample_rows = df_range.head(sample_count)
            for idx, row in sample_rows.iterrows():
                used_year_month.add((row['Year'], row['Month']))
            selected.append(sample_rows)
    
        # Combine all selected flights
        if selected:
            final_df = pd.concat(selected)
        else:
            final_df = df.head(14)
    
        # Build flight number list, excluding filtered flights
        flight_nums = []
        for idx, row in final_df.iterrows():
            fnum = row.get('Flight_Number_Reporting_Airline', 'N/A')
            if fnum != 'N/A':
                try: fnum = f"WN{int(float(fnum))}"
                except: fnum = f"WN{fnum}"
                if fnum not in flight_nums and fnum not in filtered:
                    flight_nums.append(fnum)
    
        flight_nums = pd.Series(flight_nums).sample(frac=1, random_state=42).tolist()
        return flight_nums

    flight_numbers = sample_flights_by_score(TEST_DATA_DF)

    selected_flight_num = st.selectbox("📊 Select a flight number:", options=flight_numbers, key="flight_select")

    # --- GET ROUTES FOR SELECTED FLIGHT ---
    matching_rows = []
    for idx, row in TEST_DATA_DF.iterrows():
        row_fnum = row.get('Flight_Number_Reporting_Airline','N/A')
        if row_fnum != 'N/A':
            try: f_str = f"WN{int(float(row_fnum))}"
            except: f_str = f"WN{row_fnum}"
        else: f_str = 'N/A'
        if f_str == selected_flight_num:
            raw_origin = row.get('Origin','N/A')
            raw_dest = row.get('Dest','N/A')
            try:
                origin_idx = int(float(raw_origin))
                dest_idx = int(float(raw_dest))
                origin_iata = data['Origin'].classes_[origin_idx]
                dest_iata = data['Origin'].classes_[dest_idx]
                originInfo = airports.get(origin_iata)
                destInfo = airports.get(dest_iata)
                origin_display = f"{originInfo.get('name', origin_iata)} ({origin_iata})" if originInfo else origin_iata
                dest_display = f"{destInfo.get('name', dest_iata)} ({dest_iata})" if destInfo else dest_iata
                route_label = f"{origin_display} → {dest_display}"
            except: route_label = f"{raw_origin} → {raw_dest}"
            matching_rows.append({'label': route_label,'index': idx})

    # Remove duplicate routes
    unique_routes = []
    seen = set()
    for item in matching_rows:
        if item['label'] not in seen:
            unique_routes.append(item)
            seen.add(item['label'])

    route_options = [r['label'] for r in unique_routes]
    selected_route = st.selectbox("📍 Select a route:", options=route_options, key="route_select")

    if st.button("Analyze", key="analyze_btn", use_container_width=True):
        selected_route_obj = next(r for r in unique_routes if r['label'] == selected_route)
        selected_index = selected_route_obj['index']
        selected_row = TEST_DATA_DF.loc[selected_index]

        # Flight number normalization
        flight_num = str(selected_row.get('Flight_Number_Reporting_Airline','N/A'))
        if flight_num != 'N/A':
            try: flight_num = f"WN{int(float(flight_num))}"
            except: flight_num = f"WN{flight_num}"

        # Date formatting
        try:
            mm = int(float(selected_row.get('Month',0)))
            dd = int(float(selected_row.get('DayofMonth',0)))
            yy = int(float(selected_row.get('Year',2024)))
            date_str = datetime.date(yy,mm,dd).strftime("%B %d, %Y")
        except:
            date_str = f"Q{selected_row.get('Quarter','N/A')} Day {selected_row.get('DayofMonth','N/A')}"

        def safe_float(v):
            try: val = float(v); return 0 if pd.isna(val) else val
            except: return 0

        flight_data = {
            "id": selected_index,
            "source": "CSV",
            "flight_num": flight_num,
            "date": date_str,
            "origin": str(selected_row.get('Origin','N/A')),
            "dest": str(selected_row.get('Dest','N/A')),
            "distance": safe_float(selected_row.get('Distance',0)),
            "dep_time": int(selected_row.get('CRSDepTime',0)),
            "weather_raw": {
                'tavg': safe_float(selected_row.get('tavg',0)),
                'prcp': safe_float(selected_row.get('prcp',0)),
                'snow': safe_float(selected_row.get('snow',0)),
                'wspd': safe_float(selected_row.get('wspd',0)),
                'pres': safe_float(selected_row.get('pres',0)),
            },
            "true_weather_score": float(selected_row.get('weatherScore',0))
        }
        st.session_state.selected_flight = flight_data
        st.session_state.page = 'result'
        st.rerun()

# --- RESULT PAGE ---
elif st.session_state.page == 'result':
    flight = st.session_state.selected_flight

    c1,c2 = st.columns([1,4])
    if c1.button("← Back"):
        st.session_state.page = 'landing'
        st.session_state.selected_flight = None
        st.rerun()

    weather = flight['weather_raw']
    risk_score = flight.get('true_weather_score',0)

    # Score Card
    st.markdown(f"""<div class="score-container"><div class="score-label">Weather Delay Risk (0=Best, 100=Worst)</div><div class="big-score">{risk_score:.1f}</div></div>""", unsafe_allow_html=True)

    # Gauge & Status
    col_gauge, col_status = st.columns([1,1])
    if risk_score <= 10: status_color="#4CAF50"; status_title="✅ Very Low Risk"; status_msg="Excellent conditions. Expect on-time departure."
    elif risk_score <=30: status_color="#8BC34A"; status_title="🟢 Low Risk"; status_msg="Good conditions, though minor weather factors are present."
    elif risk_score <=60: status_color="#FFB612"; status_title="⚠️ Moderate Risk"; status_msg="Weather/time of day factors present. Potential for minor delays."
    elif risk_score <=80: status_color="#FF5722"; status_title="🚨 High Risk"; status_msg= "Delays are likely."
    else: status_color="#C60C30"; status_title="⛔ Very High Risk"; status_msg="Severe weather. Significant delays or cancellations expected."

    with col_gauge:
        if HAS_PLOTTING:
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=risk_score,
                domain={'x':[0,1],'y':[0,1]},
                gauge={'axis':{'range':[0,100],'tickmode':'array','tickvals':[0,25,50,75,100],'ticktext':['0','25','50','75','100'],'tickfont':{'size':14,'color':'#000'}},
                       'bar':{'color':status_color},
                       'bgcolor':"white",
                       'steps':[{'range':[0,10],'color':'#e8f5e9'},{'range':[10,30],'color':'#f1f8e9'},{'range':[30,60],'color':'#fff8e1'},{'range':[60,80],'color':'#fbe9e7'},{'range':[80,100],'color':'#ffebee'}]}))
            fig.update_layout(height=250,margin=dict(l=40,r=40,t=20,b=20),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)

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
            
            # LOGIC remains in Metric, DISPLAY converts to Imperial
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
                # Use double newlines to force separate lines
                st.markdown("\n\n".join(risks))
            else:
                st.write("No major risk factors.")
    
    with col_dec:
        with st.expander("📉 Factors DECREASING Risk", expanded=True):
            goods = []
            # Logic check remains in Celsius
            if 15 < weather['tavg'] < 30:
                # Convert to Fahrenheit
                temp_f = (weather['tavg'] * 9/5) + 32
                # Display as whole number
                goods.append(f"• Mild Temps ({temp_f:.0f}°F)")
            
            if weather['wspd'] < 15:
                goods.append("• Calm Winds")
            if weather['pres'] >= 1015:
                goods.append("• Stable Pressure")
            if weather['prcp'] == 0:
                goods.append("• No Precipitation")
            
            if goods:
                # Use double newlines to force separate lines
                st.markdown("\n\n".join(goods))
            else:
                st.write("Standard conditions.")
    
# --- FINAL SECTION: CARDS UI ---
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    
    # 1. PREPARE DATA (Do this before columns so both cards can use it)
    if HAS_PANDAS:
        # Decode Airport Names
        origin_iata = flight['origin']
        dest_iata = flight['dest']
        
        # load airport data
        airports = airportsdata.load('IATA') 
        
        # --- FIX: Use .get() and provide safe fallback ---
        originInfo = airports.get(origin_iata)
        destInfo = airports.get(dest_iata)

        # format origin and destination
        if originInfo is None:
            originDisplay = f"{origin_iata} (Info Missing)"
        else:
            originDisplay = originInfo['name'] + " (" + origin_iata + ")"
            
        if destInfo is None:
            destDisplay = f"{dest_iata} (Info Missing)"
        else:
            destDisplay = destInfo['name'] + " (" + dest_iata + ")"
        # ------------------------------------------------

        # Format Times & Distances
        dep_time_str = f"{int(flight['dep_time']):04d}"
        formatted_dep_time = f"{dep_time_str[:2]}:{dep_time_str[2:]}"
        distance_val = f"{int(float(flight['distance']))}"
        
        # Format Weather Values
        temp_f = (weather['tavg'] * 9/5) + 32
        prcp_in = weather['prcp'] * 0.03937
        snow_in = weather['snow'] * 0.03937
        wspd_mph = weather['wspd'] * 0.621371
        pres_in = weather['pres'] * 0.02953

    # 2. CREATE COLUMNS
    c_details, c_weather = st.columns(2)
    
    # --- LEFT CARD: FLIGHT DETAILS ---
    with c_details:
        # BUILD THE FLIGHT CARD HTML
        flight_card_html = f"""
        <div class="stCard">
            <h3>✈️ Flight Details</h3>
            <table class="details-table" style="width:100%">
                <tr><td class="details-label">Flight No.</td><td class="details-value">{flight['flight_num']}</td></tr>
                <tr><td class="details-label">Route</td><td class="details-value">{originDisplay} ➝ {destDisplay}</td></tr>
                <tr><td class="details-label">Distance</td><td class="details-value">{distance_val} mi</td></tr>
                <tr><td class="details-label">Departs</td><td class="details-value">{formatted_dep_time}</td></tr>
                <tr><td class="details-label">Date</td><td class="details-value">{flight['date']}</td></tr>
            </table>
        </div>
        """
        st.markdown(flight_card_html, unsafe_allow_html=True)

    # --- RIGHT CARD: WEATHER REPORT ---
    with c_weather:
        # BUILD THE WEATHER CARD HTML (Now uses {origin_name} dynamically)
        weather_card_html = f"""
        <div class="stCard" style="border-top: 5px solid #FFB612;">
            <h3>☁️ Weather at {originDisplay}</h3>
            <table class="details-table" style="width:100%">
                <tr><td class="details-label">Temp</td><td class="details-value">{temp_f:.0f} °F</td></tr>
                <tr><td class="details-label">Wind</td><td class="details-value">{f'0 mph' if wspd_mph == 0 else f'{wspd_mph:.1f} mph'}</td></tr>
                <tr><td class="details-label">Precip</td><td class="details-value">{f'0 in' if prcp_in == 0 else f'{prcp_in:.1f} in'}</td></tr>
                <tr><td class="details-label">Pressure</td><td class="details-value">{f'0 inHg' if pres_in == 0 else f'{pres_in:.1f} inHg'}</td></tr>
                <tr><td class="details-label">Snow</td><td class="details-value">{f'0 in' if snow_in == 0 else f'{snow_in:.1f} in'}</td></tr>
            </table>
        </div>
        """
        st.markdown(weather_card_html, unsafe_allow_html=True)
