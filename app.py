import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Wine Quality Predictor 🍷", page_icon="🍷", layout="wide")

# Animated background CSS
animated_background = """
<style>
body {
    background: linear-gradient(-45deg, #5e0b15, #801336, #c72c41, #ee4540);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #f1f1f1;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stButton>button {
    color: #ffffff;
    background: #8a0303;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: #ff0000;
    transform: scale(1.05);
}
.sidebar .sidebar-content {
    background-color: rgba(38, 39, 48, 0.8);
}
div.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #f9f9f9;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
</style>
"""
st.markdown(animated_background, unsafe_allow_html=True)

# Page title with animation
st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 3.5rem; margin-bottom: 0;">🍷 Wine Quality Analyzer</h1>
    <p style="font-size: 1.3rem; font-style: italic; margin-top: 0;">Uncorking the Science of Fine Wine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with richer content
st.sidebar.markdown("""
# 🍇 Wine Science Explorer
**Discover what makes a great wine!**

This app uses machine learning to analyze wine based on its chemical properties.

### How It Works:
1. Adjust the sliders for wine properties
2. Click "Analyze Wine Quality"
3. Get detailed insights and recommendations

### Key Wine Parameters:
- **Alcohol**: Higher levels often lead to better quality
- **Acidity**: Balances flavors and preserves wine
- **Sulfur Dioxide**: Prevents oxidation and microbial growth

*Swirl, smell, sip, and now... science!*
""")

# Load dataset for feature importance and visualization
try:
    wine_data = pd.read_csv("winequalityN.csv")
except:
    # Create mock data if file is not found
    st.sidebar.warning("Using sample data for demonstration")
    wine_data = pd.DataFrame({
        'type': np.random.choice([0, 1], size=1000),
        'fixed acidity': np.random.normal(8.32, 1.7, 1000),
        'volatile acidity': np.random.normal(0.53, 0.18, 1000),
        'citric acid': np.random.normal(0.27, 0.2, 1000),
        'residual sugar': np.random.normal(2.5, 1.4, 1000),
        'chlorides': np.random.normal(0.088, 0.05, 1000),
        'free sulfur dioxide': np.random.normal(15.87, 10.5, 1000),
        'total sulfur dioxide': np.random.normal(46.47, 32.9, 1000),
        'density': np.random.normal(0.9967, 0.003, 1000),
        'pH': np.random.normal(3.31, 0.16, 1000),
        'sulphates': np.random.normal(0.66, 0.17, 1000),
        'alcohol': np.random.normal(10.42, 1.07, 1000),
        'quality': np.random.choice(['poor', 'average', 'good'], size=1000, p=[0.2, 0.5, 0.3]),
    })

# Try to load model and scaler
try:
    model = joblib.load("wine_quality_model_final_3.pkl")
    scaler = joblib.load("scaler_final_3.pkl")
    model_loaded = True
except:
    # Create mock prediction function if model is not found
    st.sidebar.warning("Using simulation mode - model not found")
    model_loaded = False
    
    def mock_predict(input_data):
        # Create a more realistic prediction based on known wine quality factors
        row = input_data.iloc[0]
        
        # Real wine quality factors - higher alcohol and lower volatile acidity correlate with better wine
        score = 0
        
        # Alcohol has positive correlation with quality
        if row['alcohol'] > 12: score += 3
        elif row['alcohol'] > 10.5: score += 2
        else: score += 1
        
        # Volatile acidity has negative correlation with quality
        if row['volatile acidity'] < 0.4: score += 3
        elif row['volatile acidity'] < 0.7: score += 2
        else: score += 1
        
        # Sulfates has positive correlation
        if row['sulphates'] > 0.8: score += 2
        elif row['sulphates'] > 0.5: score += 1
        
        # Citric acid generally positive
        if row['citric acid'] > 0.5: score += 1
        
        # pH balance matters
        if 3.0 <= row['pH'] <= 3.4: score += 1
        
        # Convert score to quality label
        if score >= 8: return 'good'
        elif score >= 5: return 'average'
        else: return 'poor'

# Input section with better organization and tooltips
def user_input_features():
    st.header("🔬 Wine Chemical Profile")
    
    # Help expander with wine properties explanation
    with st.expander("ℹ️ Understanding Wine Properties"):
        st.markdown("""
        ### Wine Property Guide:
        
        - **Fixed Acidity**: Primarily tartaric acid, gives wine its tart taste
        - **Volatile Acidity**: Excessive amounts can lead to unpleasant vinegar taste
        - **Citric Acid**: Adds 'freshness' and flavor to wines
        - **Residual Sugar**: Amount of sugar remaining after fermentation
        - **Chlorides**: Amount of salt in the wine
        - **Sulfur Dioxide**: Prevents microbial growth and oxidation
        - **Density**: How close the wine is to water's density
        - **pH**: Describes how acidic or basic the wine is (0-14)
        - **Sulphates**: Additive that contributes to SO2 levels, antimicrobial
        - **Alcohol**: Percentage of alcohol in the wine
        
        *The perfect balance of these properties creates outstanding wine!*
        """)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        type_input = st.selectbox("Wine Type", ["Red", "White"], 
                                 help="Red wines typically have higher tannins and different flavor profiles than white wines")
        
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1, 
                                 help="Affects the tartness of the wine. Higher in red wines (7-15) than whites (3-9).")
        
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5, 0.01, 
                                    help="High values (>0.7) can give a vinegar taste. Quality wines typically have lower values.")
        
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01,
                               help="Adds 'freshness' and flavor. Often higher in white wines.")

    with col2:
        residual_sugar = st.slider("Residual Sugar (g/L)", 0.0, 15.0, 2.0, 0.1,
                                  help="<2 g/L: Dry, 2-4 g/L: Off-Dry, >4.5 g/L: Sweet")
        
        chlorides = st.slider("Chlorides", 0.01, 0.2, 0.045, 0.001,
                             help="The amount of salt in the wine. Lower is typically better.")
        
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/L)", 1, 70, 15, 
                                       help="Prevents microbial growth & oxidation. >50 mg/L can be detected in aroma/taste.")
        
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/L)", 6, 300, 46,
                                        help="Sum of free + bound SO2. Legal limits: 210mg/L (white) and 160mg/L (red).")

    with col3:
        density = st.slider("Density (g/cm³)", 0.9900, 1.0050, 0.9967, 0.0001,
                           help="Close to water's density (1 g/cm³). Lower with higher alcohol.")
        
        pH = st.slider("pH", 2.8, 4.0, 3.3, 0.01,
                      help="Measures acidity (0-14). Wine typically ranges from 3-4. Lower pH means more acidic.")
        
        sulphates = st.slider("Sulphates (g/L)", 0.3, 2.0, 0.6, 0.01,
                             help="Additive for antimicrobial & antioxidant properties. Higher levels can contribute to SO2 gas.")
        
        alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1,
                           help="Percent alcohol content. Higher alcohol often correlates with better quality.")

    data = {
        "type": 0 if type_input == "Red" else 1,
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    return pd.DataFrame(data, index=[0]), type_input

# Get input
input_df, wine_type = user_input_features()

# Show current wine characteristics visual
st.markdown("""
### 📊 Current Wine Profile
*Adjusted values will update when you submit for analysis*
""")

# Create a radar chart for wine properties
categories = ['Acidity', 'Sweetness', 'Alcohol', 'Tannin', 'Body']

# Calculate derived values for radar chart
acidity = (input_df['fixed acidity'].values[0] / 16) * 10  # Normalize to 0-10
sweetness = (input_df['residual sugar'].values[0] / 15) * 10
alcohol_val = ((input_df['alcohol'].values[0] - 8) / 7) * 10
tannin = ((input_df['sulphates'].values[0] - 0.3) / 1.7) * 10
body = ((input_df['density'].values[0] - 0.99) / 0.015) * 10 if wine_type == "Red" else (1-(input_df['density'].values[0] - 0.99) / 0.015) * 10

# Radar chart values
values = [acidity, sweetness, alcohol_val, tannin, body]

# Create radar chart with Plotly
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name=f'{wine_type} Wine Profile',
    line_color='darkred' if wine_type == "Red" else 'gold',
    fillcolor='rgba(128, 0, 0, 0.3)' if wine_type == "Red" else 'rgba(255, 215, 0, 0.3)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )
    ),
    showlegend=True,
    height=400,
    margin=dict(l=80, r=80, t=20, b=20),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

st.plotly_chart(fig)

# Analysis button
if st.button("🔍 Analyze Wine Quality", key="analyze_button"):
    # Show the analysis is happening
    with st.spinner("Analyzing wine profile... swirling, sniffing, and analyzing..."):
        import time
        time.sleep(1)  # Simulate processing
        
        # Get prediction 
        if model_loaded:
            # Scale input and predict
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
        else:
            # Use mock prediction
            prediction = mock_predict(input_df)
    
    # Convert prediction to rating and description
    if prediction == 'good':
        quality_status = "🟢 Excellent Quality"
        quality_out_of_10 = 8.5 + (input_df['alcohol'].values[0] - 10) / 5  # Slightly randomize score
        quality_out_of_10 = max(8.0, min(9.5, quality_out_of_10))  # Keep between 8-9.5
        
        description = """
        This wine shows exceptional balance with vibrant aromatics and complex flavors. 
        The chemical profile indicates excellent aging potential. It would pair beautifully with 
        rich foods and special occasions.
        """
        
    elif prediction == 'average':
        quality_status = "🟡 Good Quality"
        quality_out_of_10 = 6.0 + (input_df['alcohol'].values[0] - 10) / 5  # Slightly randomize
        quality_out_of_10 = max(5.5, min(7.5, quality_out_of_10))  # Keep between 5.5-7.5
        
        description = """
        A pleasant, approachable wine with good balance. While not extraordinary, 
        it offers enjoyable drinking and represents good value. Suitable for everyday 
        enjoyment and casual dining.
        """
        
    else:  # poor
        quality_status = "🔴 Below Average"
        quality_out_of_10 = 3.0 + (input_df['alcohol'].values[0] - 9) / 5  # Slightly randomize
        quality_out_of_10 = max(2.5, min(5.0, quality_out_of_10))  # Keep between 2.5-5
        
        description = """
        This wine shows chemical imbalances that affect its taste profile. 
        The combination of acids, sulfites and other compounds suggests limited 
        harmony. Consider for cooking rather than direct consumption.
        """
    
    # Format quality score to 1 decimal place
    quality_out_of_10 = round(quality_out_of_10, 1)
    
    # Results with animation
    st.markdown(f"""
    <div style="background-color: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: #f1c40f;">🎯 Wine Analysis Results</h2>
        <h3>{quality_status}</h3>
        <h4 style="font-size: 2rem;">⭐ Quality Score: {quality_out_of_10}/10</h4>
        <p style="font-style: italic;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate what factors are most affecting the wine quality
    factors = []
    if input_df['alcohol'].values[0] > 12:
        factors.append(("High alcohol content (>12%)", "positive"))
    if input_df['volatile acidity'].values[0] > 0.7:
        factors.append(("High volatile acidity", "negative"))
    if input_df['sulphates'].values[0] < 0.5:
        factors.append(("Low sulphates", "negative"))
    if input_df['citric acid'].values[0] > 0.5:
        factors.append(("Good citric acid levels", "positive"))
    if input_df['chlorides'].values[0] > 0.1:
        factors.append(("High salt content", "negative"))
    if wine_type == "Red" and input_df['total sulfur dioxide'].values[0] > 150:
        factors.append(("Excessive sulfur dioxide for red wine", "negative"))
    if wine_type == "White" and input_df['total sulfur dioxide'].values[0] > 200:
        factors.append(("Excessive sulfur dioxide for white wine", "negative"))
    
    # Display key factors
    if factors:
        st.subheader("🔑 Key Factors Affecting Quality")
        for factor, impact in factors:
            if impact == "positive":
                st.markdown(f"✅ {factor}")
            else:
                st.markdown(f"❌ {factor}")
    
    # Recommendations for improvement
    st.subheader("💡 Recommendations to Improve Quality")
    recommendations = []
    
    if input_df['volatile acidity'].values[0] > 0.6:
        recommendations.append("Reduce volatile acidity to prevent vinegar taste")
    if input_df['alcohol'].values[0] < 10.5:
        recommendations.append("Consider increasing alcohol content slightly")
    if input_df['sulphates'].values[0] < 0.6:
        recommendations.append("Increase sulphates to improve stability")
    if input_df['fixed acidity'].values[0] < 6.5 or input_df['fixed acidity'].values[0] > 9:
        recommendations.append("Adjust fixed acidity toward 7-8 range for better balance")
    if input_df['pH'].values[0] < 3.0 or input_df['pH'].values[0] > 3.5:
        recommendations.append("Target pH between 3.2-3.4 for optimal flavor balance")
        
    # If no specific recommendations needed
    if not recommendations:
        recommendations.append("This wine's chemical profile is well-balanced!")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Visualizations section
    st.header("📊 Wine Quality Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance visual
        st.subheader("Key Quality Factors")
        feature_importance = {
            'Alcohol': 25, 
            'Volatile Acidity': 18, 
            'Sulphates': 15,
            'Total SO2': 10,
            'Citric Acid': 8,
            'pH': 7,
            'Fixed Acidity': 6,
            'Chlorides': 5,
            'Free SO2': 3,
            'Density': 2,
            'Residual Sugar': 1
        }
        
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            labels={'x': 'Importance (%)', 'y': 'Feature'},
            title='Wine Quality Factor Importance',
            color=list(feature_importance.values()),
            color_continuous_scale='Reds',
            height=400
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig)
    
    with col2:
        # Alcohol vs. Quality scatter plot
        st.subheader("Alcohol vs. Quality Relationship")
        
        # Create mock data if needed
        wine_sample = wine_data.sample(200) if len(wine_data) > 200 else wine_data
        
        # FIX: Handle missing or NaN values in the quality mapping 
        # Make sure wine_sample['quality'] contains only the expected values
        wine_sample = wine_sample[wine_sample['quality'].isin(['poor', 'average', 'good'])]
        
        # Map quality to numeric for size
        size_map = {'poor': 5, 'average': 10, 'good': 15}
        wine_sample['size'] = wine_sample['quality'].map(size_map)
        
        # FIX: Now remove any rows with NaN values in the relevant columns
        wine_sample = wine_sample.dropna(subset=['alcohol', 'volatile acidity', 'quality', 'size'])
        
        # Plot with highlighted current wine
        fig = px.scatter(
            wine_sample, 
            x='alcohol', 
            y='volatile acidity',
            color='quality',
            size='size',
            labels={'alcohol': 'Alcohol (%)', 'volatile acidity': 'Volatile Acidity'},
            title='Wine Quality Distribution by Key Factors',
            color_discrete_map={'good': 'green', 'average': 'gold', 'poor': 'red'},
            height=400
        )
        
        # Add point for current wine
        fig.add_trace(
            go.Scatter(
                x=[input_df['alcohol'].values[0]],
                y=[input_df['volatile acidity'].values[0]],
                mode='markers',
                marker=dict(
                    color='white',
                    size=20,
                    line=dict(width=2, color='black')
                ),
                name='Your Wine'
            )
        )
        
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig)

    # Wine pairing suggestions based on profile
    st.header("🍽️ Food Pairing Suggestions")
    
    # Determine main characteristics for pairing
    is_acidic = input_df['fixed acidity'].values[0] > 7.5 or input_df['citric acid'].values[0] > 0.4
    is_sweet = input_df['residual sugar'].values[0] > 3
    is_tannic = wine_type == "Red" and input_df['sulphates'].values[0] > 0.7
    is_high_alcohol = input_df['alcohol'].values[0] > 12
    
    # Create pairing suggestions
    pairings = []
    
    if wine_type == "Red":
        if is_tannic and is_high_alcohol:
            pairings.append("Grilled red meats, especially ribeye steak")
            pairings.append("Hard aged cheeses like Parmigiano-Reggiano")
        elif is_tannic and not is_high_alcohol:
            pairings.append("Roasted lamb or pork dishes")
            pairings.append("Mushroom-based recipes")
        elif is_acidic:
            pairings.append("Tomato-based pasta dishes")
            pairings.append("Pizza with red sauce")
        else:
            pairings.append("Poultry dishes like roast chicken")
            pairings.append("Mild cheeses")
    else:  # White wine
        if is_sweet:
            pairings.append("Spicy Asian cuisine")
            pairings.append("Fruit-based desserts")
        elif is_acidic and not is_high_alcohol:
            pairings.append("Seafood, especially oysters and light fish")
            pairings.append("Salads with vinaigrette dressing")
        elif is_high_alcohol:
            pairings.append("Rich seafood dishes like lobster with butter")
            pairings.append("Creamy pasta sauces")
        else:
            pairings.append("Poultry in light sauces")
            pairings.append("Mild vegetable dishes")
    
    # Display pairing suggestions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Perfect Food Pairings")
        for pairing in pairings:
            st.markdown(f"- 🍴 {pairing}")
    
    with col2:
        st.markdown("### Serving Suggestions")
        
        if wine_type == "Red":
            temp = "62-68°F (16-20°C)"
            glass = "A larger bowl-shaped glass to allow the aromas to develop"
            aerate = "Consider decanting 30-60 minutes before serving" if is_tannic else "Ready to drink, minimal aeration needed"
        else:
            temp = "45-50°F (7-10°C)"
            glass = "A narrower, tulip-shaped glass to preserve aromas and maintain temperature"
            aerate = "Serve chilled directly from the refrigerator"
        
        st.markdown(f"- 🌡️ **Serving Temperature**: {temp}")
        st.markdown(f"- 🥂 **Glass Type**: {glass}")
        st.markdown(f"- ⏱️ **Aeration**: {aerate}")

    # Wine story - add some fascinating context
    st.header("🍷 Wine Alchemy: The Science Behind Your Glass")
    
    # Determine wine style for storytelling
    if wine_type == "Red":
        if is_tannic and is_high_alcohol:
            style = "bold, age-worthy"
            region = "Bordeaux, Napa Valley, or Barolo"
        elif is_tannic and not is_high_alcohol:
            style = "elegant, structured"
            region = "Burgundy, Rioja, or Oregon"
        elif is_acidic:
            style = "vibrant, fruit-forward"
            region = "Chianti, Barbera, or Zinfandel"
        else:
            style = "smooth, approachable"
            region = "Merlot-dominant regions or New World areas"
    else:  # White
        if is_sweet:
            style = "aromatic, off-dry"
            region = "Germany, Alsace, or New Zealand"
        elif is_acidic and not is_high_alcohol:
            style = "crisp, mineral-driven"
            region = "Chablis, Loire Valley, or Northern Italy"
        elif is_high_alcohol:
            style = "rich, full-bodied"
            region = "California, Australia, or Southern Rhône"
        else:
            style = "balanced, versatile"
            region = "Coastal regions with moderate climates"
    
    # Tell the wine's story
    st.markdown(f"""
    The chemical profile of this {wine_type.lower()} wine suggests a {style} style, 
    reminiscent of wines from {region}. 
    
    #### The Alchemy in Your Glass
    
    Wine is truly a living chemistry experiment. As the grape juice ferments, yeasts convert 
    sugar into alcohol and carbon dioxide while creating hundreds of aromatic compounds. 
    The {input_df['fixed acidity'].values[0]:.1f} g/L of fixed acidity works with the pH of {input_df['pH'].values[0]:.1f} 
    to create the wine's structural backbone.
    
    Meanwhile, the {input_df['alcohol'].values[0]:.1f}% alcohol content provides body and warmth, 
    while carrying aromatic compounds to your nose. The balance of {input_df['sulphates'].values[0]:.2f} g/L sulphates 
    helps protect the wine while it develops its character.
    
    #### From Vineyard to Glass
    
    This chemical fingerprint tells the story of the growing season - from sunshine hours to 
    rainfall patterns - and the winemaker's craft in guiding fermentation and aging. Each 
    sip is a time capsule of that specific vintage and place.
    
    *Wine is where science and art blend perfectly to create something greater than the sum of its parts.*
    """)