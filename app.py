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
    <h1 style="font-size: 3.5rem; margin-bottom: 0;">🍷 WineWise</h1>
    <p style="font-size: 1.3rem; font-style: italic; margin-top: 0;">Uncorking the Science of Fine Wine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with richer content
st.sidebar.markdown("""
# 🍇 Wine Quality Analyzer
Discover what makes a great wine!

This app uses machine learning to analyze wine based on its chemical properties.

### How It Works:
1. Adjust the sliders for wine properties
2. Click "Analyze Wine Quality"
3. Get detailed insights and recommendations

### Key Wine Parameters:
- Alcohol: Higher levels often lead to better quality
- Acidity: Balances flavors and preserves wine
- Sulfur Dioxide: Prevents oxidation and microbial growth

Swirl, smell, sip, and now... science!
""")

# Load dataset for feature importance and visualization
try:
    wine_data = pd.read_csv("winequalityN.csv")
except:
    # Create mock data if file is not found
    st.sidebar.warning("Using sample data for demonstration")
    # Create more realistic sample data with quality as a categorical variable
    np.random.seed(42)  # For reproducible results
    
    # Generate sample data for white and red wines
    n_samples = 1000
    wine_types = np.random.choice([0, 1], size=n_samples)  # 0 for red, 1 for white
    
    # Different distributions for red and white wines
    alcohol_values = []
    volatile_acidity_values = []
    quality_values = []
    
    for wine_type in wine_types:
        if wine_type == 0:  # Red wine
            # Red wines tend to have higher alcohol and volatile acidity
            alcohol = np.random.normal(11.5, 1.0)
            volatile_acidity = np.random.normal(0.6, 0.15)
        else:  # White wine
            # White wines tend to have lower alcohol and volatile acidity
            alcohol = np.random.normal(10.2, 0.8)
            volatile_acidity = np.random.normal(0.3, 0.1)
        
        # Determine quality based on alcohol and volatile acidity
        # Higher alcohol generally good, higher volatile acidity generally bad
        quality_score = 5 + (alcohol - 10) * 0.8 - (volatile_acidity - 0.5) * 3
        
        if quality_score < 5:
            quality = 'poor'
        elif quality_score <= 7:
            quality = 'average'
        else:
            quality = 'good'
            
        alcohol_values.append(alcohol)
        volatile_acidity_values.append(volatile_acidity)
        quality_values.append(quality)
    
    wine_data = pd.DataFrame({
        'type': wine_types,
        'fixed acidity': np.random.normal(8.32, 1.7, n_samples),
        'volatile acidity': volatile_acidity_values,
        'citric acid': np.random.normal(0.27, 0.2, n_samples),
        'residual sugar': np.random.normal(2.5, 1.4, n_samples),
        'chlorides': np.random.normal(0.088, 0.05, n_samples),
        'free sulfur dioxide': np.random.normal(15.87, 10.5, n_samples),
        'total sulfur dioxide': np.random.normal(46.47, 32.9, n_samples),
        'density': np.random.normal(0.9967, 0.003, n_samples),
        'pH': np.random.normal(3.31, 0.16, n_samples),
        'sulphates': np.random.normal(0.66, 0.17, n_samples),
        'alcohol': alcohol_values,
        'quality': quality_values,
    })

# Try to load model and scaler
try:
    model = joblib.load("wine_rating.pkl")
    scaler = joblib.load("scaler_rating.pkl")
    model_loaded = True
except:
    # Create mock prediction function if model is not found
    st.sidebar.warning("Using simulation mode - model not found")
    model_loaded = False

def calculate_quality_score(input_data, wine_type_str):
    """Calculate wine quality score on a scale of 0-10 with different parameters for red and white wine"""
    # Base score from 0-10
    score = 5.0  # Start at middle
    
    # Common factors for both wine types
    score -= (input_data['volatile acidity'].values[0] - 0.5) * 2  # Acidity penalty
    
    # Wine type specific adjustments
    if wine_type_str == "Red":
        # Red wine specific factors
        score += (input_data['alcohol'].values[0] - 10) * 0.6  # Alcohol bonus (slightly less impact for red)
        score += (input_data['sulphates'].values[0] - 0.5) * 2.0  # Sulphates bonus (more important for red)
        
        if input_data['total sulfur dioxide'].values[0] > 150:
            score -= 1.0  # High SO2 is more negative for red wines
            
        if input_data['fixed acidity'].values[0] > 8.5:
            score += 0.5  # Higher fixed acidity can be good for red wines
            
    else:  # White wine
        # White wine specific factors
        score += (input_data['alcohol'].values[0] - 10) * 0.4  # Alcohol bonus (less impact for white)
        score += (input_data['sulphates'].values[0] - 0.5) * 1.0  # Sulphates bonus (less important for white)
        
        if input_data['residual sugar'].values[0] > 3.0:
            score += 0.7  # Sweetness can be positive for white wines
            
        if input_data['citric acid'].values[0] > 0.3:
            score += 0.8  # Citric acid more important for white wines
    
    # Additional common factors
    if 3.0 <= input_data['pH'].values[0] <= 3.4:
        score += 0.5
    
    # Ensure score is within 0-10 range
    return max(0, min(10, score))

def predict_quality(input_data, wine_type_str):
    """Predict wine quality category and score"""
    # Calculate quality score with wine type consideration
    quality_score = calculate_quality_score(input_data, wine_type_str)
    
    # Apply classification thresholds
    if quality_score < 5:
        return 'poor', quality_score
    elif quality_score <= 7:
        return 'average', quality_score
    else:
        return 'good', quality_score

# Input section with better organization and tooltips
def user_input_features():
    st.header("🔬 Wine Chemical Profile")
    
    # Help expander with wine properties explanation
    with st.expander("ℹ Understanding Wine Properties"):
        st.markdown("""
        ### Wine Property Guide:
        
        - Fixed Acidity: Primarily tartaric acid, gives wine its tart taste
        - Volatile Acidity: Excessive amounts can lead to unpleasant vinegar taste
        - Citric Acid: Adds 'freshness' and flavor to wines
        - Residual Sugar: Amount of sugar remaining after fermentation
        - Chlorides: Amount of salt in the wine
        - Sulfur Dioxide: Prevents microbial growth and oxidation
        - Density: How close the wine is to water's density
        - pH: Describes how acidic or basic the wine is (0-14)
        - Sulphates: Additive that contributes to SO2 levels, antimicrobial
        - Alcohol: Percentage of alcohol in the wine
        
        The perfect balance of these properties creates outstanding wine!
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
Adjusted values will update when you submit for analysis
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
        
        # Get prediction using the fixed function that considers wine type
        prediction, quality_out_of_10 = predict_quality(input_df, wine_type)
    
    # Convert prediction to status and description based on NEW ranges
    if prediction == 'good':  # > 7
        quality_status = "🟢 Excellent Quality"
        
        description = """
        This wine shows exceptional balance with vibrant aromatics and complex flavors. 
        The chemical profile indicates excellent aging potential. It would pair beautifully with 
        rich foods and special occasions.
        """
        
    elif prediction == 'average':  # 5-7
        quality_status = "🟡 Average Quality"
        
        description = """
        A pleasant, approachable wine with good balance. While not extraordinary, 
        it offers enjoyable drinking and represents good value. Suitable for everyday 
        enjoyment and casual dining.
        """
        
    else:  # poor < 5
        quality_status = "🔴 Poor Quality"
        
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
        <p>According to our new quality scale:</p>
        <ul>
            <li><strong>Poor:</strong> Score less than 5</li>
            <li><strong>Average:</strong> Score between 5 and 7</li>
            <li><strong>Excellent:</strong> Score greater than 7</li>
        </ul>
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
        # Alcohol vs. Quality scatter plot - FIXED
        st.subheader("Alcohol vs. Quality Relationship")
        
        # Create and handle sample data for the plot - FIXED CODE HERE
        # Create a scatterplot showing alcohol vs. volatile acidity colored by quality
        if 'quality' in wine_data.columns:
            # Ensure quality values are valid categories
            if isinstance(wine_data['quality'].iloc[0], str):
                # If quality is already categorical strings
                valid_qualities = ['poor', 'average', 'good']
                filtered_data = wine_data[wine_data['quality'].isin(valid_qualities)]
            else:
                # If quality is numeric, convert to categories
                wine_data['quality_category'] = pd.cut(
                    wine_data['quality'],
                    bins=[0, 5, 7, 10],
                    labels=['poor', 'average', 'good']
                )
                filtered_data = wine_data.copy()
                filtered_data['quality'] = filtered_data['quality_category']
            
            # Sample data for better visualization (not too many points)
            sample_size = min(300, len(filtered_data))
            plot_data = filtered_data.sample(sample_size) if len(filtered_data) > sample_size else filtered_data
            
            # Create color mapping
            color_discrete_map = {'good': 'green', 'average': 'gold', 'poor': 'red'}
            
            # Create size mapping for plot
            plot_data['marker_size'] = plot_data['quality'].map({'poor': 5, 'average': 10, 'good': 15})
            
            # Create the scatter plot
            fig = px.scatter(
                plot_data,
                x='alcohol',
                y='volatile acidity',
                color='quality',
                size='marker_size',
                size_max=15,
                opacity=0.7,
                color_discrete_map=color_discrete_map,
                labels={'alcohol': 'Alcohol (%)', 'volatile acidity': 'Volatile Acidity'},
                title='Wine Quality by Alcohol & Acidity',
                height=400
            )
            
            # Add the current wine as a highlighted point
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
            
            # Update layout for dark theme
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                legend_title_text='Quality',
                xaxis=dict(gridcolor='rgba(255,255,255,0.2)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            )
            
            st.plotly_chart(fig)
        else:
            st.error("Could not create scatter plot due to missing quality data")

    # Wine pairing suggestions based on profile
    st.header("🍽 Food Pairing Suggestions")
    
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
        
        st.markdown(f"- 🌡 Serving Temperature: {temp}")
        st.markdown(f"- 🥂 Glass Type: {glass}")
        st.markdown(f"- ⏱ Aeration: {aerate}")

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
 # Example values (you should replace these with logic based on your model or input)
style = "crisp and aromatic"
wine_type = "White"
region = "Alsace"

# Then your markdown block works:
st.markdown(f"""
The chemical profile of this {wine_type.lower()} wine suggests a {style} style, 
reminiscent of wines from {region}. 

#### The Alchemy in Your Glass

Wine is truly a living chemistry experiment. As the grape juice ferments, yeasts convert 
sugar into alcohol and carbon dioxide while creating hundreds of aromatic compounds. 
The {input_df['fixed acidity'].values[0]:.1f} g/L of fixed acidity works with the pH of {input_df['pH'].values[0]:.1f} 
to create the wine's structural backbone.

Meanwhile, the {input_df['alcohol'].values[0]:.1f}% alcohol content provides body and warmth.
""")

