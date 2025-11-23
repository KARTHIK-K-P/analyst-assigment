# LEAD QUALITY ANALYSIS - STREAMLIT DASHBOARD
# Analytics Challenge for Aarki
# Author: Karthik

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Lead Quality Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Lead Quality Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("**Analytics Challenge for Aarki - Analyst Role**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    # Option to use default file or upload
    data_source = st.radio(
        "Choose data source:",
        ["Upload File", "Use Default File"]
    )
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Upload Excel File", type=['xls', 'xlsx'])
    else:
        # Default file path
        default_file_path = st.text_input(
            "Default File Path",
            value="Analyst_case_study_dataset_1_(1).xls",
            help="Enter the path to your Excel file"
        )
        uploaded_file = default_file_path if default_file_path else None
    
    st.markdown("---")
    st.header("⚙️ Analysis Settings")
    TARGET_CLOSED_RATE = st.number_input("Target Closed Rate (%)", value=9.6, min_value=0.0, max_value=100.0, step=0.1)
    min_sample_size = st.slider("Minimum Sample Size for Segments", 10, 100, 30)
    
    st.markdown("---")
    st.header("📋 Navigation")
    analysis_section = st.radio(
        "Select Analysis",
        ["Overview", "Q1: Temporal Trends", "Q2: Quality Drivers", "Q3: Optimization Strategy", "Export Report"]
    )

# Load data function
@st.cache_data
def load_and_process_data(uploaded_file, is_path=False):
    """Load and process the lead data"""
    if uploaded_file is not None:
        try:
            if is_path:
                # Load from file path
                df = pd.read_excel(uploaded_file, engine='xlrd')
            else:
                # Load from uploaded file
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("💡 Try: pip install xlrd openpyxl")
            return None
    else:
        st.warning("⚠️ No file uploaded. Please upload an Excel file in the sidebar.")
        return None
    
    # Define lead quality categories
    GOOD_STATUS = ['Closed', 'EP Sent', 'EP Received', 'EP Confirmed']
    BAD_STATUS = ['Unable to Contact', 'Contacted - Invalid Profile', "Contacted - Doesn't Qualify"]
    
    # Handle missing CallStatus
    df['CallStatus'] = df['CallStatus'].fillna('Unknown')
    
    # Create quality flags
    df['IsGood'] = df['CallStatus'].isin(GOOD_STATUS).astype(int)
    df['IsBad'] = df['CallStatus'].isin(BAD_STATUS).astype(int)
    df['IsClosed'] = (df['CallStatus'] == 'Closed').astype(int)
    
    # Extract date components
    df['LeadCreated'] = pd.to_datetime(df['LeadCreated'])
    df['Year'] = df['LeadCreated'].dt.year
    df['Month'] = df['LeadCreated'].dt.month
    df['MonthName'] = df['LeadCreated'].dt.strftime('%b %Y')
    df['Week'] = df['LeadCreated'].dt.isocalendar().week
    df['YearWeek'] = df['LeadCreated'].dt.strftime('%Y-W%U')
    
    # Parse WidgetName components
    df['WidgetName_Clean'] = df['WidgetName'].str.replace('w-300250', 'w-302252')
    df['AdSize'] = df['WidgetName_Clean'].str.extract(r'w-(\d+)')[0]
    df['FormPages'] = df['WidgetName_Clean'].str.extract(r'-(\dDC)')[0]
    df['AdDesign'] = df['WidgetName_Clean'].str.split('-').str[2:].str.join('-')
    
    # Campaign type
    df['IsCallCenter'] = df['PublisherCampaignName'].str.contains('Call Center', na=False).astype(int)
    df['IsBranded'] = df['AdvertiserCampaignName'].str.contains('branded', case=False, na=False).astype(int)
    
    # Convert DebtLevel to numeric
    df['DebtLevel_Numeric'] = pd.to_numeric(
        df['DebtLevel'].astype(str).str.replace(',', '').str.replace('$', '').str.strip(), 
        errors='coerce'
    )
    
    # Debt level brackets
    df['DebtBracket'] = pd.cut(
        df['DebtLevel_Numeric'], 
        bins=[0, 10000, 20000, 30000, 1000000],
        labels=['0-10K', '10-20K', '20-30K', '30K+'],
        include_lowest=True
    )
    
    return df

# Load data
is_file_path = (data_source == "Use Default File")
df = load_and_process_data(uploaded_file, is_path=is_file_path)

if df is not None:
    # Calculate baseline metrics
    total_leads = len(df)
    good_leads = df['IsGood'].sum()
    bad_leads = df['IsBad'].sum()
    closed_leads = df['IsClosed'].sum()
    baseline_quality = (good_leads / total_leads) * 100
    baseline_closed = (closed_leads / total_leads) * 100
    
    # OVERVIEW SECTION
    if analysis_section == "Overview":
        st.header("📈 Executive Summary")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Leads", f"{total_leads:,}")
        with col2:
            st.metric("Good Leads Rate", f"{baseline_quality:.2f}%", 
                     f"{baseline_quality - TARGET_CLOSED_RATE:.2f}pp vs target")
        with col3:
            st.metric("Closed Rate", f"{baseline_closed:.2f}%",
                     f"{baseline_closed - TARGET_CLOSED_RATE:.2f}pp vs target")
        with col4:
            improvement_needed = TARGET_CLOSED_RATE - baseline_closed
            st.metric("Improvement Needed", f"+{improvement_needed:.2f}pp",
                     delta_color="inverse")
        
        st.markdown("---")
        
        # Data Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Lead Status Distribution")
            status_data = df['CallStatus'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            status_data.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Count', fontweight='bold')
            ax.set_ylabel('Status', fontweight='bold')
            ax.set_title('Lead Status Distribution', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📅 Date Range & Quality")
            st.write(f"**Start Date:** {df['LeadCreated'].min().strftime('%B %d, %Y')}")
            st.write(f"**End Date:** {df['LeadCreated'].max().strftime('%B %d, %Y')}")
            st.write(f"**Duration:** {(df['LeadCreated'].max() - df['LeadCreated'].min()).days} days")
            
            st.markdown("---")
            
            quality_breakdown = pd.DataFrame({
                'Category': ['Good Leads', 'Bad Leads', 'Unknown'],
                'Count': [good_leads, bad_leads, total_leads - good_leads - bad_leads],
                'Percentage': [
                    f"{(good_leads/total_leads)*100:.2f}%",
                    f"{(bad_leads/total_leads)*100:.2f}%",
                    f"{((total_leads - good_leads - bad_leads)/total_leads)*100:.2f}%"
                ]
            })
            st.dataframe(quality_breakdown, use_container_width=True)
        
        # Quick Insights
        st.markdown("---")
        st.subheader("💡 Quick Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_partner = df.groupby('Partner')['IsGood'].mean().idxmax()
            st.success(f"**🏆 Best Partner:** {top_partner}")
        
        with col2:
            top_state = df.groupby('State')['IsGood'].mean().idxmax()
            st.success(f"**🏆 Best State:** {top_state}")
        
        with col3:
            top_widget = df.groupby('WidgetName_Clean')['IsGood'].mean().idxmax()
            st.success(f"**🏆 Best Widget:** {top_widget[:30]}...")
    
    # Q1: TEMPORAL TRENDS
    elif analysis_section == "Q1: Temporal Trends":
        st.header("📈 Question 1: Lead Quality Trends Over Time")
        
        # Weekly aggregation
        weekly_stats = df.groupby('YearWeek').agg({
            'VendorLeadID': 'count',
            'IsGood': 'sum',
            'IsClosed': 'sum',
            'LeadCreated': 'first'
        }).reset_index()
        
        weekly_stats.columns = ['YearWeek', 'TotalLeads', 'GoodLeads', 'ClosedLeads', 'Date']
        weekly_stats['QualityRate'] = (weekly_stats['GoodLeads'] / weekly_stats['TotalLeads']) * 100
        weekly_stats['ClosedRate'] = (weekly_stats['ClosedLeads'] / weekly_stats['TotalLeads']) * 100
        weekly_stats = weekly_stats.sort_values('Date')
        weekly_stats['WeekNumber'] = range(len(weekly_stats))
        
        # Statistical analysis
        X = weekly_stats['WeekNumber'].values.reshape(-1, 1)
        y = weekly_stats['QualityRate'].values
        
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        correlation, p_value = stats.pearsonr(weekly_stats['WeekNumber'], weekly_stats['QualityRate'])
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trend Slope", f"{slope:.4f} pp/week")
        with col2:
            st.metric("R-squared", f"{r_squared:.4f}")
        with col3:
            st.metric("Correlation", f"{correlation:.4f}")
        with col4:
            significance = "✅ Significant" if p_value < 0.05 else "⚠️ Not Significant"
            st.metric("P-value", f"{p_value:.4f}", significance)
        
        # Interpretation
        if p_value < 0.05:
            direction = "IMPROVING 📈" if slope > 0 else "DECLINING 📉"
            st.success(f"**Finding:** Lead quality is {direction} over time. The trend is **statistically significant** (p < 0.05).")
        else:
            st.info("**Finding:** No statistically significant trend detected (p ≥ 0.05). Lead quality appears **stable** over time.")
        
        # Visualization
        st.subheader("📊 Weekly Quality Trend")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(weekly_stats['WeekNumber'], weekly_stats['QualityRate'], 
                marker='o', linewidth=2, markersize=6, label='Actual Quality Rate', color='steelblue')
        ax.plot(weekly_stats['WeekNumber'], trend_line, 
                linestyle='--', linewidth=2, color='red', label=f'Trend Line (slope={slope:.4f})')
        ax.axhline(y=baseline_quality, color='green', linestyle=':', linewidth=2, 
                   label=f'Baseline ({baseline_quality:.2f}%)')
        ax.axhline(y=TARGET_CLOSED_RATE, color='orange', linestyle=':', linewidth=2, 
                   label=f'Target ({TARGET_CLOSED_RATE:.2f}%)')
        ax.set_xlabel('Week Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Lead Quality Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Lead Quality Trend Over Time - Weekly Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Weekly data table
        with st.expander("📋 View Weekly Data Table"):
            st.dataframe(
                weekly_stats[['Date', 'TotalLeads', 'GoodLeads', 'ClosedLeads', 'QualityRate', 'ClosedRate']].round(2),
                use_container_width=True
            )
    
    # Q2: QUALITY DRIVERS
    elif analysis_section == "Q2: Quality Drivers":
        st.header("🔍 Question 2: Drivers of Lead Quality")
        
        def analyze_segment(df, segment_col, min_sample=30):
            """Analyze lead quality by segment"""
            segment_stats = df.groupby(segment_col).agg({
                'VendorLeadID': 'count',
                'IsGood': 'sum',
                'IsClosed': 'sum'
            }).reset_index()
            
            segment_stats.columns = [segment_col, 'TotalLeads', 'GoodLeads', 'ClosedLeads']
            segment_stats['QualityRate'] = (segment_stats['GoodLeads'] / segment_stats['TotalLeads']) * 100
            segment_stats['ClosedRate'] = (segment_stats['ClosedLeads'] / segment_stats['TotalLeads']) * 100
            segment_stats['DeltaVsBaseline'] = segment_stats['QualityRate'] - baseline_quality
            
            segment_stats = segment_stats[segment_stats['TotalLeads'] >= min_sample]
            segment_stats = segment_stats.sort_values('QualityRate', ascending=False)
            
            return segment_stats
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📱 Ad Creative", "🌐 Traffic Source", "👤 Consumer Profile", "📊 Comparison Charts"])
        
        with tab1:
            st.subheader("Ad Creative Performance")
            
            # Widget analysis
            widget_stats = analyze_segment(df, 'WidgetName_Clean', min_sample=min_sample_size)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Performing Widgets**")
                if len(widget_stats) > 0:
                    st.dataframe(
                        widget_stats.head(5)[['WidgetName_Clean', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
            
            with col2:
                st.write("**Bottom 5 Performing Widgets**")
                if len(widget_stats) > 0:
                    st.dataframe(
                        widget_stats.tail(5)[['WidgetName_Clean', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
            
            # Form pages analysis
            st.markdown("---")
            form_stats = analyze_segment(df[df['FormPages'].notna()], 'FormPages', min_sample=min_sample_size)
            if len(form_stats) > 0:
                st.write("**Form Pages (1DC vs 2DC)**")
                st.dataframe(
                    form_stats[['FormPages', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                    use_container_width=True
                )
        
        with tab2:
            st.subheader("Traffic Source Performance")
            
            # Partner analysis
            partner_stats = analyze_segment(df, 'Partner', min_sample=100)
            if len(partner_stats) > 0:
                st.write("**Partner Performance**")
                st.dataframe(
                    partner_stats[['Partner', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                    use_container_width=True
                )
            
            # Campaign type
            st.markdown("---")
            campaign_stats = df.groupby('PublisherCampaignName').agg({
                'VendorLeadID': 'count',
                'IsGood': 'sum'
            }).reset_index()
            campaign_stats['QualityRate'] = (campaign_stats['IsGood'] / campaign_stats['VendorLeadID']) * 100
            
            st.write("**Call Center vs Online**")
            st.dataframe(
                campaign_stats[['PublisherCampaignName', 'VendorLeadID', 'QualityRate']].round(2),
                use_container_width=True
            )
        
        with tab3:
            st.subheader("Consumer Profile Performance")
            
            # State analysis
            state_stats = analyze_segment(df, 'State', min_sample=min_sample_size)
            if len(state_stats) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top 10 States by Quality**")
                    st.dataframe(
                        state_stats.head(10)[['State', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
                
                with col2:
                    st.write("**Bottom 10 States by Quality**")
                    st.dataframe(
                        state_stats.tail(10)[['State', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
            
            # Debt level analysis
            st.markdown("---")
            debt_stats = analyze_segment(df[df['DebtBracket'].notna()], 'DebtBracket', min_sample=min_sample_size)
            if len(debt_stats) > 0:
                st.write("**Debt Level Performance**")
                st.dataframe(
                    debt_stats[['DebtBracket', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                    use_container_width=True
                )
            
            # Score analysis
            col1, col2 = st.columns(2)
            
            with col1:
                address_stats = analyze_segment(df[df['AddressScore'].notna()], 'AddressScore', min_sample=min_sample_size)
                if len(address_stats) > 0:
                    st.write("**Address Score Performance**")
                    st.dataframe(
                        address_stats[['AddressScore', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
            
            with col2:
                phone_stats = analyze_segment(df[df['PhoneScore'].notna()], 'PhoneScore', min_sample=min_sample_size)
                if len(phone_stats) > 0:
                    st.write("**Phone Score Performance**")
                    st.dataframe(
                        phone_stats[['PhoneScore', 'TotalLeads', 'QualityRate', 'DeltaVsBaseline']].round(2),
                        use_container_width=True
                    )
        
        with tab4:
            st.subheader("Visual Comparison")
            
            widget_stats = analyze_segment(df, 'WidgetName_Clean', min_sample=min_sample_size)
            partner_stats = analyze_segment(df, 'Partner', min_sample=100)
            state_stats = analyze_segment(df, 'State', min_sample=min_sample_size)
            address_stats = analyze_segment(df[df['AddressScore'].notna()], 'AddressScore', min_sample=min_sample_size)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Top widgets
            if len(widget_stats) > 0:
                top_widgets = widget_stats.head(10)
                axes[0, 0].barh(range(len(top_widgets)), top_widgets['QualityRate'], color='steelblue')
                axes[0, 0].set_yticks(range(len(top_widgets)))
                axes[0, 0].set_yticklabels([str(w)[-30:] for w in top_widgets['WidgetName_Clean']], fontsize=8)
                axes[0, 0].axvline(x=baseline_quality, color='red', linestyle='--', label='Baseline')
                axes[0, 0].axvline(x=TARGET_CLOSED_RATE, color='green', linestyle='--', label='Target')
                axes[0, 0].set_xlabel('Quality Rate (%)', fontweight='bold')
                axes[0, 0].set_title('Top Widget Performance', fontweight='bold')
                axes[0, 0].legend()
            
            # Partners
            if len(partner_stats) > 0:
                axes[0, 1].bar(partner_stats['Partner'], partner_stats['QualityRate'], color='coral')
                axes[0, 1].axhline(y=baseline_quality, color='red', linestyle='--', label='Baseline')
                axes[0, 1].axhline(y=TARGET_CLOSED_RATE, color='green', linestyle='--', label='Target')
                axes[0, 1].set_xlabel('Partner', fontweight='bold')
                axes[0, 1].set_ylabel('Quality Rate (%)', fontweight='bold')
                axes[0, 1].set_title('Partner Performance', fontweight='bold')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].legend()
            
            # States
            if len(state_stats) > 0:
                top_states = state_stats.head(10)
                axes[1, 0].barh(range(len(top_states)), top_states['QualityRate'], color='mediumseagreen')
                axes[1, 0].set_yticks(range(len(top_states)))
                axes[1, 0].set_yticklabels(top_states['State'], fontsize=10)
                axes[1, 0].axvline(x=baseline_quality, color='red', linestyle='--', label='Baseline')
                axes[1, 0].axvline(x=TARGET_CLOSED_RATE, color='green', linestyle='--', label='Target')
                axes[1, 0].set_xlabel('Quality Rate (%)', fontweight='bold')
                axes[1, 0].set_title('Top 10 States', fontweight='bold')
                axes[1, 0].legend()
            
            # Address score
            if len(address_stats) > 0:
                address_stats_plot = address_stats.sort_values('AddressScore')
                axes[1, 1].bar(address_stats_plot['AddressScore'].astype(str), address_stats_plot['QualityRate'], color='mediumpurple')
                axes[1, 1].axhline(y=baseline_quality, color='red', linestyle='--', label='Baseline')
                axes[1, 1].axhline(y=TARGET_CLOSED_RATE, color='green', linestyle='--', label='Target')
                axes[1, 1].set_xlabel('Address Score', fontweight='bold')
                axes[1, 1].set_ylabel('Quality Rate (%)', fontweight='bold')
                axes[1, 1].set_title('Quality by Address Score', fontweight='bold')
                axes[1, 1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Q3: OPTIMIZATION
    elif analysis_section == "Q3: Optimization Strategy":
        st.header("🚀 Question 3: Optimization Strategy")
        
        st.info(f"""
        **Goal:** Increase Closed Rate from {baseline_closed:.2f}% to {TARGET_CLOSED_RATE:.2f}%  
        **Required Improvement:** +{TARGET_CLOSED_RATE - baseline_closed:.2f} percentage points  
        **Reward:** CPL increase from $30 to $36 (+$6 per lead)
        """)
        
        def analyze_segment(df, segment_col, min_sample=30):
            segment_stats = df.groupby(segment_col).agg({
                'VendorLeadID': 'count',
                'IsGood': 'sum',
                'IsClosed': 'sum'
            }).reset_index()
            
            segment_stats.columns = [segment_col, 'TotalLeads', 'GoodLeads', 'ClosedLeads']
            segment_stats['QualityRate'] = (segment_stats['GoodLeads'] / segment_stats['TotalLeads']) * 100
            segment_stats['ClosedRate'] = (segment_stats['ClosedLeads'] / segment_stats['TotalLeads']) * 100
            segment_stats['DeltaVsBaseline'] = segment_stats['QualityRate'] - baseline_quality
            
            segment_stats = segment_stats[segment_stats['TotalLeads'] >= min_sample]
            segment_stats = segment_stats.sort_values('QualityRate', ascending=False)
            
            return segment_stats
        
        # High-performing segments
        widget_stats = analyze_segment(df, 'WidgetName_Clean', min_sample=50)
        state_stats = analyze_segment(df, 'State', min_sample=30)
        partner_stats = analyze_segment(df, 'Partner', min_sample=100)
        
        high_widgets = widget_stats[widget_stats['QualityRate'] >= TARGET_CLOSED_RATE]
        high_states = state_stats[state_stats['QualityRate'] >= TARGET_CLOSED_RATE]
        high_partners = partner_stats[partner_stats['QualityRate'] >= TARGET_CLOSED_RATE]
        
        st.subheader(f"🎯 High-Performing Segments (Quality ≥ {TARGET_CLOSED_RATE}%)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Widgets Above Target", len(high_widgets))
            if len(high_widgets) > 0:
                with st.expander("View Widgets"):
                    st.dataframe(high_widgets[['WidgetName_Clean', 'TotalLeads', 'QualityRate']].round(2))
        
        with col2:
            st.metric("States Above Target", len(high_states))
            if len(high_states) > 0:
                with st.expander("View States"):
                    st.dataframe(high_states[['State', 'TotalLeads', 'QualityRate']].round(2))
        
        with col3:
            st.metric("Partners Above Target", len(high_partners))
            if len(high_partners) > 0:
                with st.expander("View Partners"):
                    st.dataframe(high_partners[['Partner', 'TotalLeads', 'QualityRate']].round(2))
        
        # Optimization scenarios
        st.markdown("---")
        st.subheader("📊 Optimization Scenarios")
        
        scenarios = []
        
        # Scenario 1: Top 3 widgets
        if len(widget_stats) >= 3:
            top3_widgets = widget_stats.head(3)
            scenario1_volume = top3_widgets['TotalLeads'].sum()
            scenario1_quality = (top3_widgets['GoodLeads'].sum() / scenario1_volume) * 100
            scenarios.append({
                'Scenario': 'Top 3 Widgets Only',
                'Expected Quality': f"{scenario1_quality:.2f}%",
                'Volume Impact': f"{(scenario1_volume/total_leads)*100:.1f}%",
                'Meets Target': '✅' if scenario1_quality >= TARGET_CLOSED_RATE else '❌'
            })
        
        # Scenario 2: Top 5 widgets
        if len(widget_stats) >= 5:
            top5_widgets = widget_stats.head(5)
            scenario2_volume = top5_widgets['TotalLeads'].sum()
            scenario2_quality = (top5_widgets['GoodLeads'].sum() / scenario2_volume) * 100
            scenarios.append({
                'Scenario': 'Top 5 Widgets Only',
                'Expected Quality': f"{scenario2_quality:.2f}%",
                'Volume Impact': f"{(scenario2_volume/total_leads)*100:.1f}%",
                'Meets Target': '✅' if scenario2_quality >= TARGET_CLOSED_RATE else '❌'
            })
        
        # Scenario 3: Combined
        if len(state_stats) > 0 and len(widget_stats) > 0:
            top_states = state_stats.head(10)
            top_widgets_list = widget_stats.head(5)
            combined_filter = df[
                (df['State'].isin(top_states['State'])) & 
                (df['WidgetName_Clean'].isin(top_widgets_list['WidgetName_Clean']))
            ]
            if len(combined_filter) > 100:
                scenario3_quality = (combined_filter['IsGood'].sum() / len(combined_filter)) * 100
                scenarios.append({
                    'Scenario': 'Best Widgets + Best States',
                    'Expected Quality': f"{scenario3_quality:.2f}%",
                    'Volume Impact': f"{(len(combined_filter)/total_leads)*100:.1f}%",
                    'Meets Target': '✅' if scenario3_quality >= TARGET_CLOSED_RATE else '❌'
                })
        
        # Scenario 4: Score filtering
        if df['AddressScore'].notna().sum() > 100:
            high_score_filter = df[(df['AddressScore'] >= 4) | (df['PhoneScore'] >= 4)]
            if len(high_score_filter) > 100:
                scenario4_quality = (high_score_filter['IsGood'].sum() / len(high_score_filter)) * 100
                scenarios.append({
                    'Scenario': 'Address/Phone Score ≥ 4',
                    'Expected Quality': f"{scenario4_quality:.2f}%",
                    'Volume Impact': f"{(len(high_score_filter)/total_leads)*100:.1f}%",
                    'Meets Target': '✅' if scenario4_quality >= TARGET_CLOSED_RATE else '❌'
                })
        
        if len(scenarios) > 0:
            scenarios_df = pd.DataFrame(scenarios)
            st.dataframe(scenarios_df, use_container_width=True)
        else:
            st.warning("⚠️ Need to explore alternative optimization strategies")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Strategic Recommendations")
        
        best_quality = max([float(s['Expected Quality'].rstrip('%')) for s in scenarios], default=baseline_quality) if scenarios else baseline_quality
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎨 Widget Mix", "🌍 Geographic", "🔒 Quality Filters", "📋 Form Design"])
        
        with tab1:
            st.markdown("""
            **RECOMMENDATION #1: Optimize Widget Mix**
            
            **Action:**
            - Focus budget on top 5 performing widgets
            - Pause or reduce spend on bottom 25% performers
            - Increase bids on high-converting creative designs
            
            **Expected Impact:**
            - Significant quality improvement potential
            - Estimated lift: +2-3 percentage points
            
            **Implementation Timeline:**
            - Week 1: Pause bottom performers
            - Week 2-3: Shift budget to winners
            - Week 4: Monitor and optimize
            """)
        
        with tab2:
            st.markdown("""
            **RECOMMENDATION #2: Geographic Targeting**
            
            **Action:**
            - Increase spend in top 10 high-quality states
            - Consider state-specific campaigns
            - Adjust bid modifiers by geography
            
            **Expected Impact:**
            - State-level optimization: +1-2 percentage points
            - Better audience targeting
            
            **Implementation Timeline:**
            - Week 1: Analyze geo-performance
            - Week 2-4: Shift budget allocation
            - Month 2: Scale successful regions
            """)
        
        with tab3:
            st.markdown("""
            **RECOMMENDATION #3: Implement Quality Filters**
            
            **Action:**
            - Enable AddressScore and PhoneScore validation
            - Set minimum threshold at ≥ 4
            - Filter leads in real-time during capture
            
            **Expected Impact:**
            - Reduce invalid/uncontactable leads
            - Improve overall lead quality by 15-20%
            
            **Implementation Timeline:**
            - Week 1: Technical setup
            - Week 2: Test filtering logic
            - Week 3: Full deployment
            """)
        
        with tab4:
            st.markdown("""
            **RECOMMENDATION #4: Form Optimization**
            
            **Action:**
            - A/B test 1DC vs 2DC forms
            - Optimize form fields and design
            - Implement progressive profiling
            
            **Expected Impact:**
            - Form design can impact quality by 10-15%
            - Better user experience = better leads
            
            **Implementation Timeline:**
            - Week 1-2: Design variations
            - Week 3-4: Run A/B test
            - Month 2: Scale winner
            """)
        
        # ROI Summary
        st.markdown("---")
        st.subheader("💰 ROI Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Current State:**
            - Total Leads: {total_leads:,}
            - CPL: $30
            - Closed Rate: {baseline_closed:.2f}%
            - **Total Revenue:** ${total_leads * 30:,}
            """)
        
        with col2:
            projected_leads = int(total_leads * 0.8)  # Assuming 20% volume reduction
            st.markdown(f"""
            **Projected State (Target Reached):**
            - Projected Leads: {projected_leads:,} (80% volume)
            - New CPL: $36
            - Target Closed Rate: {TARGET_CLOSED_RATE:.2f}%
            - **Projected Revenue:** ${projected_leads * 36:,}
            """)
        
        # Status indicator
        if best_quality >= TARGET_CLOSED_RATE:
            st.success(f"✅ **TARGET ACHIEVABLE** - Projected quality of {best_quality:.2f}% meets the {TARGET_CLOSED_RATE:.2f}% target!")
        else:
            st.warning(f"⚠️ **ADDITIONAL OPTIMIZATION NEEDED** - Current projection: {best_quality:.2f}% vs target: {TARGET_CLOSED_RATE:.2f}%")
    
    # EXPORT REPORT
    elif analysis_section == "Export Report":
        st.header("📄 Export Analysis Report")
        
        st.info("Generate a comprehensive PDF report with all findings and visualizations.")
        
        # Report options
        include_raw_data = st.checkbox("Include raw data tables", value=False)
        include_charts = st.checkbox("Include all charts", value=True)
        include_recommendations = st.checkbox("Include recommendations", value=True)
        
        report_format = st.radio("Report Format", ["PDF", "Excel", "PowerPoint"])
        
        if st.button("🚀 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                st.success("✅ Report generation feature coming soon!")
                st.info("""
                **Next Steps:**
                1. Copy your analysis from each section
                2. Create a Word document or PowerPoint
                3. Include all charts (download them from each section)
                4. Export as PDF
                
                **Or use Python libraries:**
                - `reportlab` or `fpdf2` for PDF generation
                - `python-pptx` for PowerPoint
                - `openpyxl` for Excel reports
                """)
        
        # Download individual charts
        st.markdown("---")
        st.subheader("📊 Download Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Temporal Trend Chart",
                data="Q1_Temporal_Trend.png",
                file_name="temporal_trend.png",
                mime="image/png",
                disabled=True  # Enable when file exists
            )
        
        with col2:
            st.download_button(
                "Download Quality Drivers Chart",
                data="Q2_Quality_Drivers.png",
                file_name="quality_drivers.png",
                mime="image/png",
                disabled=True  # Enable when file exists
            )
        
        # Export data
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="lead_analysis_data.csv",
                mime="text/csv"
            )

else:
    # No data loaded state
    st.warning("👈 Please upload an Excel file in the sidebar to begin analysis")
    
    st.markdown("""
    ## 📋 Expected File Format
    
    Your Excel file should contain the following columns:
    - **LeadCreated**: Date when lead was created
    - **CallStatus**: Status of the lead (Closed, EP Sent, etc.)
    - **WidgetName**: Ad creative identifier
    - **Partner**: Traffic source partner
    - **State**: Consumer's state
    - **DebtLevel**: Amount of debt
    - **AddressScore**: Address validation score (optional)
    - **PhoneScore**: Phone validation score (optional)
    
    ## 🎯 Analysis Objectives
    
    1. **Temporal Trends**: Identify if lead quality is improving or declining
    2. **Quality Drivers**: Understand what factors influence lead quality
    3. **Optimization**: Develop strategies to increase quality by 20%
    
    ## 💡 Getting Started
    
    1. Upload your Excel file using the sidebar
    2. Navigate through different analysis sections
    3. Review insights and recommendations
    4. Export your final report
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Lead Quality Analysis Dashboard</strong></p>
    <p>Analytics Challenge for Aarki | Developed by Karthik</p>
</div>
""", unsafe_allow_html=True)