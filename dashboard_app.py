import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Matplotlib Installation Check (REMOVE AFTER VERIFICATION) ---
import sys
try:
    import matplotlib
    st.success(f"Matplotlib version: {matplotlib.__version__} loaded from: {os.path.dirname(matplotlib.__file__)}")
except ImportError:
    st.error("Matplotlib is not installed or cannot be found. Please run 'pip install matplotlib' in your terminal.")
    st.stop()
# ------------------------------------------------------------------

# --- Helper function for K formatting (moved here to be accessible for all graphs)
def format_k(num):
    if abs(num) >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    if abs(num) >= 1000:
        return f'{num/1000:.1f}K'
    return str(int(num)) # Ensure displayed as integer if not thousands

# --- Helper function to normalize ingredient names (moved to global scope) ---
def normalize_ingredient(ingredient):
    ingredient = ingredient.strip().lower()
    # Basic normalization: remove trailing 's' for plurals if not part of original word (e.g., 'cheese' vs 'cheeses')
    if ingredient.endswith('s') and not ingredient.endswith('ss') and len(ingredient) > 2 and ingredient not in ['olives', 'peppers', 'mushrooms', 'tomatoes', 'onions']:
        ingredient = ingredient[:-1]
    # More specific normalization rules can be added here
    if 'bell pepper' in ingredient:
        return 'pepper'
    if 'kalamata' in ingredient:
        return 'kalamata olive'
    if 'green olive' in ingredient:
        return 'green olive'
    if 'red onion' in ingredient:
        return 'onion'
    if 'white onion' in ingredient:
        return 'onion'
    if 'red pepper' in ingredient:
        return 'pepper'
    if 'green pepper' in ingredient:
        return 'pepper'
    if 'jalapeno' in ingredient:
        return 'jalapeno'
    if 'chorizo' in ingredient:
        return 'chorizo'
    if 'soppressata' in ingredient:
        return 'soppressata'
    return ingredient

# General Streamlit page settings
st.set_page_config(layout="wide", page_title="דשבורד מכירות פיצה")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Align main headings (h1, h2) to the right */
    .stApp h1, .stApp h2 {
        text-align: right;
        direction: rtl;
    }

    /* Align sidebar headings to the right */
    [data-testid="stSidebarHeader"] > div { /* Main sidebar title */
        text-align: right;
        direction: rtl;
        width: 100%;
    }
    [data-testid="stSidebar"] h2 { /* Subheader in sidebar */
        text-align: right;
        direction: rtl;
        width: 100%;
    }

    /* Styling for KPI boxes - applied directly to st.metric */
    [data-testid="stMetric"] {
        border: 1px solid #e0e0e0; /* Subtle border color */
        border-radius: 8px; /* Rounded corners */
        padding: 15px; /* Inner padding */
        margin-bottom: 15px; /* Bottom margin between boxes */
        background-color: #f9f9f9; /* Light background for the box */
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1); /* Subtle shadow */
        display: flex; /* Use flexbox to center content */
        flex-direction: column;
        justify-content: center; /* Center vertically */
        align-items: center; /* Center horizontally */
        height: 100%; /* Ensure full height in column */
        text-align: center; /* Center text within the box */
        direction: rtl; /* Right-to-left direction for box content */
    }

    /* Font and color adjustments within the metric */
    [data-testid="stMetric"] > div:first-child { /* Metric label */
        font-size: 1.1em;
        font-weight: bold;
        color: #555;
    }
    [data-testid="stMetric"] > div:nth-child(2) { /* Metric numerical value */
        font-size: 2.2em;
        font-weight: bold;
        color: #4CAF50; /* Green color for emphasis */
    }
    [data-testid="stMetric"] > div:nth-child(3) { /* Difference (if any) */
        font-size: 0.9em;
        color: #777;
    }

    /* Style for the new table */
    .stDataFrame {
        direction: rtl; /* Ensure table content is RTL */
    }
    .stDataFrame thead th {
        text-align: right !important; /* Align headers to the right */
    }
    .stDataFrame tbody tr td {
        text-align: center; /* Center body cell content */
    }
    .stDataFrame tbody tr:last-child { /* Style for the total row */
        font-weight: bold;
        background-color: #e6ffe6; /* Light green background for total row */
    }
    .stDataFrame tbody tr:last-child td {
        border-top: 2px solid #ccc; /* Top border for total row */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Path to the unified file
# Removed the absolute path and made it relative to the script's location
data_file_name = 'full_pizza_sales_data.csv'
full_data_path = os.path.join(os.path.dirname(__file__), data_file_name)


# --- Load Data (cached) ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Ensure datetime columns are in datetime format
    df['order_datetime'] = pd.to_datetime(df['order_datetime'], format="%d/%m/%Y %H:%M")
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")

    # Create year and quarter columns for filtering and analysis
    df['year'] = df['order_datetime'].dt.year
    df['quarter'] = df['order_datetime'].dt.quarter
    df['year_quarter'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)

    return df

try:
    df = load_data(full_data_path)
    st.success(f"נתונים נטענו בהצלחה מ- {data_file_name}")
except FileNotFoundError:
    st.error(f"שגיאה: הקובץ {data_file_name} לא נמצא בנתיב {full_data_path}. אנא ודא שהרצת את הסקריפט לאיחוד הנתונים ושהקובץ נמצא באותה תיקייה כמו הקוד ב-GitHub.")
    st.stop() # Stop app execution if file not found

# --- Sidebar Filters ---
st.sidebar.header("פילטרים")

# 1. Date Range Filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "בחר טווח תאריכים",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Data filtered by date (first step)
filtered_df_by_date = df.copy()
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    filtered_df_by_date = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
else:
    st.sidebar.warning("אנא בחר טווח תאריכים (תאריך התחלה וסיום).")

# 2. Quarter Filter (years and quarters available from date-filtered data)
st.sidebar.markdown("---") # Visual separator line
st.sidebar.subheader("סינון לפי רבעון")

# Get unique years from the date-filtered data
unique_years = sorted(filtered_df_by_date['year'].unique(), reverse=True)
selected_year = st.sidebar.selectbox("בחר שנה", options=['הכל'] + list(unique_years))

# If a specific year is selected, show quarters for it
selected_quarter = 'הכל'
if selected_year != 'הכל':
    unique_quarters_in_year = sorted(filtered_df_by_date[filtered_df_by_date['year'] == selected_year]['quarter'].unique())
    # Convert quarter numbers to Q1, Q2 etc. for display
    quarter_options_display = [f"Q{q}" for q in unique_quarters_in_year]
    selected_quarter_display = st.sidebar.selectbox("בחר רבעון", options=['הכל'] + quarter_options_display)

    if selected_quarter_display != 'הכל':
        selected_quarter = int(selected_quarter_display[1:]) # Extract number from 'Q'
else:
    # If 'All' is selected for the year, no need to show specific quarters
    st.sidebar.selectbox("בחר רבעון", options=['הכל'], disabled=True)

# --- Final DataFrame Filtering ---
# Start with the date-filtered DataFrame
filtered_df = filtered_df_by_date.copy()

# If a specific year is selected, also filter by year
if selected_year != 'הכל':
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

# If a specific quarter is selected (not 'All'), also filter by quarter
if selected_quarter != 'הכל':
    filtered_df = filtered_df[filtered_df['quarter'] == selected_quarter]

# --- Dashboard Title ---
st.markdown("<h1 style='text-align: right; direction: rtl;'>🍕 דשבורד מכירות פיצה</h1>", unsafe_allow_html=True)

# --- Display KPI Metrics ---
st.markdown("<h2 style='text-align: right; direction: rtl;'>מדדי ביצועים מרכזיים (KPIs)</h2>", unsafe_allow_html=True)

# Calculate KPIs on the final filtered data
total_revenue = filtered_df['item_total_price'].sum()
total_orders = filtered_df['order_id'].nunique() # Number of unique orders
total_pizzas_sold = filtered_df['quantity'].sum()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="סה\"כ הכנסות", value=f"${total_revenue:,.2f}")
with col2:
    st.metric(label="מספר הזמנות", value=f"{total_orders:,}")
with col3:
    st.metric(label="פיצות נמכרות", value=f"{total_pizzas_sold:,}")
with col4:
    st.metric(label="ממוצע הזמנה", value=f"${avg_order_value:,.2f}")

# --- Graphs ---
# --- Line Chart - Daily/Monthly Revenue ---
st.write("---") # Visual separator line

# Toggle to choose between daily and monthly view
view_monthly = st.toggle("הצג נתונים חודשיים", value=True)


if view_monthly:
    # Group by month
    monthly_revenue = filtered_df.groupby(pd.Grouper(key='order_datetime', freq='M'))['item_total_price'].sum().reset_index()
    monthly_revenue.columns = ['Date', 'Revenue']
    chart_data_revenue = monthly_revenue
    chart_title_revenue = 'מגמת הכנסות חודשית'
else:
    # Group by day
    daily_revenue = filtered_df.groupby(pd.to_datetime(filtered_df['order_datetime'].dt.date))['item_total_price'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    chart_data_revenue = daily_revenue
    chart_title_revenue = 'מגמת הכנסות יומית'

# Create the graph with the appropriate data
fig_revenue_trend_final = px.line(
    chart_data_revenue,
    x='Date',
    y='Revenue',
    title=chart_title_revenue,
    labels={'Date': 'תאריך', 'Revenue': 'הכנסה ($)'},
    template="plotly_white"
)
fig_revenue_trend_final.update_layout(title_x=0.95, title_xanchor='right')

# Create annotations with K format
annotations_revenue = []
# Iterate over data and add an annotation for each point
for index, row in chart_data_revenue.iterrows():
    annotations_revenue.append(dict(
        x=row['Date'],
        y=row['Revenue'],
        xref='x', yref='y',
        text=f"${format_k(row['Revenue'])}", # K format with dollar sign
        showarrow=False,
        yshift=10, # Shift up above the point
        font=dict(size=9, color='black')
    ))

fig_revenue_trend_final.update_layout(annotations=annotations_revenue)

st.plotly_chart(fig_revenue_trend_final, use_container_width=True)

# --- NEW: Table: Distribution of orders by number of pizzas ---
st.write("---")
st.subheader("התפלגות הזמנות לפי כמות פיצות")

# Group by order_id and count total quantity per order
pizzas_per_order = filtered_df.groupby('order_id')['quantity'].sum().reset_index()
pizzas_per_order.columns = ['order_id', 'total_pizzas_in_order']

# Count how many orders have 1 pizza, 2 pizzas, etc.
pizzas_per_order_distribution = pizzas_per_order['total_pizzas_in_order'].value_counts().reset_index()
pizzas_per_order_distribution.columns = ['כמות פיצות', 'כמות הזמנות']
pizzas_per_order_distribution = pizzas_per_order_distribution.sort_values(by='כמות פיצות', ascending=True)

# Calculate frequency
total_orders_in_distribution = pizzas_per_order_distribution['כמות הזמנות'].sum()
pizzas_per_order_distribution['שכיחות'] = (pizzas_per_order_distribution['כמות הזמנות'] / total_orders_in_distribution * 100).round(2)

# Add a total row
total_row_dist = pd.DataFrame({
    'כמות פיצות': ['סה"כ'],
    'כמות הזמנות': [pizzas_per_order_distribution['כמות הזמנות'].sum()],
    'שכיחות': ['100.00'] # Store as string for easy concat, will add '%' later
})

# Concatenate the distribution with the total row
pizzas_per_order_distribution_final = pd.concat([pizzas_per_order_distribution, total_row_dist], ignore_index=True)

# Format 'שכיחות' column to add '%'
pizzas_per_order_distribution_final['שכיחות'] = pizzas_per_order_distribution_final['שכיחות'].astype(str) + '%'


st.dataframe(pizzas_per_order_distribution_final, hide_index=True, use_container_width=True)


# --- Bar Chart - Top 10 Bestselling Pizzas (stacked by size) ---
st.write("---")
st.subheader("עשר הפיצות הנמכרות ביותר (כמות נמכרת לפי גודל)")

# 1. Calculate TOP 10 total pizzas (regardless of size) and maintain order
# Use this for both sorting and summary labels
total_quantities_per_pizza = filtered_df.groupby('name')['quantity'].sum().nlargest(10)
# Here's the fix: we want the highest quantity pizzas to be at the top,
# so the categoryarray order should be reverse of the nlargest order
top_10_pizza_names_ordered = total_quantities_per_pizza.index.tolist()[::-1]

top_10_pizza_sums = total_quantities_per_pizza.to_dict() # Dictionary of pizza name: total quantity

# 2. Filter the original DataFrame to include only these pizzas
filtered_top_10_df = filtered_df[filtered_df['name'].isin(top_10_pizza_names_ordered)].copy()

# 3. Group again by name and size for the stacked chart
top_10_pizzas_by_size = filtered_top_10_df.groupby(['name', 'size'])['quantity'].sum().reset_index()

# Create the chart
fig_top_pizzas_stacked = px.bar(
    top_10_pizzas_by_size,
    x='quantity',
    y='name',
    color='size', # Color by size
    orientation='h',
    title='כמות נמכרת לפי גודל עבור עשר הפיצות המובילות', # Chart title
    labels={'quantity': 'כמות נמכרת', 'name': 'שם הפיצה', 'size': 'גודל פיצה'},
    template="plotly_white",
    barmode='stack', # Stacked mode
    text='quantity' # Add labels inside stacked bars
)

# Set Y-axis order by total quantity of each pizza (from largest to smallest, but in reverse order for categoryarray)
fig_top_pizzas_stacked.update_layout(
    yaxis={'categoryorder':'array', 'categoryarray': top_10_pizza_names_ordered}
)

fig_top_pizzas_stacked.update_layout(title_x=0.95, title_xanchor='right')

# Adjust text position on columns (within each stacked part)
fig_top_pizzas_stacked.update_traces(texttemplate='%{text}', textposition='auto')

# Add summary labels outside the columns
annotations = []
# Use the original list of total_quantities_per_pizza.index.tolist() because the order in this loop is largest to smallest
for name in total_quantities_per_pizza.index.tolist():
    total_quantity = top_10_pizza_sums[name]
    # Annotation position: X-axis at the end of the bar, Y-axis as the pizza name
    annotations.append(dict(
        xref='x', yref='y',
        x=total_quantity + max(total_quantities_per_pizza) * 0.02, # Position slightly to the right of the end
        y=name,
        text=format_k(total_quantity), # K format
        showarrow=False,
        xanchor='left', # Text alignment
        font=dict(size=9, color='black') # Font size for summary label
    ))

fig_top_pizzas_stacked.update_layout(annotations=annotations)

st.plotly_chart(fig_top_pizzas_stacked, use_container_width=True)

# --- Bar Chart - 10 Least Selling Pizzas ---
st.write("---")
least_10_pizzas = filtered_df.groupby('name')['quantity'].sum().sort_values(ascending=True).head(10).reset_index()
fig_least_pizzas = px.bar(
    least_10_pizzas,
    x='quantity',
    y='name',
    orientation='h',
    title='עשר הפיצות הנמכרות הכי פחות',
    labels={'quantity': 'כמות נמכרת', 'name': 'שם הפיצה'},
    template="plotly_white",
    text='quantity'
)
fig_least_pizzas.update_layout(yaxis={'categoryorder':'total descending'}) # Here descending because we want the least sold at the bottom
fig_least_pizzas.update_layout(title_x=0.95, title_xanchor='right')
fig_least_pizzas.update_traces(texttemplate='%{text:.2s}', textposition='outside')
st.plotly_chart(fig_least_pizzas, use_container_width=True)


# --- NEW SECTION: Ingredient Analysis ---
st.write("---")
st.markdown("<h2 style='text-align: right; direction: rtl;'>ניתוח מרכיבי פיצה</h2>", unsafe_allow_html=True)


# --- Function to analyze ingredients ---
@st.cache_data # Cache this function as well, as it can be computationally intensive
def analyze_ingredients(df_to_analyze):
    if 'ingredients' not in df_to_analyze.columns:
        st.warning("עמודת 'ingredients' לא נמצאה בנתונים. אנא ודא שהיא קיימת בקובץ המאוחד.")
        return pd.DataFrame(), [] # Return empty DataFrame and empty list

    # Prepare ingredients list for each unique pizza type
    unique_pizza_types = df_to_analyze[['pizza_type_id', 'ingredients']].drop_duplicates()
    
    pizza_ingredients_map = {}
    for idx, row in unique_pizza_types.iterrows():
        if pd.isna(row['ingredients']):
            pizza_ingredients_map[row['pizza_type_id']] = []
            continue
        # Normalize and split ingredients for each pizza type
        normalized_ingredients_list = [normalize_ingredient(item) for item in row['ingredients'].split(',')]
        pizza_ingredients_map[row['pizza_type_id']] = normalized_ingredients_list

    # Create a DataFrame where each row is an order_detail + one ingredient
    df_with_normalized_ingredients = df_to_analyze.copy()
    df_with_normalized_ingredients['normalized_ingredients_list'] = df_with_normalized_ingredients['pizza_type_id'].map(pizza_ingredients_map)

    # Explode the dataframe so each row represents one ingredient within an order_detail
    df_exploded_ingredients = df_with_normalized_ingredients.explode('normalized_ingredients_list')

    # Group by the normalized ingredient and sum the item_total_price and quantity
    accurate_ingredient_summary = df_exploded_ingredients.groupby('normalized_ingredients_list').agg(
        total_revenue=('item_total_price', 'sum'),
        total_quantity=('quantity', 'sum') # Sum of quantities where this ingredient appears
    ).reset_index()
    accurate_ingredient_summary = accurate_ingredient_summary.rename(columns={'normalized_ingredients_list': 'ingredient'})

    ingredient_summary = accurate_ingredient_summary.sort_values(by='total_revenue', ascending=False)

    return ingredient_summary, list(ingredient_summary.sort_values(by='total_quantity', ascending=False)['ingredient'].head(10)) # Also return top 10 by quantity


# --- Function to create co-occurrence matrix with enhancements and styling ---
def create_co_occurrence_matrix_styled(df_to_analyze, top_ingredients, total_pizzas_overall):
    if not top_ingredients:
        return pd.DataFrame().style

    unique_pizza_types = df_to_analyze[['pizza_type_id', 'ingredients']].drop_duplicates()
    
    pizza_ingredients_map_normalized = {}
    for idx, row in unique_pizza_types.iterrows():
        if pd.isna(row['ingredients']):
            pizza_ingredients_map_normalized[row['pizza_type_id']] = []
            continue
        normalized_list = []
        for item in row['ingredients'].split(','):
            norm_ing = normalize_ingredient(item)
            if norm_ing in top_ingredients: # Only consider top ingredients for the matrix
                normalized_list.append(norm_ing)
        pizza_ingredients_map_normalized[row['pizza_type_id']] = sorted(list(set(normalized_list))) # Use set to avoid duplicates from normalization

    # Step 1: Initialize co_occurrence_abs_counts with 0s
    co_occurrence_abs_counts = pd.DataFrame(0, index=top_ingredients, columns=top_ingredients)

    # Step 2: Populate co_occurrence_abs_counts for all pairs (including self-pairs for diagonal)
    for index, row in df_to_analyze.iterrows():
        pizza_type_id = row['pizza_type_id']
        pizza_ingredients = pizza_ingredients_map_normalized.get(pizza_type_id, [])
        pizza_quantity = row['quantity']

        if not pizza_ingredients:
            continue
        
        # Count co-occurrences for all pairs including self-pairs
        for i in range(len(pizza_ingredients)):
            for j in range(i, len(pizza_ingredients)): # Start j from i to include self-pairs (diagonal)
                ing1 = pizza_ingredients[i]
                ing2 = pizza_ingredients[j]
                
                co_occurrence_abs_counts.loc[ing1, ing2] += pizza_quantity
                if ing1 != ing2: # For off-diagonal, ensure symmetry
                    co_occurrence_abs_counts.loc[ing2, ing1] += pizza_quantity

    # Ensure correct order for columns and index based on top_ingredients
    co_occurrence_abs_counts = co_occurrence_abs_counts.loc[top_ingredients, top_ingredients]

    # Step 3: Get the total quantity of pizzas for each ingredient (for percentage divisor)
    # This is effectively the diagonal values, which represent the total count of pizzas for that single ingredient
    # We'll use these counts as the base for row-wise percentages
    row_divisors = pd.Series(0, index=top_ingredients, dtype=float)
    for ing in top_ingredients:
        row_divisors[ing] = co_occurrence_abs_counts.loc[ing, ing] # The diagonal already holds the correct total for a single ingredient

    # Step 4: Convert to percentages based on the row's ingredient total
    percentage_df_values = pd.DataFrame(index=top_ingredients, columns=top_ingredients, dtype=float)
    for ing_row in top_ingredients:
        divisor = row_divisors[ing_row]
        
        if divisor > 0:
            percentage_df_values.loc[ing_row] = (co_occurrence_abs_counts.loc[ing_row] / divisor) * 100
        else:
            percentage_df_values.loc[ing_row] = 0.0 # If divisor is 0, all percentages are 0 for this row

    # Step 5: Add 'סה"כ פיצות לשורה' column (absolute quantities from ingredient_summary_df)
    # Re-fetch or ensure ingredient_summary is up-to-date with current filters
    ingredient_summary_for_totals, _ = analyze_ingredients(df_to_analyze)
    total_q_per_ingredient_map = ingredient_summary_for_totals.set_index('ingredient')['total_quantity']

    final_df_display = percentage_df_values.copy()
    final_df_display['סה"כ פיצות לשורה'] = final_df_display.index.map(total_q_per_ingredient_map).round(0).astype(int)

    # Step 6: Create the total row for the bottom ('סה"כ פיצות לעמודה')
    total_row_data = {}
    for col in top_ingredients: # For all ingredient columns
        total_row_data[col] = total_q_per_ingredient_map.get(col, 0) # Use actual totals from summary
    
    # The grand total at the intersection of totals should be the total pizzas overall (from KPI)
    total_row_data['סה"כ פיצות לשורה'] = total_pizzas_overall 

    total_row_df = pd.DataFrame([total_row_data], index=['סה"כ פיצות לעמודה'])
    total_row_df = total_row_df.reindex(columns=final_df_display.columns, fill_value=0) # Ensure columns match
    total_row_df = total_row_df.round(0).astype(int)

    # Step 7: Concatenate the percentage_df with the total row
    final_df = pd.concat([final_df_display, total_row_df])

    # Step 8: Apply styling
    percentage_cols_for_styling = top_ingredients # These are the ingredient columns
    
    # Max value for gradient scaling should consider all percentage values (0-100)
    # The max value could theoretically be 100 if all pizzas with one ingredient also have another
    max_percentage_val = final_df.loc[top_ingredients, percentage_cols_for_styling].max().max()
    if max_percentage_val == 0: # Avoid division by zero or weird gradients if all are zero
        max_percentage_val = 1 # Set a default to ensure some gradient range

    styled_df = final_df.style.background_gradient(
        cmap='Greens',
        axis=None, # Apply gradient across the specified subset of cells
        subset=pd.IndexSlice[top_ingredients, percentage_cols_for_styling], # Only apply to the main grid of percentages
        vmin=0,
        vmax=max_percentage_val # Scale based on the maximum percentage in the grid (can be 100)
    ).format(
        formatter="{:.2f}%", # Format percentage cells
        subset=pd.IndexSlice[top_ingredients, percentage_cols_for_styling]
    ).format(
        formatter="{:,.0f}", # Format total column cells as integers (no decimals, comma separator)
        subset=pd.IndexSlice[top_ingredients, ['סה"כ פיצות לשורה']]
    ).format(
        formatter="{:,.0f}", # Format the entire total row (all cells) as integers
        subset=pd.IndexSlice[['סה"כ פיצות לעמודה'], final_df.columns.tolist()]
    ).set_properties(
        subset=pd.IndexSlice[top_ingredients, percentage_cols_for_styling], **{'text-align': 'center'}
    ).set_properties(
        subset=pd.IndexSlice[top_ingredients, ['סה"כ פיצות לשורה']],
        **{'text-align': 'center', 'font-weight': 'bold', 'background-color': '#e6ffe6'}
    ).set_properties(
        subset=pd.IndexSlice[['סה"כ פיצות לעמודה'], final_df.columns.tolist()],
        **{'text-align': 'center', 'font-weight': 'bold', 'background-color': '#e6ffe6'}
    )
            
    return styled_df


# Get the ingredient analysis summary for the current filtered data
ingredient_summary_df, top_10_ingredients_list = analyze_ingredients(filtered_df)

if not ingredient_summary_df.empty:
    # --- Top 10 Ingredients by Revenue ---
    st.subheader("עשרת המרכיבים המובילים לפי הכנסה")
    top_10_ingredients_revenue = ingredient_summary_df.sort_values(by='total_revenue', ascending=False).head(10)
    fig_top_ingredients_revenue = px.bar(
        top_10_ingredients_revenue,
        x='total_revenue',
        y='ingredient',
        orientation='h',
        title='הכנסות למרכיבים מובילים',
        labels={'total_revenue': 'סה"כ הכנסה ($)', 'ingredient': 'מרכיב'},
        template="plotly_white",
        text='total_revenue'
    )
    fig_top_ingredients_revenue.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_top_ingredients_revenue.update_layout(title_x=0.95, title_xanchor='right')
    fig_top_ingredients_revenue.update_traces(texttemplate='$%{text:,.2s}', textposition='outside')
    st.plotly_chart(fig_top_ingredients_revenue, use_container_width=True)

    # --- Top 10 Ingredients by Quantity Sold ---
    st.subheader("עשרת המרכיבים המובילים לפי כמות נמכרת")
    top_10_ingredients_quantity = ingredient_summary_df.sort_values(by='total_quantity', ascending=False).head(10)
    fig_top_ingredients_quantity = px.bar(
        top_10_ingredients_quantity,
        x='total_quantity',
        y='ingredient',
        orientation='h',
        title='כמות נמכרת למרכיבים מובילים',
        labels={'total_quantity': 'סה"כ כמות נמכרת', 'ingredient': 'מרכיב'},
        template="plotly_white",
        text='total_quantity'
    )
    fig_top_ingredients_quantity.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_top_ingredients_quantity.update_layout(title_x=0.95, title_xanchor='right')
    fig_top_ingredients_quantity.update_traces(texttemplate='%{text:,.0f}', textposition='outside') # No decimals for quantity
    st.plotly_chart(fig_top_ingredients_quantity, use_container_width=True)

    # --- NEW: Co-occurrence Matrix of Top 10 Ingredients ---
    st.write("---")
    st.subheader("מטריצת שילובים של עשרת המרכיבים המובילים (לפי כמות פיצות)")

    if top_10_ingredients_list: # Check if the list is not empty
        # Call the new styled function and pass total_pizzas_sold
        co_occurrence_matrix_styled_df = create_co_occurrence_matrix_styled(
            filtered_df, top_10_ingredients_list, total_pizzas_sold
        )
        st.write(co_occurrence_matrix_styled_df) # Use st.write for styled dataframes
        st.info("המספרים במטריצה מציינים את אחוז יחידות הפיצה המכילות את המרכיב בשורה, אשר מכילות גם את המרכיב בעמודה.")
        st.markdown("""
            <details>
                <summary>הסבר על מטריצת השילובים</summary>
                <p style="text-align: right; direction: rtl;">
                    המטריצה מציגה את האחוזים של שילובים שונים של מרכיבים:
                    <ul>
                        <li style="text-align: right; direction: rtl;"><strong>בתאים האלכסוניים (היכן ששם המרכיב בשורה זהה לשם המרכיב בעמודה):</strong> תמיד יהיה <strong>100%</strong>, מכיוון שאנו בוחנים את כלל הפיצות המכילות את המרכיב הזה בשורה.</li>
                        <li style="text-align: right; direction: rtl;"><strong>בתאים מחוץ לאלכסון:</strong> מופיע אחוז יחידות הפיצה המכילות את המרכיב <b>שבשורה</b>, אשר מכילות <b>גם</b> את המרכיב שבראש העמודה. לדוגמה, אם בשורת "שום" בתא של "בצל" מופיע 30%, זה אומר ש-30% מהפיצות המכילות שום הכילו גם בצל.</li>
                        <li style="text-align: right; direction: rtl;"><strong>עמודה "סה"כ פיצות לשורה":</strong> סך הכמות המוחלטת של יחידות הפיצה שנמכרו המכילות את המרכיב בשורה זו.</li>
                        <li style="text-align: right; direction: rtl;"><strong>שורה "סה"כ פיצות לעמודה":</strong> סך הכמות המוחלטת של יחידות הפיצה שנמכרו המכילות את המרכיב בעמודה זו.</li>
                    </ul>
                    זהו כלי שימושי לזיהוי באיזו תדירות מרכיבים מסוימים נוטים להופיע יחד עם מרכיב אחר. ככל שהרקע ירוק יותר, כך האחוז גבוה יותר.
                </p>
            </details>
        """, unsafe_allow_html=True)

    else:
        st.info("אין מספיק מרכיבים מובילים ליצירת מטריצת שילובים.")

else:
    st.info("אין נתונים לניתוח מרכיבים עבור הבחירות הנוכחיות.")


# Bar Chart - Revenue by Category
st.write("---")
revenue_by_category = filtered_df.groupby('category')['item_total_price'].sum().sort_values(ascending=False).reset_index()
fig_revenue_category = px.bar(
    revenue_by_category,
    x='category',
    y='item_total_price',
    title='הכנסות לפי קטגוריית פיצה',
    labels={'category': 'קטגוריה', 'item_total_price': 'הכנסה ($)'},
    template="plotly_white",
    text='item_total_price'
)
fig_revenue_category.update_layout(title_x=0.95, title_xanchor='right')
fig_revenue_category.update_traces(texttemplate='$%{text:,.2s}', textposition='outside')
st.plotly_chart(fig_revenue_category, use_container_width=True)

# Bar Chart - Revenue by Time of Day
st.write("---")
filtered_df['hour'] = filtered_df['order_datetime'].dt.hour
revenue_by_hour = filtered_df.groupby('hour')['item_total_price'].sum().reset_index()
fig_revenue_hour = px.bar(
    revenue_by_hour,
    x='hour',
    y='item_total_price',
    title='הכנסות לפי שעת היום',
    labels={'hour': 'שעת היום (0-23)', 'item_total_price': 'הכנסה ($)'},
    template="plotly_white",
    text='item_total_price'
)
fig_revenue_hour.update_layout(title_x
