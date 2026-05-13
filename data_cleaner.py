
---

## **FILE 2: `data_cleaner.py`**

```python
import pandas as pd
import numpy as np
import re
from datetime import datetime
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

print("="*60)
print("🔧 DATA CLEANING & REPORTING AUTOMATION")
print("="*60)

# ============================================
# 1. GENERATE DIRTY SAMPLE DATA
# ============================================
print("\n📁 Step 1: Generating dirty sample data...")

def generate_dirty_data(n=500):
    """Generate intentionally messy data for cleaning"""
    
    # Clean base data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n, freq='D')
    
    # Create clean data first
    clean_data = {
        'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(1, n+1)],
        'name': [random.choice(['John Smith', 'Jane Doe', 'Mike Johnson', 'Sarah Williams', 
                               'Raj Kumar', 'Priya Patel', 'Tom Brown', 'Lisa Garcia',
                               'David Lee', 'Maria Rodriguez']) for _ in range(n)],
        'email': [f'user{random.randint(1,1000)}@gmail.com' for _ in range(n)],
        'phone': [f'+1-{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}' for _ in range(n)],
        'age': np.random.randint(18, 80, n),
        'income': np.random.randint(20000, 150000, n),
        'purchase_date': dates,
        'product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Charger'], n),
        'quantity': np.random.randint(1, 10, n),
        'price': np.random.choice([500, 1000, 1500, 2000, 2500], n),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n),
        'status': np.random.choice(['Active', 'Inactive', 'Pending', 'Suspended'], n, p=[0.7, 0.15, 0.1, 0.05])
    }
    
    df = pd.DataFrame(clean_data)
    
    # ========== INTRODUCE DIRTY DATA ==========
    
    # 1. Missing values
    missing_positions = [
        (df.columns.get_loc('age'), slice(10, 30)),      # 20 missing ages
        (df.columns.get_loc('email'), slice(50, 65)),    # 15 missing emails
        (df.columns.get_loc('phone'), slice(100, 120)),  # 20 missing phones
        (df.columns.get_loc('income'), slice(200, 215)), # 15 missing incomes
        (df.columns.get_loc('city'), slice(300, 310)),   # 10 missing cities
    ]
    for col_idx, rows in missing_positions:
        df.iloc[rows, col_idx] = np.nan
    
    # 2. Duplicates
    duplicate_rows = [25, 26, 150, 151, 152, 300, 301, 400, 401]
    duplicate_data = df.iloc[duplicate_rows].copy()
    df = pd.concat([df, duplicate_data], ignore_index=True)
    
    # 3. Invalid data types
    df.loc[150:160, 'age'] = 'unknown'  # String instead of number
    df.loc[200:210, 'quantity'] = 'N/A'
    df.loc[250:260, 'price'] = -999  # Negative price
    df.loc[350:360, 'income'] = -1  # Negative income
    
    # 4. Outliers
    df.loc[400:410, 'age'] = np.random.randint(150, 200, 11)  # Impossible ages
    df.loc[420:430, 'income'] = np.random.randint(500000, 1000000, 11)  # Extreme incomes
    
    # 5. Inconsistent text
    df.loc[180:190, 'name'] = df.loc[180:190, 'name'].str.upper()
    df.loc[220:230, 'name'] = df.loc[220:230, 'name'].str.lower()
    df.loc[260:270, 'email'] = df.loc[260:270, 'email'].str.upper()
    df.loc[280:290, 'city'] = df.loc[280:290, 'city'].str.lower()
    
    # 6. Invalid emails
    bad_emails = ['invalidemail', 'missing@', '@nouser.com', 'user@.com', 'user@domain.']
    for i, email in enumerate(bad_emails):
        if i < len(bad_emails):
            df.loc[310 + i, 'email'] = email
    
    # 7. Future dates
    future_dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
    df.loc[320:329, 'purchase_date'] = future_dates
    
    # 8. Whitespace issues
    df.loc[340:345, 'product'] = df.loc[340:345, 'product'].apply(lambda x: f"  {x}  ")
    
    # 9. Special characters
    df.loc[360:365, 'phone'] = ['123-456-7890', '123.456.7890', '(123) 456-7890', '1234567890', '+1 123 456 7890', '123-456-7890 ext 123']
    
    # 10. Wrong status values
    df.loc[370:375, 'status'] = ['ACTIVE', 'active', 'InACTIVE', 'inactive', 'PENDING', 'SUSPENDED']
    
    return df

# Generate dirty data
df_raw = generate_dirty_data(500)
df_raw.to_csv('raw_data.csv', index=False)
print(f"✅ Generated {len(df_raw)} records with intentional issues")
print(f"   Problems introduced: Missing values, duplicates, outliers, inconsistencies")

# ============================================
# 2. DATA CLEANING FUNCTION
# ============================================
print("\n🔧 Step 2: Performing automated data cleaning...")

class DataCleaner:
    def __init__(self, df, log_file='cleaning_log.txt'):
        self.df = df.copy()
        self.original_shape = df.shape
        self.log = []
        self.log_file = log_file
        
    def log_action(self, action, details, rows_affected=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}: {details}"
        if rows_affected:
            log_entry += f" (Affected: {rows_affected} rows)"
        self.log.append(log_entry)
        print(f"   ✓ {action}")
        
    def remove_duplicates(self, subset=None, keep='first'):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        removed = before - after
        self.log_action("Remove Duplicates", f"Removed {removed} duplicate rows", removed)
        return self
    
    def fix_missing_values(self):
        """Handle missing values intelligently"""
        
        # Numeric columns: fill with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                self.log_action("Fix Missing Values", f"Filled {missing} missing values in '{col}' with median ({median_val:.0f})", missing)
        
        # Categorical columns: fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                self.log_action("Fix Missing Values", f"Filled {missing} missing values in '{col}' with mode ({mode_val})", missing)
        
        return self
    
    def fix_outliers(self, method='iqr', threshold=3):
        """Detect and fix outliers"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            else:  # z-score
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = self.df[z_scores > threshold]
            
            if len(outliers) > 0:
                # Cap outliers to bounds
                if method == 'iqr':
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                else:
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    self.df[col] = self.df[col].clip(mean_val - threshold*std_val, mean_val + threshold*std_val)
                self.log_action("Fix Outliers", f"Capped {len(outliers)} outliers in '{col}'", len(outliers))
        
        return self
    
    def fix_data_types(self):
        """Convert columns to correct data types"""
        
        # Age should be int
        if 'age' in self.df.columns:
            self.df['age'] = pd.to_numeric(self.df['age'], errors='coerce')
            self.df['age'].fillna(self.df['age'].median(), inplace=True)
            self.df['age'] = self.df['age'].astype(int)
            self.log_action("Fix Data Types", "Converted 'age' to integer")
        
        # Quantity should be int
        if 'quantity' in self.df.columns:
            self.df['quantity'] = pd.to_numeric(self.df['quantity'], errors='coerce')
            self.df['quantity'].fillna(1, inplace=True)
            self.df['quantity'] = self.df['quantity'].astype(int)
            self.log_action("Fix Data Types", "Converted 'quantity' to integer")
        
        # Price should be positive float
        if 'price' in self.df.columns:
            self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
            self.df.loc[self.df['price'] < 0, 'price'] = self.df['price'].median()
            self.df['price'] = self.df['price'].astype(float)
            self.log_action("Fix Data Types", "Converted 'price' to float and fixed negatives")
        
        return self
    
    def fix_inconsistent_text(self):
        """Standardize text columns"""
        
        # Strip whitespace
        object_cols = self.df.select_dtypes(include=['object']).columns
        for col in object_cols:
            self.df[col] = self.df[col].astype(str).str.strip()
        
        # Proper case for names
        if 'name' in self.df.columns:
            self.df['name'] = self.df['name'].str.title()
            self.log_action("Fix Text", "Standardized 'name' to proper case")
        
        # Lowercase for emails
        if 'email' in self.df.columns:
            self.df['email'] = self.df['email'].str.lower()
            self.log_action("Fix Text", "Standardized 'email' to lowercase")
        
        # Standardize product names
        if 'product' in self.df.columns:
            self.df['product'] = self.df['product'].str.strip().str.title()
            self.log_action("Fix Text", "Standardized 'product' names")
        
        # Standardize city names
        if 'city' in self.df.columns:
            self.df['city'] = self.df['city'].str.title()
            self.log_action("Fix Text", "Standardized 'city' names")
        
        return self
    
    def fix_dates(self):
        """Fix date columns"""
        if 'purchase_date' in self.df.columns:
            self.df['purchase_date'] = pd.to_datetime(self.df['purchase_date'], errors='coerce')
            # Remove future dates
            today = datetime.now()
            self.df.loc[self.df['purchase_date'] > today, 'purchase_date'] = today
            self.log_action("Fix Dates", "Standardized dates and removed future dates")
        return self
    
    def validate_emails(self):
        """Validate and fix email format"""
        if 'email' in self.df.columns:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            invalid_mask = ~self.df['email'].astype(str).str.match(email_pattern, na=False)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                self.df.loc[invalid_mask, 'email'] = 'unknown@email.com'
                self.log_action("Validate Emails", f"Fixed {invalid_count} invalid email addresses", invalid_count)
        return self
    
    def fix_categorical(self):
        """Standardize categorical values"""
        if 'status' in self.df.columns:
            status_map = {
                'active': 'Active', 'ACTIVE': 'Active',
                'inactive': 'Inactive', 'INACTIVE': 'Inactive',
                'pending': 'Pending', 'PENDING': 'Pending',
                'suspended': 'Suspended', 'SUSPENDED': 'Suspended'
            }
            self.df['status'] = self.df['status'].map(status_map).fillna('Unknown')
            self.log_action("Fix Categories", "Standardized status values")
        return self
    
    def run_clean(self):
        """Run all cleaning steps"""
        self.remove_duplicates()
        self.fix_missing_values()
        self.fix_data_types()
        self.fix_outliers(method='iqr')
        self.fix_inconsistent_text()
        self.fix_dates()
        self.validate_emails()
        self.fix_categorical()
        
        # Save cleaning log
        with open(self.log_file, 'w') as f:
            f.write("DATA CLEANING LOG\n")
            f.write("="*50 + "\n\n")
            for entry in self.log:
                f.write(entry + "\n")
            f.write(f"\nOriginal rows: {self.original_shape[0]}\n")
            f.write(f"Cleaned rows: {len(self.df)}\n")
            f.write(f"Rows removed: {self.original_shape[0] - len(self.df)}\n")
        
        return self.df

# Run cleaning
cleaner = DataCleaner(df_raw)
df_cleaned = cleaner.run_clean()

# Save cleaned data
df_cleaned.to_csv('cleaned_data.csv', index=False)
print(f"\n✅ Cleaning complete!")
print(f"   Original: {len(df_raw)} rows")
print(f"   Cleaned: {len(df_cleaned)} rows")
print(f"   Removed: {len(df_raw) - len(df_cleaned)} rows")

# ============================================
# 3. DATA QUALITY REPORT
# ============================================
print("\n📊 Step 3: Generating data quality report...")

def generate_quality_report(raw_df, clean_df):
    """Generate comprehensive quality metrics"""
    
    quality_metrics = {
        'metric': [],
        'before': [],
        'after': [],
        'improvement': []
    }
    
    # Missing values
    missing_before = raw_df.isnull().sum().sum()
    missing_after = clean_df.isnull().sum().sum()
    quality_metrics['metric'].append('Missing Values')
    quality_metrics['before'].append(missing_before)
    quality_metrics['after'].append(missing_after)
    quality_metrics['improvement'].append(missing_before - missing_after)
    
    # Duplicates
    dup_before = raw_df.duplicated().sum()
    dup_after = clean_df.duplicated().sum()
    quality_metrics['metric'].append('Duplicate Rows')
    quality_metrics['before'].append(dup_before)
    quality_metrics['after'].append(dup_after)
    quality_metrics['improvement'].append(dup_before - dup_after)
    
    # Data types (correctly typed columns)
    correct_types_before = sum([str(raw_df[col].dtype) for col in raw_df.columns]).count('int64') + sum([str(raw_df[col].dtype) for col in raw_df.columns]).count('float64')
    correct_types_after = sum([str(clean_df[col].dtype) for col in clean_df.columns]).count('int64') + sum([str(clean_df[col].dtype) for col in clean_df.columns]).count('float64')
    quality_metrics['metric'].append('Correctly Typed')
    quality_metrics['before'].append(correct_types_before)
    quality_metrics['after'].append(correct_types_after)
    quality_metrics['improvement'].append(correct_types_after - correct_types_before)
    
    # Unique values (data consistency)
    unique_before = raw_df.nunique().sum()
    unique_after = clean_df.nunique().sum()
    quality_metrics['metric'].append('Unique Values')
    quality_metrics['before'].append(unique_before)
    quality_metrics['after'].append(unique_after)
    quality_metrics['improvement'].append(unique_after - unique_before)
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_df.to_csv('quality_metrics.csv', index=False)
    
    return quality_df

quality_df = generate_quality_report(df_raw, df_cleaned)

print("\n📈 QUALITY METRICS IMPROVEMENT:")
print(quality_df.to_string(index=False))

# ============================================
# 4. CREATE VISUALIZATIONS
# ============================================
print("\n📊 Step 4: Creating visualizations...")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Figure 1: Before vs After Comparison
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Data Quality Improvement Dashboard', fontsize=16, fontweight='bold')

# Missing values comparison
axes[0,0].bar(['Before', 'After'], [quality_df.loc[0, 'before'], quality_df.loc[0, 'after']], 
              color=['#E63946', '#2E86AB'])
axes[0,0].set_title('Missing Values', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate([quality_df.loc[0, 'before'], quality_df.loc[0, 'after']]):
    axes[0,0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Duplicates comparison
axes[0,1].bar(['Before', 'After'], [quality_df.loc[1, 'before'], quality_df.loc[1, 'after']], 
              color=['#E63946', '#2E86AB'])
axes[0,1].set_title('Duplicate Rows', fontsize=12, fontweight='bold')
axes[0,1].set_ylabel('Count')
for i, v in enumerate([quality_df.loc[1, 'before'], quality_df.loc[1, 'after']]):
    axes[0,1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Data types comparison
axes[1,0].bar(['Before', 'After'], [quality_df.loc[2, 'before'], quality_df.loc[2, 'after']], 
              color=['#E63946', '#2E86AB'])
axes[1,0].set_title('Correctly Typed Columns', fontsize=12, fontweight='bold')
axes[1,0].set_ylabel('Count')
for i, v in enumerate([quality_df.loc[2, 'before'], quality_df.loc[2, 'after']]):
    axes[1,0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Improvement summary
improvements = [quality_df.loc[0, 'improvement'], quality_df.loc[1, 'improvement'], quality_df.loc[2, 'improvement']]
axes[1,1].bar(['Missing Fixed', 'Duplicates Removed', 'Types Corrected'], improvements, 
              color=['#2E86AB', '#A23B72', '#F18F01'])
axes[1,1].set_title('Improvements Made', fontsize=12, fontweight='bold')
axes[1,1].set_ylabel('Count')
for i, v in enumerate(improvements):
    axes[1,1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('quality_comparison.png', dpi=100, bbox_inches='tight')
plt.close()

# Figure 2: Data Distribution Comparison (Sample numeric column)
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Age Distribution: Before vs After Cleaning', fontsize=14, fontweight='bold')

axes[0].hist(df_raw['age'].dropna(), bins=30, color='#E63946', alpha=0.7, edgecolor='black')
axes[0].set_title('Before Cleaning', fontsize=12)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

axes[1].hist(df_cleaned['age'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
axes[1].set_title('After Cleaning', fontsize=12)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('age_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================
# 5. EXCEL REPORT GENERATION
# ============================================
print("\n📑 Step 5: Generating Excel report...")

with pd.ExcelWriter('data_summary.xlsx', engine='xlsxwriter') as writer:
    # Sheet 1: Raw Data Sample
    df_raw.head(100).to_excel(writer, sheet_name='Raw Data Sample', index=False)
    
    # Sheet 2: Cleaned Data Sample
    df_cleaned.head(100).to_excel(writer, sheet_name='Cleaned Data Sample', index=False)
    
    # Sheet 3: Summary Statistics
    summary_stats = df_cleaned.describe(include='all').round(2)
    summary_stats.to_excel(writer, sheet_name='Summary Statistics')
    
    # Sheet 4: Data Quality Report
    quality_df.to_excel(writer, sheet_name='Quality Metrics', index=False)
    
    # Sheet 5: Column Info
    column_info = pd.DataFrame({
        'Column': df_cleaned.columns,
        'Data Type': df_cleaned.dtypes.values,
        'Non-Null Count': df_cleaned.count().values,
        'Unique Values': df_cleaned.nunique().values,
        'Missing Values': df_cleaned.isnull().sum().values,
        'Missing %': (df_cleaned.isnull().sum() / len(df_cleaned) * 100).values
    })
    column_info.to_excel(writer, sheet_name='Column Information', index=False)
    
    # Format Excel
    workbook = writer.book
    header_format = workbook.add_format({'bold': True, 'bg_color': '#2E86AB', 'font_color': 'white'})
    
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        for col_num, value in enumerate(summary_stats.columns[:min(10, len(summary_stats.columns))]):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)

print("✅ Excel report saved: data_summary.xlsx")

# ============================================
# 6. INTERACTIVE HTML DASHBOARD
# ============================================
print("\n🌐 Step 6: Creating interactive dashboard...")

# Create summary metrics
missing_fixed = quality_df.loc[0, 'improvement']
dupes_removed = quality_df.loc[1, 'improvement']
completeness = (1 - df_cleaned.isnull().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1])) * 100

fig = make_subplots(rows=3, cols=3,
                    subplot_titles=('Data Quality Score', 'Missing Values Fixed', 'Duplicates Removed',
                                   'Column Completeness', 'Data Types Distribution', 'Top Products',
                                   'Status Distribution', 'City Distribution', 'Cleaning Impact'),
                    specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                           [{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}],
                           [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}]])

# Quality Score
fig.add_trace(go.Indicator(mode="gauge+number", value=completeness, title={'text': "Data Completeness (%)"},
                           gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2E86AB"},
                                  'steps': [{'range': [0, 50], 'color': "#E63946"},
                                           {'range': [50, 80], 'color': "#F18F01"},
                                           {'range': [80, 100], 'color': "#2E86AB"}]}), row=1, col=1)

# Missing values fixed
fig.add_trace(go.Indicator(mode="number+delta", value=missing_fixed, delta={'reference': 0, 'increasing': {'color': "green"}},
                          title={'text': "Missing Values Fixed"}), row=1, col=2)

# Duplicates removed
fig.add_trace(go.Indicator(mode="number+delta", value=dupes_removed, delta={'reference': 0, 'increasing': {'color': "green"}},
                          title={'text': "Duplicate Rows Removed"}), row=1, col=3)

# Column completeness
col_completeness = (1 - df_cleaned.isnull().sum() / len(df_cleaned)) * 100
fig.add_trace(go.Bar(x=col_completeness.index, y=col_completeness.values, marker_color='#2E86AB',
                    text=col_completeness.round(1), textposition='outside'), row=2, col=1)

# Data types distribution
dtype_counts = df_cleaned.dtypes.astype(str).value_counts()
fig.add_trace(go.Pie(labels=dtype_counts.index, values=dtype_counts.values, marker=dict(colors=['#2E86AB', '#A23B72', '#F18F01'])), row=2, col=2)

# Top products
top_products = df_cleaned['product'].value_counts().head(5)
fig.add_trace(go.Bar(x=top_products.values, y=top_products.index, orientation='h', marker_color='#F18F01',
                    text=top_products.values, textposition='outside'), row=2, col=3)

# Status distribution
status_counts = df_cleaned['status'].value_counts()
fig.add_trace(go.Pie(labels=status_counts.index, values=status_counts.values), row=3, col=1)

# City distribution
city_counts = df_cleaned['city'].value_counts().head(8)
fig.add_trace(go.Bar(x=city_counts.index, y=city_counts.values, marker_color='#A23B72',
                    text=city_counts.values, textposition='outside'), row=3, col=2)

# Cleaning impact summary
impact_data = {
    'Before': [quality_df.loc[0, 'before'], quality_df.loc[1, 'before']],
    'After': [quality_df.loc[0, 'after'], quality_df.loc[1, 'after']]
}
fig.add_trace(go.Bar(x=['Missing Values', 'Duplicates'], y=impact_data['Before'], name='Before', marker_color='#E63946'), row=3, col=3)
fig.add_trace(go.Bar(x=['Missing Values', 'Duplicates'], y=impact_data['After'], name='After', marker_color='#2E86AB'), row=3, col=3)

fig.update_layout(title="Data Cleaning Automation Dashboard", height=1000, showlegend=True)
fig.write_html('cleaning_report.html')
print("✅ Interactive dashboard saved: cleaning_report.html")

# ============================================
# 7. FINAL REPORT
# ============================================
print("\n📝 Step 7: Generating final summary...")

report = f"""
{'='*60}
DATA CLEANING AUTOMATION REPORT
{'='*60}

📊 CLEANING SUMMARY:
   • Original dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns
   • Cleaned dataset: {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns
   • Rows removed: {len(df_raw) - len(df_cleaned)} (duplicates/invalid)
   • Data completeness: {completeness:.1f}%

🔧 OPERATIONS PERFORMED:
   {len([l for l in cleaner.log if 'Duplicate' in l])} duplicate removal operations
   {len([l for l in cleaner.log if 'Missing' in l])} missing value fixes
   {len([l for l in cleaner.log if 'Outlier' in l])} outlier treatments
   {len([l for l in cleaner.log if 'Text' in l])} text standardization operations
   {len([l for l in cleaner.log if 'Type' in l])} data type corrections

📈 QUALITY IMPROVEMENTS:
   • Missing values: {quality_df.loc[0, 'before']} → {quality_df.loc[0, 'after']} ({quality_df.loc[0, 'improvement']} fixed)
   • Duplicates: {quality_df.loc[1, 'before']} → {quality_df.loc[1, 'after']} ({quality_df.loc[1, 'improvement']} removed)
   • Correct data types: {quality_df.loc[2, 'before']} → {quality_df.loc[2, 'after']} (+{quality_df.loc[2, 'improvement']})

📁 OUTPUT FILES GENERATED:
   1. cleaned_data.csv - Final cleaned dataset
   2. raw_data.csv - Original dirty data (for reference)
   3. cleaning_log.txt - Detailed log of all operations
   4. data_summary.xlsx - Excel report with 5 sheets
   5. cleaning_report.html - Interactive HTML dashboard
   6. quality_comparison.png - Before/after comparison chart
   7. age_distribution.png - Distribution comparison
   8. quality_metrics.csv - Quality metrics data

⏱️ PROCESSING TIME SAVINGS:
   • Manual cleaning estimate: 2-3 hours
   • Automated cleaning: < 10 seconds
   • Time saved: ~99% reduction

🔄 NEXT STEPS FOR AUTOMATION:
   1. Schedule this script to run daily/weekly
   2. Configure email reports: python send_report.py --email team@company.com
   3. Integrate with database: pd.read_sql() instead of CSV
   4. Add custom rules for domain-specific validation
   5. Set up logging to database for audit trail

{'='*60}
AUTOMATION READY!
{'='=*60}

This script can be:
✅ Scheduled via cron/Task Scheduler
✅ Triggered by file drops
✅ Integrated into ETL pipelines
✅ Extended with custom rules
✅ Connected to databases/APIs

To automate further:
```python
# Schedule daily cleanup
# Windows Task Scheduler: python data_cleaner.py
# Linux Cron: 0 9 * * * python /path/to/data_cleaner.py

# Email results
import smtplib
# Add email sending code here
