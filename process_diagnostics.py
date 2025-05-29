
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def process_revenue_data(filepath):
    df = pd.read_excel(filepath)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Fill blank with 0s
    df = df.fillna(0)

    # Add calculated column
    df['pct_visits_with_labs'] = df['Visits With Lab Count'] / df['Visit Count'].replace(0, np.nan)

    # Weekly Aggregates
    weekly_agg = df.groupby(['Year', 'Week']).agg({
        'Total Payments': 'sum'
    }).reset_index()

    # Add average weekly features for regression
    weekly_features = df.groupby(['Year', 'Week']).agg({
        '% of Total Payments': 'mean',
        'Avg. Payment Per Visit': 'mean',
        'Avg. Chart E/M Weight': 'mean',
        'Charge Amount': 'mean',
        'Collection %': 'mean',
        'Visit Count': 'sum',
        'pct_visits_with_labs': 'mean'
    }).reset_index()

    full_weekly = pd.merge(weekly_agg, weekly_features, on=['Year', 'Week'])

    # Regression modeling
    X = full_weekly.drop(columns=['Total Payments', 'Year', 'Week'])
    y = full_weekly['Total Payments']

    model = LinearRegression()
    model.fit(X, y)
    full_weekly['Predicted Payments'] = model.predict(X)

    # Classification
    full_weekly['% Error'] = (full_weekly['Total Payments'] - full_weekly['Predicted Payments']) / full_weekly['Predicted Payments']
    full_weekly['Absolute Error'] = full_weekly['% Error'].abs()
    full_weekly['Performance Diagnostic'] = full_weekly['% Error'].apply(
        lambda x: 'Over Performed' if x > 0.025 else ('Under Performed' if x < -0.025 else 'Average Performance')
    )

    # Add average values per payer + E/M
    group_means = df.groupby(['Payer', 'Group E/M codes']).agg({
        'Avg. Payment Per Visit': 'mean',
        'Charge Amount': 'mean',
        'Collection %': 'mean',
        'Visit Count': 'mean'
    }).reset_index()

    group_means.columns = ['Payer', 'Group E/M codes',
                           'Avg Pay Per Visit Avg',
                           'Charge Amount Avg',
                           'Collection % Avg',
                           'Visit Count Avg']

    df = df.merge(group_means, on=['Payer', 'Group E/M codes'], how='left')

    for metric, avg_col in [
        ('Avg. Payment Per Visit', 'Avg Pay Per Visit Avg'),
        ('Charge Amount', 'Charge Amount Avg'),
        ('Collection %', 'Collection % Avg'),
        ('Visit Count', 'Visit Count Avg')
    ]:
        df[f'{metric} Diff'] = df[metric] - df[avg_col]

    # Generate diagnostics
    diagnostic_results = []

    for (year, week), week_data in df.groupby(['Year', 'Week']):
        diagnostics = {
            'Year': year,
            'Week': week,
            'What Went Well': 'null',
            'What Can Be Improved': 'null',
            'Aetna Analysis': 'null',
            'BCBS Analysis': 'null'
        }

        def extract(group, condition):
            reasons = []
            for metric, avg_col in [
                ('Avg. Payment Per Visit', 'Avg Pay Per Visit Avg'),
                ('Charge Amount', 'Charge Amount Avg'),
                ('Collection %', 'Collection % Avg'),
                ('Visit Count', 'Visit Count Avg')
            ]:
                diff_col = f'{metric} Diff'
                if diff_col in group:
                    g = group[condition(group[diff_col])].copy()
                    g['abs_diff'] = g[diff_col].abs()
                    top = g.nlargest(2, 'abs_diff')
                    for _, row in top.iterrows():
                        base = f"{row['Payer']} - {row['Group E/M codes']} {metric}"
                        val = row[metric]
                        comp = row[avg_col]
                        reasons.append(f"{base} is {val:.2f}, while its overall average is {comp:.2f}.")
            return reasons

        diagnostics['What Went Well'] = ' | '.join(extract(week_data, lambda x: x > 0)) or 'null'
        diagnostics['What Can Be Improved'] = ' | '.join(extract(week_data, lambda x: x < 0)) or 'null'
        diagnostics['Aetna Analysis'] = ' | '.join(extract(week_data[week_data['Payer'].str.contains("AETNA", na=False)], lambda x: x != 0)) or 'null'
        diagnostics['BCBS Analysis'] = ' | '.join(extract(week_data[week_data['Payer'].str.contains("BCBS", na=False)], lambda x: x != 0)) or 'null'

        diagnostic_results.append(diagnostics)

    diag_df = pd.DataFrame(diagnostic_results)
    result = pd.merge(full_weekly, diag_df, on=['Year', 'Week'], how='left')

    return result

if __name__ == '__main__':
    input_file = 'Weekly Performance Analsysis Export '24 & '24 W019.xlsx'
    df_result = process_revenue_data(input_file)
    df_result.to_excel('Weekly_Model_Full_Diagnostics.xlsx', index=False)
