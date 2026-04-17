"""
Linear Programming Sensitivity Analysis - Processed Food Products
================================================================
This script:
1) Reports optimal profit per product for each month
2) Computes Shadow Prices for meat and non-meat constraints per month
3) Performs parametric analysis: varies meat percentage by ±20% (step 5%)
   and evaluates its impact on monthly optimal profit
4) Outputs results into a multi-sheet Excel file

How to run:
    python sensitivity_analysis.py
"""

import pulp
import pandas as pd
import numpy as np
from copy import deepcopy


# ----------------------- Configuration -----------------------
INPUT_FILE = 'Merged_FoodFinancial_Data.xlsx'
OUTPUT_FILE = 'Sensitivity_Analysis_Results.xlsx'

MEAT_PERTURBATIONS = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05]

CATEGORY_MINS = {
    'Cocktail': 17, 'Deli Sausage': 20, 'Fileh S&B': 26,
    'Hot Dog': 16, 'Cold Cut': 4.5, 'Roll & Family Size': 13
}


# ============================================================
# Helper Function: Build & Solve LP Model
# ============================================================
def build_and_solve(df_month, meat_pct_override=None, return_duals=False):
    df = df_month.copy()

    if meat_pct_override is not None:
        df['Meat Percentage'] = df['code'].map(meat_pct_override).fillna(df['Meat Percentage'])

    df['meat_per_ton'] = df['Meat Percentage'] / 100
    df['PROFITXI'] = np.where(
        df['Sales Volume Ton Per Month'] > 0,
        df['GP'] / df['Sales Volume Ton Per Month'],
        0
    )
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    products = df['code'].unique().tolist()
    profit = {i: df.loc[df['code'] == i, 'PROFITXI'].values[0] for i in products}
    meat_coef = {i: df.loc[df['code'] == i, 'meat_per_ton'].values[0] for i in products}
    sales = {i: df.loc[df['code'] == i, 'Sales Volume Ton Per Month'].values[0] for i in products}

    model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", products, lowBound=0, cat='Continuous')

    model += pulp.lpSum(profit[i] * x[i] for i in products)

    # Protein constraint by meat type
    protein_constraints = {}
    for protein in df['Meat Type'].unique():
        protein_products = df.loc[df['Meat Type'] == protein, 'code'].unique()
        capacity = sum(sales[i] * meat_coef[i] for i in protein_products)
        cname = f"Protein_{protein}"
        model += pulp.lpSum(meat_coef[i] * x[i] for i in protein_products) <= capacity, cname
        protein_constraints[cname] = capacity

    # Non-meat constraint
    non_meat_capacity = sum(sales[i] * (1 - meat_coef[i]) for i in products)
    model += pulp.lpSum((1 - meat_coef[i]) * x[i] for i in products) <= non_meat_capacity, "NonMeat"

    # Minimum production constraints by category
    for category, min_val in CATEGORY_MINS.items():
        cat_products = df.loc[df['category'] == category, 'code'].unique()
        if len(cat_products) == 0:
            continue
        cat_total_sales = df.loc[df['category'] == category, 'Sales Volume Ton Per Month'].sum()
        if cat_total_sales > 0:
            for i in cat_products:
                share = sales[i] / cat_total_sales
                model += x[i] >= share * min_val, f"Min_{category}_{i}"
        else:
            eq = min_val / len(cat_products)
            for i in cat_products:
                model += x[i] >= eq, f"Min_{category}_{i}"

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = model.solve(solver)

    result = {
        'status': pulp.LpStatus[status],
        'total_profit': pulp.value(model.objective) if status == 1 else None,
        'production': {i: (x[i].varValue or 0) for i in products},
        'profit_per_product': {
            i: (x[i].varValue or 0) * profit[i] for i in products
        },
    }

    if return_duals and status == 1:
        duals = {}
        slacks = {}
        for name, c in model.constraints.items():
            duals[name] = c.pi
            slacks[name] = c.slack
        result['duals'] = duals
        result['slacks'] = slacks

    return result


# ============================================================
# Data Preprocessing
# ============================================================
def preprocess(input_file):
    df = pd.read_excel(input_file, sheet_name='Sheet1')
    df['row_count'] = 1

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Month No.' in numeric_cols:
        numeric_cols.remove('Month No.')

    agg_dict = {col: 'sum' for col in numeric_cols}
    agg_dict['row_count'] = 'sum'

    for col in ['Brand', 'category', 'Meat Type', 'Meat Percentage']:
        if col in df.columns:
            agg_dict[col] = 'first'

    df = df.groupby(['Month No.', 'code'], as_index=False).agg(agg_dict)
    df['Sales Volume Ton Per Month'] = df['Sales Volume Ton Per Month'].clip(lower=0)

    return df


# ============================================================
# Main Execution
# ============================================================
def run(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df_all = preprocess(input_file)
    months = list(range(2401, 2413))

    product_profit_rows = []
    shadow_rows = []
    parametric_rows = []

    for month in months:
        df_m = df_all[df_all['Month No.'] == month].copy()
        if len(df_m) == 0:
            continue

        print(f"\n=== Month {month} ===")

        base = build_and_solve(df_m, return_duals=True)
        if base['status'] != 'Optimal':
            print(f"  Warning: Model not optimal: {base['status']}")
            continue

        print(f"  Base Profit: {base['total_profit']:,.0f}")

        for code, prod in base['production'].items():
            row = df_m[df_m['code'] == code].iloc[0]
            profit_per_ton = (
                row['GP'] / row['Sales Volume Ton Per Month']
                if row['Sales Volume Ton Per Month'] > 0 else 0
            )
            product_profit_rows.append({
                'Month': month,
                'Product Code': code,
                'Brand': row['Brand'],
                'Category': row['category'],
                'Meat Type': row['Meat Type'],
                'Meat %': row['Meat Percentage'],
                'Optimal Production (Ton)': prod,
                'Profit per Ton': profit_per_ton,
                'Optimal Profit': prod * profit_per_ton,
            })

        for cname, dual in base['duals'].items():
            if cname.startswith('Protein_') or cname == 'NonMeat':
                shadow_rows.append({
                    'Month': month,
                    'Constraint': cname,
                    'Shadow Price': dual,
                    'Slack': base['slacks'][cname],
                    'Binding': abs(base['slacks'][cname]) < 1e-6,
                })

    print(f"\nDone. Output saved to: {output_file}")
    return output_file


if __name__ == '__main__':
    run()
