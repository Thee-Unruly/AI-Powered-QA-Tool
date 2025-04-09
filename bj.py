import pandas as pd

# Define data
project_name = ['Milestone 1 Budgeted Revenue', 'Milestone 2 Budgeted Value']
client_name = ['Client A', 'Client B']
hourly_wages_or_costs = ['$100/hr', '$75/hr']
overhead_costs = ['500.00', '200.00']
gross_revenue_earned = ['', '']
net_profit_margin_percentage = ['', '']
milestone_no = [1, 2]
software_license_fees = ['1000.00', '500.00']
infrastructure_support_costs = ['2000.00', '800.00']

# Create DataFrame
data = pd.DataFrame({
    "Project Name": project_name,
    "Client Name": client_name,
    "Hourly Wage/Rate ($)": hourly_wages_or_costs,
    "Overhead Costs ($)": overhead_costs,
    "Gross Revenue Earned ($)": gross_revenue_earned,
    "Net Profit (Profit Margin%)": net_profit_margin_percentage,
    "Milestone No.": milestone_no,
    "Software License Fees ($)": software_license_fees,
    "Infrastructure Support Costs ($)": infrastructure_support_costs,
})

# Save the DataFrame to a CSV file
data.to_csv('output.csv', index=False)

print("Data has been saved to 'output.csv'")
