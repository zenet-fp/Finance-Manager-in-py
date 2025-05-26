import json
import os
import tkinter
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dateutil.relativedelta import relativedelta
import requests
import numpy as np

# API for currency conversion (free tier)
CURRENCY_API_URL = "https://api.exchangerate-api.com/v4/latest/GBP"  # Base currency is GBP


# ----------------- Forecasting Algorithms -----------------
class HoltWinters:
    """Holt-Winters Exponential Smoothing with trend and seasonality."""

    def __init__(self, season_period=12):
        self.alpha = 0.2  # Level smoothing
        self.beta = 0.1  # Trend smoothing
        self.gamma = 0.1  # Seasonality smoothing
        self.season_period = season_period

    def fit(self, data):
        n = len(data)
        self.level = np.zeros(n)
        self.trend = np.zeros(n)
        self.season = np.ones(n) * np.nan
        self.fitted = np.zeros(n)

        # Initialize seasonal indices
        self.season[:self.season_period] = data[:self.season_period] / np.mean(data[:self.season_period])
        self.level[0] = data[0]
        self.trend[0] = data[1] - data[0]

        for i in range(1, n):
            if i >= self.season_period:
                self.season[i] = self.gamma * (data[i] / self.level[i - self.season_period]) + (1 - self.gamma) * \
                                 self.season[i - self.season_period]
            self.level[i] = self.alpha * (data[i] / self.season[i]) + (1 - self.alpha) * (
                        self.level[i - 1] + self.trend[i - 1])
            self.trend[i] = self.beta * (self.level[i] - self.level[i - 1]) + (1 - self.beta) * self.trend[i - 1]
            self.fitted[i] = (self.level[i - 1] + self.trend[i - 1]) * self.season[i]

    def forecast(self, steps):
        forecast = []
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        last_season = self.season[-self.season_period:]
        for i in range(steps):
            forecast_value = (last_level + (i + 1) * last_trend) * last_season[i % self.season_period]
            forecast.append(forecast_value)
        return forecast


def linear_regression_forecast(data, steps):
    """Linear regression model for trend-based forecasting."""
    n = len(data)
    x = np.arange(n)
    y = np.array(data)
    x_mean, y_mean = np.mean(x), np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return [y_mean] * steps  # Fallback for constant data
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return [slope * (n + i) + intercept for i in range(steps)]


def moving_average_forecast(data, steps, window=12):
    """Moving average model for stable predictions."""
    if len(data) < window:
        return [np.mean(data)] * steps if len(data) > 0 else [0] * steps
    return [np.mean(data[-window:])] * steps


# Data Management
class UserData:
    def __init__(self, username):
        self.username = username
        self.data_file = f"{username}.json"
        self.default_data = {
            'password': '',
            'transactions': [],
            'settings': {
                'currency': 'GBP',
                'theme': '#f0f0f0',
                'graph_type': 'bar'
            }
        }
        self.data = self.default_data.copy()
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data.update(json.load(f))

    def save_data(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f)

    def add_transaction(self, transaction):
        self.data['transactions'].append(transaction)
        self.save_data()

    def get_transactions(self):
        return self.data['transactions']

    def update_settings(self, settings):
        self.data['settings'].update(settings)
        self.save_data()


# Currency Converter
class CurrencyConverter:
    @staticmethod
    def get_exchange_rates():
        try:
            response = requests.get(CURRENCY_API_URL)
            response.raise_for_status()
            return response.json()['rates']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch exchange rates: {str(e)}")
            return None

    @staticmethod
    def convert_amount(amount, from_currency, to_currency):
        rates = CurrencyConverter.get_exchange_rates()
        if rates:
            if from_currency != "GBP":
                amount = amount / rates[from_currency]  # Convert to GBP first
            return amount * rates[to_currency]
        return amount  # Fallback if API fails


# Main Application
class FinanceManager:
    def __init__(self, root, user_data):
        self.root = root
        self.user_data = user_data
        self.root.title("Finance Manager")
        self.months = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        self.graph_types = ['Bar Chart', 'Pie Chart', 'Line Chart']
        self.current_sort = {'column': None, 'reverse': False}

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.apply_theme()

        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs
        self.create_home_tab()
        self.create_transactions_tab()
        self.create_forecast_tab()
        self.create_settings_tab()

    def apply_theme(self):
        bg_color = self.user_data.data['settings']['theme']
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color)
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'), background=bg_color)

    def create_home_tab(self):
        self.home_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.home_tab, text='Home')

        # Header
        header = ttk.Frame(self.home_tab)
        header.pack(pady=10, fill=tk.X)
        ttk.Label(header, text=f"Hello: {self.user_data.username}", style='Header.TLabel', font=('Comfortaa', 14)).pack(side=tk.LEFT)
        ttk.Button(header, text="Logout", command=self.logout).pack(side=tk.RIGHT, padx=5)

        # Recent Transactions
        recent_frame = ttk.Frame(self.home_tab)
        recent_frame.pack(pady=10, fill=tk.X)
        ttk.Label(recent_frame, text="Recent Transactions", style='Header.TLabel', font=('Comfortaa', 14)).pack(anchor=tk.W)

        columns = ('date', 'amount', 'description', 'type')
        self.tree = ttk.Treeview(recent_frame, columns=columns, show='headings')
        for col in columns:
            self.tree.heading(col, text=col.capitalize(),
                              command=lambda c=col: self.sort_treeview(self.tree, c))
        self.tree.pack(fill=tk.X)

        # Chart Section
        chart_frame = ttk.Frame(self.home_tab)
        chart_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Graph Type Selection
        graph_control_frame = ttk.Frame(chart_frame)
        graph_control_frame.pack(pady=5)

        self.graph_type_var = tk.StringVar(value=self.user_data.data['settings']['graph_type'])
        ttk.Label(graph_control_frame, text="Graph Type:").pack(side=tk.LEFT)
        graph_combobox = ttk.Combobox(graph_control_frame, textvariable=self.graph_type_var,
                                      values=self.graph_types, width=10)
        graph_combobox.pack(side=tk.LEFT, padx=5)
        graph_combobox.bind('<<ComboboxSelected>>', lambda e: self.update_chart())

        self.month_var = tk.StringVar()
        month_combobox = ttk.Combobox(graph_control_frame, textvariable=self.month_var, values=self.months)
        month_combobox.current(datetime.now().month - 1)
        month_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Button(graph_control_frame, text="Update Chart", command=self.update_chart).pack(side=tk.LEFT, padx=5)

        self.figure = plt.figure(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.load_recent_transactions()
        self.update_chart()

    def create_transactions_tab(self):
        self.transactions_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.transactions_tab, text='Transactions')

        main_pane = ttk.PanedWindow(self.transactions_tab, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left side - Form and List
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame)

        # Transaction Form
        form_frame = ttk.Frame(left_frame)
        form_frame.pack(pady=10, fill=tk.X)

        # Date Components
        ttk.Label(form_frame, text="Day:", font=('Comfortaa', 14)).grid(row=0, column=0, padx=5, sticky=tk.W)
        self.day_entry = ttk.Entry(form_frame, width=5)
        self.day_entry.grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(form_frame, text="Month:", font=('Comfortaa', 14)).grid(row=0, column=2, padx=5, sticky=tk.W)
        self.month_combobox = ttk.Combobox(form_frame, values=self.months, width=10)
        self.month_combobox.current(datetime.now().month - 1)
        self.month_combobox.grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(form_frame, text="Year:", font=('Comfortaa', 14)).grid(row=0, column=4, padx=5, sticky=tk.W)
        self.year_entry = ttk.Entry(form_frame, width=6)
        self.year_entry.insert(0, str(datetime.now().year))
        self.year_entry.grid(row=0, column=5, padx=5, sticky=tk.W)

        ttk.Label(form_frame, text="Amount:", font=('Comfortaa', 14)).grid(row=1, column=0, padx=5, sticky=tk.W)
        self.amount_entry = ttk.Entry(form_frame)
        self.amount_entry.grid(row=1, column=1, columnspan=5, padx=5, sticky=tk.EW)

        ttk.Label(form_frame, text="Description:", font=('Comfortaa', 14)).grid(row=2, column=0, padx=5, sticky=tk.W)
        self.desc_entry = ttk.Entry(form_frame)
        self.desc_entry.grid(row=2, column=1, columnspan=5, padx=5, sticky=tk.EW)

        ttk.Label(form_frame, text="Type:", font=('Comfortaa', 14)).grid(row=3, column=0, padx=5, sticky=tk.W)
        self.type_var = tk.StringVar()
        self.type_combobox = ttk.Combobox(form_frame, textvariable=self.type_var,
                                          values=['Food', 'Rent', 'Utilities', 'Transport',
                                                  'Entertainment', 'Healthcare', 'Other'],
                                          width=14)
        self.type_combobox.grid(row=3, column=1, padx=5, sticky=tk.W)
        self.type_combobox.bind('<<ComboboxSelected>>', self.check_type)

        self.custom_type_entry = ttk.Entry(form_frame, state=tk.DISABLED)
        self.custom_type_entry.grid(row=3, column=2, columnspan=4, padx=5, sticky=tk.EW)

        ttk.Button(form_frame, text="Add Transaction", command=self.add_transaction).grid(
            row=4, column=0, columnspan=6, pady=10, sticky=tk.E)

        # Transaction List
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        columns = ('date', 'amount', 'description', 'type')
        self.transaction_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        for col in columns:
            self.transaction_tree.heading(col, text=col.capitalize(),
                                          command=lambda c=col: self.sort_treeview(self.transaction_tree, c))
        self.transaction_tree.pack(fill=tk.BOTH, expand=True)

        # Right side - Legend
        legend_frame = ttk.Frame(main_pane)
        main_pane.add(legend_frame)

        legend = ttk.LabelFrame(legend_frame, text="Legend", padding=10)
        legend.pack(padx=10, pady=10, fill=tk.BOTH)

        tips = [
            "Use '//' in description to mark no description",
            f"Amounts are in {self.user_data.data['settings']['currency']}",
            "Click column headers to sort",
            "Select 'Other' type for custom categories"
        ]

        for tip in tips:
            ttk.Label(legend, text=tip).pack(anchor=tk.W)

        self.load_all_transactions()

    def create_forecast_tab(self):
        self.forecast_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.forecast_tab, text='3-Year Forecast')

        # Forecast controls
        control_frame = ttk.Frame(self.forecast_tab)
        control_frame.pack(pady=10)
        ttk.Button(control_frame, text="Generate Forecast", command=self.generate_forecast).pack()

        # Forecast display
        self.forecast_tree = ttk.Treeview(self.forecast_tab, columns=('date', 'balance'), show='headings')
        self.forecast_tree.heading('date', text='Month')
        self.forecast_tree.heading('balance', text='Projected Balance')
        self.forecast_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_settings_tab(self):
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text='Settings')

        # Currency Settings
        currency_frame = ttk.LabelFrame(self.settings_tab, text="Currency Settings", padding=10)
        currency_frame.pack(padx=10, pady=5, fill=tk.X)

        self.currency_var = tk.StringVar(value=self.user_data.data['settings']['currency'])
        ttk.Label(currency_frame, text="Select Currency:", font=('Comfortaa', 14)).pack(side=tk.LEFT)
        currency_combobox = ttk.Combobox(currency_frame, textvariable=self.currency_var,
                                         values=['GBP', 'USD', 'EUR', 'JPY'], width=5)
        currency_combobox.pack(side=tk.LEFT, padx=5)
        ttk.Button(currency_frame, text="Save", command=self.save_currency).pack(side=tk.LEFT, padx=5)

        # Theme Settings
        theme_frame = ttk.LabelFrame(self.settings_tab, text="UI Theme", padding=10)
        theme_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(theme_frame, text="Choose Color", command=self.change_theme_color).pack(side=tk.LEFT)

        # Account Actions
        action_frame = ttk.LabelFrame(self.settings_tab, text="Account Actions", padding=10)
        action_frame.pack(padx=10, pady=5, fill=tk.X)

        ttk.Button(action_frame, text="Delete Account", command=self.delete_account).pack(side=tk.LEFT)
        ttk.Button(action_frame, text="Logout", command=self.logout).pack(side=tk.LEFT, padx=5)

    def save_currency(self):
        new_currency = self.currency_var.get()
        old_currency = self.user_data.data['settings']['currency']

        # Convert all transactions to the new currency
        for transaction in self.user_data.get_transactions():
            transaction['amount'] = CurrencyConverter.convert_amount(
                transaction['amount'], old_currency, new_currency
            )

        self.user_data.update_settings({'currency': new_currency})
        self.load_all_transactions()
        self.load_recent_transactions()
        self.update_chart()

    def change_theme_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.user_data.update_settings({'theme': color})
            self.apply_theme()

    def delete_account(self):
        if messagebox.askyesno("Warning", "This will permanently delete all your data!\nAre you sure?"):
            os.remove(self.user_data.data_file)
            self.logout()

    def logout(self):
        self.root.destroy()
        root = tk.Tk()
        LoginSystem(root)
        root.mainloop()

    def check_type(self, event):
        if self.type_var.get() == 'Other':
            self.custom_type_entry.config(state=tk.NORMAL)
        else:
            self.custom_type_entry.config(state=tk.DISABLED)

    def add_transaction(self):
        try:
            day = int(self.day_entry.get())
            month = self.months.index(self.month_combobox.get()) + 1
            year = int(self.year_entry.get())
            date = f"{year}-{month:02d}-{day:02d}"
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Error", "Invalid date components")
            return

        amount = self.amount_entry.get()
        description = self.desc_entry.get().strip() or '//'
        trans_type = self.type_var.get()

        if trans_type == 'Other':
            trans_type = self.custom_type_entry.get().strip()
            if not trans_type:
                messagebox.showerror("Error", "Please enter a custom type")
                return

        try:
            amount = float(amount)
        except ValueError:
            messagebox.showerror("Error", "Invalid amount format")
            return

        transaction = {
            'date': date,
            'amount': amount,
            'description': description,
            'type': trans_type
        }

        self.user_data.add_transaction(transaction)
        self.clear_form()
        self.load_all_transactions()
        self.load_recent_transactions()
        self.update_chart()
        messagebox.showinfo("Success", "Transaction added successfully")

    def clear_form(self):
        self.day_entry.delete(0, tk.END)
        self.month_combobox.current(datetime.now().month - 1)
        self.year_entry.delete(0, tk.END)
        self.year_entry.insert(0, str(datetime.now().year))
        self.amount_entry.delete(0, tk.END)
        self.desc_entry.delete(0, tk.END)
        self.type_combobox.set('')
        self.custom_type_entry.config(state=tk.NORMAL)
        self.custom_type_entry.delete(0, tk.END)
        self.custom_type_entry.config(state=tk.DISABLED)

    def load_recent_transactions(self):
        self.update_treeview(self.tree, self.user_data.get_transactions()[-5:])

    def load_all_transactions(self):
        self.update_treeview(self.transaction_tree, self.user_data.get_transactions())

    def update_treeview(self, treeview, transactions):
        treeview.delete(*treeview.get_children())
        currency = self.user_data.data['settings']['currency']
        for transaction in transactions:
            treeview.insert('', tk.END, values=(
                transaction['date'],
                f"{currency}{transaction['amount']:.2f}",
                transaction['description'],
                transaction['type']
            ))

    def sort_treeview(self, treeview, col):
        data = [(treeview.set(child, col), child) for child in treeview.get_children('')]

        try:
            if col == 'amount':
                data.sort(key=lambda x: float(x[0].replace(self.user_data.data['settings']['currency'], '')),
                          reverse=self.current_sort['reverse'])
            elif col == 'date':
                data.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'),
                          reverse=self.current_sort['reverse'])
            else:
                data.sort(reverse=self.current_sort['reverse'])
        except Exception as e:
            data.sort(reverse=self.current_sort['reverse'])

        for index, (_, child) in enumerate(data):
            treeview.move(child, '', index)

        self.current_sort['reverse'] = not self.current_sort['reverse']
        self.current_sort['column'] = col

    def update_chart(self, event=None):  # Added event=None to handle dropdown selection
        selected_month = self.month_var.get()
        month_number = self.months.index(selected_month) + 1

        filtered = [t for t in self.user_data.get_transactions()
                    if datetime.strptime(t['date'], '%Y-%m-%d').month == month_number]

        type_amounts = {}
        for trans in filtered:
            t_type = trans['type']
            type_amounts[t_type] = type_amounts.get(t_type, 0) + trans['amount']

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if type_amounts:
            graph_type = self.graph_type_var.get().lower()
            if 'bar' in graph_type:
                ax.bar(type_amounts.keys(), type_amounts.values())
                ax.set_ylabel('Amount')
            elif 'pie' in graph_type:
                ax.pie(type_amounts.values(), labels=type_amounts.keys(), autopct='%1.1f%%')
            elif 'line' in graph_type:
                ax.plot(type_amounts.keys(), type_amounts.values(), marker='o')
                ax.set_ylabel('Amount')

            ax.set_title(f'Spending Distribution - {selected_month}')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center')

        self.canvas.draw()

    def generate_forecast(self):
        try:
            # Aggregate monthly balances (transactions + income)
            transactions = self.user_data.get_transactions()
            monthly_balances = {}

            # Add transactions and income
            for t in transactions:
                month = datetime.strptime(t['date'], '%Y-%m-%d').strftime('%Y-%m')
                monthly_balances[month] = monthly_balances.get(month, 0) + t['amount']
            monthly_income = self.user_data.data['income']['monthly_breakdown']
            for month_num, amount in monthly_income.items():
                year = datetime.now().year
                month_key = datetime(year, int(month_num), 1).strftime('%Y-%m')
                monthly_balances[month_key] = monthly_balances.get(month_key, 0) + amount

            # Sort and prepare data
            sorted_balances = sorted(monthly_balances.items(), key=lambda x: x[0])
            if len(sorted_balances) < 12:
                messagebox.showwarning("Warning", "At least 12 months of data required")
                return

            dates = [item[0] for item in sorted_balances]
            balances = np.array([item[1] for item in sorted_balances])

            # Split into training/validation (last 12 months for validation)
            validation_size = 12
            train_data = balances[:-validation_size] if len(balances) > validation_size else balances
            val_data = balances[-validation_size:] if len(balances) > validation_size else None

            # Initialize models and forecasts
            models = {
                "Holt-Winters": HoltWinters(),
                "Linear Regression": None,
                "Moving Average": None
            }
            forecasts = {}
            maes = {}

            # Holt-Winters
            try:
                hw = HoltWinters()
                hw.fit(train_data)
                hw_forecast = hw.forecast(36)
                forecasts["Holt-Winters"] = hw_forecast
                if val_data is not None:
                    hw_val = hw.forecast(validation_size)
                    maes["Holt-Winters"] = np.mean(np.abs(hw_val - val_data))
            except Exception as e:
                maes["Holt-Winters"] = np.inf

            # Linear Regression
            try:
                lr_forecast = linear_regression_forecast(train_data, 36)
                forecasts["Linear Regression"] = lr_forecast
                if val_data is not None:
                    lr_val = linear_regression_forecast(train_data, validation_size)
                    maes["Linear Regression"] = np.mean(np.abs(lr_val - val_data))
            except Exception as e:
                maes["Linear Regression"] = np.inf

            # Moving Average
            try:
                ma_forecast = moving_average_forecast(train_data, 36)
                forecasts["Moving Average"] = ma_forecast
                if val_data is not None:
                    ma_val = moving_average_forecast(train_data, validation_size)
                    maes["Moving Average"] = np.mean(np.abs(ma_val - val_data))
            except Exception as e:
                maes["Moving Average"] = np.inf

            # Select best model
            best_model = min(maes, key=maes.get) if maes else "Holt-Winters"
            best_forecast = forecasts[best_model]

            # Update UI with best forecast
            self.forecast_tree.delete(*self.forecast_tree.get_children())
            currency = self.user_data.data['settings']['currency']
            current_date = datetime.now().date()

            for i, amount in enumerate(best_forecast):
                current_date += relativedelta(months=1)
                self.forecast_tree.insert('', tk.END,
                                          values=(current_date.strftime('%Y-%m'),
                                                  f"{currency}{amount:.2f}",
                                                  best_model))

            # Show model comparison
            comparison = "\n".join([f"{model}: MAE = {mae:.2f}" for model, mae in maes.items()])
            messagebox.showinfo("Model Comparison", f"Best Model: {best_model}\n\n{comparison}")

        except Exception as e:
            messagebox.showerror("Error", f"Forecast failed: {str(e)}")


# Login System
class LoginSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack()

        ttk.Label(frame, text="Username:").grid(row=0, column=0, pady=5, sticky=tk.W)
        self.username_entry = ttk.Entry(frame)
        self.username_entry.grid(row=0, column=1, pady=5)

        ttk.Label(frame, text="Password:").grid(row=1, column=0, pady=5, sticky=tk.W)
        self.password_entry = ttk.Entry(frame, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Login", command=self.login).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Sign Up", command=self.signup).pack(side=tk.LEFT, padx=5)

    def validate_credentials(self, username, password):
        if not username or not password:
            messagebox.showerror("Error", "Username and password cannot be empty")
            return False
        return True

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if not self.validate_credentials(username, password):
            return

        if not os.path.exists(f"{username}.json"):
            messagebox.showerror("Error", "User does not exist")
            return

        with open(f"{username}.json", 'r') as f:
            user_data = json.load(f)

        if user_data.get('password') != password:
            messagebox.showerror("Error", "Incorrect password")
            return

        self.root.destroy()
        root = tk.Tk()
        FinanceManager(root, UserData(username))
        root.mainloop()

    def signup(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if not self.validate_credentials(username, password):
            return

        if os.path.exists(f"{username}.json"):
            messagebox.showerror("Error", "Username already exists")
            return

        user_data = {
            'password': password,
            'transactions': [],
            'settings': {
                'currency': 'GBP',
                'theme': '#f0f0f0',
                'graph_type': 'bar'
            }
        }

        with open(f"{username}.json", 'w') as f:
            json.dump(user_data, f)

        messagebox.showinfo("Success", "Account created successfully")


if __name__ == "__main__":
    root = tk.Tk()
    LoginSystem(root)
    root.mainloop()
