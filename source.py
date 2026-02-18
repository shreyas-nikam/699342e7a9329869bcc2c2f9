import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time # Needed for timing comparison

# --- 1. Global Configuration and Constants ---

def configure_plot_aesthetics():
    """Configures matplotlib and seaborn plot aesthetics."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100

# System prompt for the rebalancing agent
REBALANCE_SYSTEM_PROMPT = """
You are a portfolio operations agent named "OptiWealth Rebalance Agent". Your primary goal is to compute rebalancing trades for an investment portfolio, ensuring all calculations are precise and constraints are met.

PROCESS:
1. Get current holdings using the 'get_holdings' tool.
2. Get current prices for all holdings using the 'get_current_prices' tool.
3. Determine the target allocation. For this run, the target allocation is an equal-weight policy (10% each) for all positions, which you will receive directly as JSON content for the 'calculate_trades' tool.
4. Use the 'calculate_trades' tool to compute the exact trades needed.
5. Use the 'check_constraints' tool to verify all pre-defined portfolio constraints are met.
6. If constraints are violated, you should attempt to explain the violation and suggest a conceptual adjustment (e.g., "reduce turnover" or "scale down largest buy trades"). *However, for this lab, do not attempt to programmatically adjust trades; simply report the violations.*
7. Present the final trade ticket for human approval.

CRITICAL RULES:
- NEVER compute arithmetic yourself. ALWAYS use the calculator tools provided for any numerical calculation.
- All trades must be whole shares (no fractional shares). The 'calculate_trades' tool handles this.
- Report all dollar amounts in USD, rounded to two decimal places.
- The final trade ticket must clearly include: ticker, action (BUY/SELL), shares, approximate dollar amount, current weight, target weight, and drift.
- End the final trade ticket with "STATUS: PENDING HUMAN APPROVAL". Never mark as executed.
"""

# Tool schemas for OpenAI function calling
REBALANCE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_holdings",
            "description": "Reads current portfolio positions (ticker, shares, avg_cost) from a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings_file_path": {"type": "string", "description": "Path to the CSV file containing portfolio holdings."}
                },
                "required": ["holdings_file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_prices",
            "description": "Fetches current market prices for a comma-separated list of tickers.",
            "parameters": {
                "type": "object",
                "properties": {"tickers": {"type": "string", "description": "Comma-separated list of ticker symbols"}},
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_trades",
            "description": "Computes rebalancing trades given current holdings, live prices, and target weights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings_json": {"type": "string", "description": "JSON string of current portfolio holdings"},
                    "prices_json": {"type": "string", "description": "JSON string of current market prices"},
                    "target_weights_json": {"type": "string", "description": "JSON string of target allocation weights"},
                    "total_value": {"type": "number", "description": "Optional: total portfolio value to use. If None, calculated from holdings and prices."}
                },
                "required": ["holdings_json", "prices_json", "target_weights_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_constraints",
            "description": "Verifies post-trade constraints (e.g., max position weight, turnover, min trade size) against proposed trades.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trades_summary_json": {"type": "string", "description": "JSON string of the trades summary output from calculate_trades"},
                    "max_position_pct": {"type": "number", "description": "Maximum allowed weight for any single position (e.g., 10.0 for 10%)"},
                    "max_turnover_pct": {"type": "number", "description": "Maximum allowed portfolio turnover percentage (e.g., 15.0 for 15%)"},
                    "min_trade_usd": {"type": "number", "description": "Minimum dollar amount for a trade (trades below this might be flagged or skipped)"}
                },
                "required": ["trades_summary_json"],
            },
        },
    },
]

# --- 2. Core Rebalancing Tools (Functions) ---

def get_holdings(holdings_file_path: str = 'portfolio_holdings.csv') -> str:
    """Reads current portfolio holdings from a specified CSV file."""
    try:
        holdings = pd.read_csv(holdings_file_path)
        return holdings.to_json(orient='records')
    except FileNotFoundError:
        return json.dumps({"error": f"{holdings_file_path} not found"})
    except Exception as e:
        return json.dumps({"error": f"Error reading holdings file: {e}"})

def get_current_prices(tickers: str) -> str:
    """Fetches current market prices for a comma-separated list of tickers using yfinance."""
    ticker_list = [t.strip() for t in tickers.split(',')]
    prices = {}
    for t in ticker_list:
        try:
            stock = yf.Ticker(t)
            # Use currentPrice or previousClose if currentPrice is not available
            info = stock.info
            prices[t] = round(info.get('currentPrice', info.get('previousClose', 0)), 2)
            if prices[t] == 0:
                print(f"Warning: Could not fetch price for {t}. Using 0.")
        except Exception as e:
            prices[t] = 0
            print(f"Error fetching price for {t}: {e}. Using 0.")
    return json.dumps(prices)

def calculate_trades(holdings_json: str, prices_json: str, target_weights_json: str, total_value: float = None) -> str:
    """
    Computes trades needed to rebalance to target weights.
    Returns: list of {ticker, action, shares, dollar_amount, current_weight, target_weight, drift_pct}
    """
    holdings = json.loads(holdings_json)
    prices = json.loads(prices_json)
    targets = json.loads(target_weights_json)

    current_values = {}
    for h in holdings:
        ticker = h['ticker']
        price = prices.get(ticker, 0)
        current_values[ticker] = h['shares'] * price

    if total_value is None:
        total_value = sum(current_values.values())
        if total_value == 0:
            return json.dumps({"error": "Total portfolio value is zero, cannot calculate trades."})

    current_weights = {t: v / total_value for t, v in current_values.items()}

    trades = []
    all_tickers = set(list(targets.keys()) + list(current_weights.keys()))

    for ticker in sorted(list(all_tickers)): # Sort for consistent output
        target_w = targets.get(ticker, 0)
        current_w = current_weights.get(ticker, 0)

        drift = target_w - current_w
        dollar_trade = drift * total_value

        price = prices.get(ticker, 1e-6) # Avoid division by zero
        share_trade = 0
        if price > 0:
            share_trade = int(dollar_trade / price) # Floor to whole shares, can be negative

        # Only add trade if significant shares need to be bought/sold
        if abs(share_trade) > 0:
            # Recalculate dollar_amount based on actual whole shares traded
            actual_dollar_trade = share_trade * price

            trades.append({
                'ticker': ticker,
                'action': 'BUY' if actual_dollar_trade > 0 else 'SELL',
                'shares': abs(share_trade),
                'dollar_amount': round(abs(actual_dollar_trade), 2),
                'current_weight': round(current_w * 100, 2),
                'target_weight': round(target_w * 100, 2),
                'drift_pct': round(drift * 100, 2),
            })

    # Calculate post-trade weights for constraint checking and visualization
    post_trade_values = current_values.copy()
    for trade in trades:
        ticker = trade['ticker']
        direction_factor = 1 if trade['action'] == 'BUY' else -1
        # Calculate new shares based on current holdings + shares to trade
        current_shares = next((h['shares'] for h in holdings if h['ticker'] == ticker), 0)
        new_shares = current_shares + (direction_factor * trade['shares'])
        post_trade_values[ticker] = new_shares * prices.get(ticker, 0)

    post_total_value = sum(post_trade_values.values())
    post_trade_weights = {t: v / post_total_value if post_total_value > 0 else 0 for t, v in post_trade_values.items()}

    total_turnover_usd = sum(t['dollar_amount'] for t in trades)
    total_turnover_pct = (total_turnover_usd / total_value * 100) if total_value > 0 else 0

    summary = {
        'total_portfolio_value': round(total_value, 2),
        'total_turnover_pct': round(total_turnover_pct, 2),
        'n_trades': len(trades),
        'trades': sorted(trades, key=lambda x: x['dollar_amount'], reverse=True),
        'post_trade_weights': {t: round(w*100, 2) for t, w in post_trade_weights.items()}
    }
    return json.dumps(summary, indent=2)

def check_constraints(trades_summary_json: str, max_position_pct: float = 10.0,
                      max_turnover_pct: float = 15.0, min_trade_usd: float = 1000.0) -> str:
    """
    Verifies post-trade constraints are satisfied.
    Sector exposure is not included here as 'sector' data is not in holdings.csv.
    """
    trades_summary_data = json.loads(trades_summary_json)
    violations = []
    warnings = []

    # Check turnover
    turnover = trades_summary_data.get('total_turnover_pct', 0)
    if turnover > max_turnover_pct:
        violations.append(
            f"TURNOVER: {turnover:.1f}% exceeds max {max_turnover_pct:.1f}%"
        )

    # Check individual position limits based on post-trade weights
    post_trade_weights = trades_summary_data.get('post_trade_weights', {})
    for ticker, weight in post_trade_weights.items():
        if weight > max_position_pct:
            violations.append(
                f"POSITION: {ticker} target {weight:.1f}% > max {max_position_pct:.1f}%"
            )

    # Check minimum trade size
    for trade in trades_summary_data.get('trades', []):
        if trade['dollar_amount'] < min_trade_usd:
            warnings.append(
                f"MIN_TRADE: {trade['ticker']} ${trade['dollar_amount']:.0f} < min ${min_trade_usd:.0f} (may skip)"
            )

    result = {
        'all_constraints_met': len(violations) == 0,
        'violations': violations,
        'warnings': warnings,
        'constraints_checked': {
            'max_position_pct': max_position_pct,
            'max_turnover_pct': max_turnover_pct,
            'min_trade_usd': min_trade_usd,
        }
    }
    return json.dumps(result, indent=2)

# Mapping of tool names to actual functions
TOOLS = {
    "get_holdings": get_holdings,
    "get_current_prices": get_current_prices,
    "calculate_trades": calculate_trades,
    "check_constraints": check_constraints,
}

# --- 3. LLM Agent Orchestration ---

def run_rebalancing_agent(openai_client: OpenAI, goal: str, holdings_file_path: str, target_allocation_policy_json_str: str, max_iterations: int = 10) -> dict:
    """Executes the rebalancing agent workflow."""
    messages = [
        {"role": "system", "content": REBALANCE_SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    trace = []

    for iteration in range(max_iterations):
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=REBALANCE_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = response.choices[0].message
        messages.append(msg)
        trace.append({"role": "assistant", "message": msg.model_dump_json()})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                function_name = tc.function.name
                function_args = json.loads(tc.function.arguments)

                # Special handling for get_holdings: inject the actual holdings file path
                if function_name == "get_holdings":
                    if "holdings_file_path" not in function_args or not function_args["holdings_file_path"]:
                        function_args["holdings_file_path"] = holdings_file_path

                # Special handling for calculate_trades: inject the actual target allocation JSON content.
                # The prompt implies the agent will somehow get the target allocation.
                # We inject the pre-loaded target allocation policy content here directly.
                if function_name == "calculate_trades":
                    function_args["target_weights_json"] = target_allocation_policy_json_str

                if function_name in TOOLS:
                    print(f"\n[{iteration}] Calling tool: {function_name} with args: {function_args}")
                    tool_output = TOOLS[function_name](**function_args)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": tool_output}
                    )
                    trace.append({"role": "tool", "name": function_name, "args": function_args, "result": tool_output[:500] + "..." if len(tool_output) > 500 else tool_output})
                    print(f"[{iteration}] Tool {function_name} output received.")
                else:
                    print(f"Error: Tool {function_name} not found.")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": f"Tool {function_name} not found."})}
                    )
                    trace.append({"role": "tool", "name": function_name, "args": function_args, "result": "Error: Tool not found"})
        else:
            return {'trade_ticket': msg.content, 'trace': trace, 'iterations': iteration + 1}

    return {'trade_ticket': 'Max iterations reached without generating a final trade ticket.', 'trace': trace, 'iterations': max_iterations}


# --- 4. Human Approval Simulation ---

def simulate_human_approval(trade_ticket_content: str, rebalancing_summary: dict, auto_approve: bool = True) -> bool:
    """
    Simulates a human approval process for the trade ticket.
    In a production environment, this would be a UI widget or email notification.
    Includes safety checks before prompting for approval.
    """
    print("\n" + "=" * 60)
    print("TRADE TICKET - PENDING HUMAN APPROVAL")
    print("=" * 60)
    
    # Construct a more readable trade ticket if rebalancing_summary is available
    if rebalancing_summary and 'trades' in rebalancing_summary:
        trades_df = pd.DataFrame(rebalancing_summary['trades'])
        formatted_ticket = f"Portfolio Value: ${rebalancing_summary['total_portfolio_value']:.2f}\n"
        formatted_ticket += f"Total Turnover: {rebalancing_summary['total_turnover_pct']:.2f}%\n"
        formatted_ticket += "\nProposed Trades:\n"
        formatted_ticket += trades_df[['ticker', 'action', 'shares', 'dollar_amount', 'current_weight', 'target_weight']].to_string(index=False)
        formatted_ticket += "\n\nSTATUS: PENDING HUMAN APPROVAL"
        print(formatted_ticket)
    else:
        print(trade_ticket_content) # Fallback to agent's raw output
    
    print("\n" + "=" * 60)

    # Verify key safety checks from the trade ticket content
    checks = {
        'has_pending_status': 'PENDING' in trade_ticket_content.upper(),
        'has_buy_sell_keywords': any(kw in trade_ticket_content.upper() for kw in ('BUY', 'SELL')),
        'has_share_counts': any(char.isdigit() for char in trade_ticket_content) and 'shares' in trade_ticket_content.lower(),
        'no_executed_claim': 'EXECUTED' not in trade_ticket_content.upper(),
    }
    
    # Additional check for positive dollar amounts, needs a robust parser or rely on rebalancing_summary
    if rebalancing_summary and 'trades' in rebalancing_summary:
        checks['has_positive_dollar_amounts'] = all(t['dollar_amount'] >= 0 for t in rebalancing_summary['trades'])
    else:
        checks['has_positive_dollar_amounts'] = True # Assume true if summary not available, or implement more complex regex

    all_safe = all(checks.values())

    print("\n--- SAFETY CHECKS (Pre-Approval) ---")
    for check_name, passed in checks.items():
        print(f" {'PASS' if passed else 'FAIL'}: {check_name}")

    if not all_safe:
        print("\nBLOCKED: Safety checks failed. Cannot approve.")
        return False

    if auto_approve:
        approval = "yes"
        print(f"\nSimulating human input: '{approval}'")
    else:
        approval = input("\nApprove trades? (yes/no): ").lower()

    if approval == 'yes':
        print("\nSTATUS: APPROVED BY HUMAN")
        print("Trades would now be sent to execution system (if integrated).")
        return True
    else:
        print("\nSTATUS: REJECTED BY HUMAN")
        print("No trades executed.")
        return False

# --- 5. Visualization Functions ---

def plot_rebalance_weights(current_weights: dict, target_weights: dict, post_trade_weights: dict, title: str = 'Pre/Post Rebalance Weight Comparison') -> plt.Figure:
    """
    Generates a bar plot comparing current, target, and post-rebalance portfolio weights.
    Returns a matplotlib Figure object.
    """
    configure_plot_aesthetics() # Apply aesthetics
    all_tickers = sorted(list(set(current_weights.keys()) | set(target_weights.keys()) | set(post_trade_weights.keys())))

    plot_data = []
    for ticker in all_tickers:
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Current', 'Weight': current_weights.get(ticker, 0)})
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Target', 'Weight': target_weights.get(ticker, 0)})
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Post-Rebalance', 'Weight': post_trade_weights.get(ticker, 0)})

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x='Ticker', y='Weight', hue='Weight Type', data=plot_df, palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Weight (%)')
    ax.set_xlabel('Asset Ticker')
    ax.legend(title='Weight Type')
    ax.set_ylim(0, max(plot_df['Weight'].max() * 1.1, 15))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig

def plot_turnover_pie_chart(rebalancing_summary: dict, title: str = 'Distribution of Rebalancing Turnover (Dollar Amount)') -> plt.Figure:
    """
    Generates a pie chart for rebalancing turnover.
    Returns a matplotlib Figure object.
    """
    configure_plot_aesthetics() # Apply aesthetics
    if not rebalancing_summary or 'trades' not in rebalancing_summary:
        print("No rebalancing summary or trades data available for turnover pie chart.")
        return None

    buy_turnover = sum(t['dollar_amount'] for t in rebalancing_summary['trades'] if t['action'] == 'BUY')
    sell_turnover = sum(t['dollar_amount'] for t in rebalancing_summary['trades'] if t['action'] == 'SELL')

    turnover_components = {
        'Buy Turnover': buy_turnover,
        'Sell Turnover': sell_turnover,
    }

    filtered_turnover_components = {k: v for k, v in turnover_components.items() if v > 0}

    if filtered_turnover_components:
        labels = filtered_turnover_components.keys()
        sizes = filtered_turnover_components.values()
        colors = sns.color_palette('pastel')[0:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'${(p/100)*sum(sizes):,.2f}', startangle=90, pctdistance=0.85)
        ax.set_title(title)
        ax.axis('equal')
        fig.tight_layout()
        return fig
    else:
        print("No significant turnover for pie chart visualization.")
        return None

# --- 6. Demo Data Generation (for script execution) ---

def _generate_demo_data(holdings_file: str = 'portfolio_holdings.csv', target_alloc_file: str = 'target_allocation_policy.json'):
    """Creates dummy portfolio holdings CSV and target allocation JSON files."""
    # Create a dummy portfolio holdings CSV file for demonstration
    portfolio_data = {
        'ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'BRK-B', 'UNH', 'V'],
        'shares': [150, 200, 80, 100, 250, 180, 300, 120, 90, 130],
        'avg_cost': [145.00, 310.00, 140.00, 155.00, 160.00, 95.00, 350.00, 480.00, 250.00, 145.00]
    }
    portfolio_holdings_df = pd.DataFrame(portfolio_data)
    portfolio_holdings_df.to_csv(holdings_file, index=False)
    print(f"Generated '{holdings_file}':")
    print(portfolio_holdings_df.head())

    # Define a target asset allocation policy (e.g., equal weight across all positions)
    target_allocation_policy = {
        'AAPL': 0.10, 'MSFT': 0.10, 'AMZN': 0.10, 'GOOGL': 0.10, 'JPM': 0.10,
        'JNJ': 0.10, 'XOM': 0.10, 'BRK-B': 0.10, 'UNH': 0.10, 'V': 0.10
    }
    with open(target_alloc_file, 'w') as f:
        json.dump(target_allocation_policy, f, indent=2)
    print(f"\nGenerated '{target_alloc_file}':")
    print(json.dumps(target_allocation_policy, indent=2))
    return portfolio_holdings_df, target_allocation_policy

# --- 7. Main Workflow Orchestration Function ---

def perform_rebalancing_workflow(
    openai_key: str,
    rebalance_goal: str,
    holdings_file_path: str,
    target_allocation_file_path: str,
    max_position_weight_pct: float = 12.0,
    max_turnover_pct: float = 15.0,
    min_trade_usd: float = 1000.0,
    auto_approve_trades: bool = True
) -> dict:
    """
    Orchestrates the entire rebalancing workflow using the LLM agent and tools.

    Returns a dictionary containing the agent's output, compliance report, and plot figures.
    """
    if not openai_key or openai_key == "YOUR_OPENAI_KEY":
        raise ValueError("OpenAI API key is missing or invalid. Please set OPENAI_API_KEY environment variable.")

    openai_client = OpenAI(api_key=openai_key)

    # Load target allocation policy as JSON string for the agent
    with open(target_allocation_file_path, 'r') as f:
        target_allocation_policy_dict = json.load(f)
    target_allocation_policy_json_str = json.dumps(target_allocation_policy_dict)

    print(f"\n--- Running OptiWealth Rebalance Agent for: {rebalance_goal} ---")
    agent_result = run_rebalancing_agent(
        openai_client=openai_client,
        goal=rebalance_goal,
        holdings_file_path=holdings_file_path,
        target_allocation_policy_json_str=target_allocation_policy_json_str
    )

    print(f"\n--- Agent Execution Completed in {agent_result['iterations']} iterations ---")
    print("\n--- Raw Trade Ticket Proposal from Agent ---")
    print(agent_result['trade_ticket'])

    print("\n--- Agent Reasoning Trace (for auditability) ---")
    for entry in agent_result['trace']:
        if entry['role'] == 'assistant':
            print(f"ASSISTANT: {entry['message']}")
        elif entry['role'] == 'tool':
            print(f"TOOL CALL: {entry['name']}({entry['args']}) -> {entry['result']}")

    # Extract the trades summary from the agent's output for explicit constraint checking
    rebalancing_summary = None
    constraints_report = None
    trades_summary_json_output = None
    for entry in reversed(agent_result['trace']):
        if entry['role'] == 'tool' and entry['name'] == 'calculate_trades':
            trades_summary_json_output = entry['result']
            break

    if trades_summary_json_output:
        try:
            rebalancing_summary = json.loads(trades_summary_json_output)

            print(f"\n--- OptiWealth Compliance Check Report ---")
            print(f"Checking against: Max Position={max_position_weight_pct}%, Max Turnover={max_turnover_pct}%, Min Trade=${min_trade_usd}")

            constraints_report_json = check_constraints(
                trades_summary_json=trades_summary_json_output,
                max_position_pct=max_position_weight_pct,
                max_turnover_pct=max_turnover_pct,
                min_trade_usd=min_trade_usd
            )
            constraints_report = json.loads(constraints_report_json)

            print("\nCompliance Status:")
            if constraints_report['all_constraints_met']:
                print("🟢 All key constraints are met!")
            else:
                print("🔴 **WARNING: Constraint VIOLATIONS detected!**")
                for violation in constraints_report['violations']:
                    print(f"- VIOLATION: {violation}")

            if constraints_report['warnings']:
                print("\nWarnings/Minor Issues:")
                for warning in constraints_report['warnings']:
                    print(f"- WARNING: {warning}")

            print("\nConstraints Checked Summary:")
            for k, v in constraints_report['constraints_checked'].items():
                print(f"- {k}: {v}")

        except json.JSONDecodeError as e:
            print(f"Error parsing agent's trade ticket or tool output: {e}")
            print("Please ensure the agent's final output or the calculate_trades tool output is valid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred during constraint checking: {e}")
    else:
        print("Could not find trades summary in agent's trace to perform explicit constraint check.")

    approved = False
    if rebalancing_summary:
        approved = simulate_human_approval(agent_result['trade_ticket'], rebalancing_summary, auto_approve=auto_approve_trades)
    else:
        print("\nCannot proceed to human approval as rebalancing summary was not generated successfully.")

    # Generate visualizations
    weight_fig = None
    turnover_fig = None
    if rebalancing_summary:
        current_holdings_df = pd.read_csv(holdings_file_path)
        tickers = ','.join(current_holdings_df['ticker'].tolist())
        current_prices = json.loads(get_current_prices(tickers))

        current_values_dict = {row['ticker']: row['shares'] * current_prices.get(row['ticker'], 0)
                               for idx, row in current_holdings_df.iterrows()}
        total_current_value = sum(current_values_dict.values())
        current_weights_viz = {t: (v / total_current_value * 100) for t, v in current_values_dict.items()}

        target_weights_viz = {k: v * 100 for k, v in target_allocation_policy_dict.items()}
        post_trade_weights_viz = rebalancing_summary['post_trade_weights']

        weight_fig = plot_rebalance_weights(current_weights_viz, target_weights_viz, post_trade_weights_viz)
        turnover_fig = plot_turnover_pie_chart(rebalancing_summary)
    else:
        print("\nSkipping visualizations as rebalancing summary was not generated successfully.")

    return {
        'agent_result': agent_result,
        'rebalancing_summary': rebalancing_summary,
        'constraints_report': constraints_report,
        'approved': approved,
        'weight_plot_figure': weight_fig,
        'turnover_plot_figure': turnover_fig
    }


def scripted_rebalancing(holdings_file_path: str, target_allocation_file_path: str, constraint_params: dict) -> tuple:
    """Pure Python rebalancing logic, without LLM orchestration."""
    start_time = time.time()

    holdings_json_data = get_holdings(holdings_file_path=holdings_file_path)
    holdings = json.loads(holdings_json_data)
    tickers_list = [h['ticker'] for h in holdings]
    tickers_str = ','.join(tickers_list)

    prices_json_data = get_current_prices(tickers_str)

    # Load target allocation from the predefined file
    with open(target_allocation_file_path, 'r') as f:
        target_allocation = json.load(f)

    target_weights_json_data = json.dumps(target_allocation)

    trades_summary_json_data = calculate_trades(
        holdings_json=holdings_json_data,
        prices_json=prices_json_data,
        target_weights_json=target_weights_json_data
    )

    constraints_json_data = check_constraints(
        trades_summary_json=trades_summary_json_data,
        **constraint_params
    )

    elapsed_time = time.time() - start_time
    return trades_summary_json_data, constraints_json_data, elapsed_time


# --- 8. Main Execution Block (for direct script running / demonstration) ---

if __name__ == "__main__":
    # Load OpenAI API Key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None or openai_api_key == "YOUR_OPENAI_KEY":
        print("WARNING: OPENAI_API_KEY environment variable not set or using placeholder. Please set it for full functionality.")
        # Attempt to use placeholder for demonstration if not set
        openai_api_key = "YOUR_OPENAI_KEY" 

    # Define file paths for demo data
    HOLDINGS_CSV = 'portfolio_holdings.csv'
    TARGET_ALLOCATION_JSON = 'target_allocation_policy.json'

    # Generate dummy data for demonstration
    _generate_demo_data(HOLDINGS_CSV, TARGET_ALLOCATION_JSON)

    # Define constraint parameters
    constraint_parameters = {
        'max_position_pct': 12.0,
        'max_turnover_pct': 15.0,
        'min_trade_usd': 1000.0,
    }

    # Alex's goal: Rebalance the portfolio to an equal-weight allocation across all positions.
    rebalance_goal = "Rebalance the portfolio to equal-weight (10% each) across all positions, then check constraints."

    try:
        # --- Run LLM Agent Workflow ---
        agent_orchestration_result = perform_rebalancing_workflow(
            openai_key=openai_api_key,
            rebalance_goal=rebalance_goal,
            holdings_file_path=HOLDINGS_CSV,
            target_allocation_file_path=TARGET_ALLOCATION_JSON,
            **constraint_parameters,
            auto_approve_trades=True # Set to False to manually approve during script execution
        )

        # Display plots if figures were generated
        if agent_orchestration_result['weight_plot_figure']:
            agent_orchestration_result['weight_plot_figure'].show()
        if agent_orchestration_result['turnover_plot_figure']:
            agent_orchestration_result['turnover_plot_figure'].show()

        # --- Timing Comparison ---
        print("\n--- REBALANCING APPROACH COMPARISON ---")
        print("=" * 60)

        # Approach A: Manual spreadsheet (simulated timing)
        manual_time = 25 * 60 # seconds

        # Approach B: Pure Python script
        script_trades, script_constraints, script_time = scripted_rebalancing(
            holdings_file_path=HOLDINGS_CSV,
            target_allocation_file_path=TARGET_ALLOCATION_JSON,
            constraint_params=constraint_parameters
        )

        # Approach C: LLM Agent
        # Assuming ~3 seconds per iteration for LLM calls + tool execution overhead
        agent_estimated_time = agent_orchestration_result['agent_result']['iterations'] * 3 # seconds

        print(f"{'Approach':<25s} {'Time':>10s} {'Flexibility':>12s} {'Accuracy':>12s} {'Auditability':>12s}")
        print("-" * 60)
        print(f"{'Manual (spreadsheet)':<25s} {f'{manual_time/60:.0f} min':>10s} {'Low':>12s} {'Medium':>12s} {'Low':>12s}")
        print(f"{'Pure Python Script':<25s} {f'{script_time:.2f}s':>10s} {'Low':>12s} {'Perfect':>12s} {'High':>12s}")
        print(f"{'OptiWealth AI Agent':<25s} {f'~{agent_estimated_time:.0f}s':>10s} {'High':>12s} {'High*':>12s} {'High':>12s}")
        print("\n* Agent accuracy depends on tool delegation (arithmetic by tools). Without tools, accuracy would be Low.")
        print("\nKEY INSIGHT:")
        print("- For FIXED rebalancing rules (e.g., equal-weight to target), a pure Python script is FASTER and more ACCURATE.")
        print("- The AI agent adds value when:")
        print("  - The rebalancing rule is described in NATURAL LANGUAGE (e.g., 'underweight energy, overweight tech, keep cash at 5%').")
        print("  - CONSTRAINTS CHANGE dynamically (e.g., 'also minimize tax impact this quarter').")
        print("  - HUMAN COMMUNICATION is needed (e.g., explaining the trades to a client).")
        print("\nConclusion for OptiWealth:")
        print("  - Use dedicated scripts for standardized, repeatable rebalancing with fixed rules.")
        print("  - Leverage the AI agent for flexible, context-dependent rebalancing that requires natural language interpretation, dynamic constraint handling, and client-ready explanations.")

    except ValueError as e:
        print(f"Error during rebalancing workflow: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
    finally:
        # Clean up dummy files (optional)
        if os.path.exists(HOLDINGS_CSV):
            os.remove(HOLDINGS_CSV)
        if os.path.exists(TARGET_ALLOCATION_JSON):
            os.remove(TARGET_ALLOCATION_JSON)
        print(f"\nCleaned up dummy files: {HOLDINGS_CSV}, {TARGET_ALLOCATION_JSON}")
