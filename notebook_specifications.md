
# Portfolio Rebalancing Agent with Human-in-the-Loop: An Applied Workflow for Investment Professionals

**Persona:** Alex Chen, Senior Portfolio Operations Specialist at OptiWealth Asset Management.

**Organization:** OptiWealth Asset Management is a boutique investment firm managing a diverse range of client portfolios.

**Scenario:** Alex, a CFA Charterholder, faces the recurring challenge of rebalancing client portfolios to target asset allocations. This task is crucial for maintaining investment policy compliance but is notoriously time-consuming and prone to manual errors, especially with numerous positions, complex constraints (e.g., position limits, turnover budgets), and frequently changing market prices. OptiWealth is exploring how AI agents can automate the quantitative aspects of rebalancing while ensuring compliance and critical human oversight.

This notebook demonstrates a real-world workflow where an AI agent assists Alex in this high-stakes task. Instead of manually calculating trades and checking constraints, Alex will leverage a set of specialized tools orchestrated by an AI agent. The agent will propose trades, verify compliance, and present a detailed "Trade Ticket" for Alex's explicit review and approval, embodying the "human-in-the-loop" principle. This approach aims to significantly increase efficiency, accuracy, and compliance, allowing Alex to focus on strategic judgment rather than tedious calculations.

---

## 1. Setup: Installing Libraries and Importing Dependencies

Alex begins by setting up his environment, installing the necessary Python libraries for data handling, financial data fetching, AI agent orchestration, and visualization.

```python
!pip install pandas numpy yfinance openai matplotlib seaborn python-dotenv # python-dotenv for API key if not set as environment variable
```

```python
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load environment variables if needed (e.g., for OPENAI_API_KEY)
# from dotenv import load_dotenv
# load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Configure plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
```

---

## 2. The Problem: Manual Rebalancing & Introducing Our Quantitative Tool Suite

Alex often deals with portfolios that drift from their target allocations due to market movements. Manually calculating the necessary buy/sell trades and ensuring they adhere to a multitude of client-specific and regulatory constraints is a time-consuming and error-prone process. To address this, OptiWealth has developed a suite of robust, dedicated Python tools that can perform precise financial calculations and data fetching, ensuring reliability where Large Language Models (LLMs) might falter.

The core rebalancing logic involves comparing current portfolio weights to target weights to determine the necessary dollar amount for each trade, then converting that into shares. This process is governed by the formula:

$$\Delta_i = (w_i^{target} - w_i^{current}) \times V_{portfolio}$$

Where:
- $\Delta_i$ is the dollar amount to trade for asset $i$.
- $w_i^{target}$ is the target weight for asset $i$.
- $w_i^{current}$ is the current weight for asset $i$.
- $V_{portfolio}$ is the total current market value of the portfolio.

The share quantity for asset $i$ is then calculated as $n_i = \lfloor \frac{\Delta_i}{\text{price}_i} \rfloor$, ensuring whole shares are traded.

Constraints are critical. Alex must ensure trades do not violate maximum individual position weights, total portfolio turnover limits, or minimum trade sizes. These checks are also embedded in specialized tools.

### Defining Sample Portfolio Data and Target Allocation

First, we simulate a current portfolio and define the target asset allocation policy.

```python
# Create a dummy portfolio holdings CSV file for demonstration
portfolio_data = {
    'ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'BRK-B', 'UNH', 'V'],
    'shares': [150, 200, 80, 100, 250, 180, 300, 120, 90, 130],
    'avg_cost': [145.00, 310.00, 140.00, 155.00, 160.00, 95.00, 350.00, 480.00, 250.00, 145.00]
}
portfolio_holdings_df = pd.DataFrame(portfolio_data)
portfolio_holdings_df.to_csv('portfolio_holdings.csv', index=False)
print("Generated 'portfolio_holdings.csv':")
print(portfolio_holdings_df.head())

# Define a target asset allocation policy (e.g., equal weight across all positions)
target_allocation_policy = {
    'AAPL': 0.10, 'MSFT': 0.10, 'AMZN': 0.10, 'GOOGL': 0.10, 'JPM': 0.10,
    'JNJ': 0.10, 'XOM': 0.10, 'BRK-B': 0.10, 'UNH': 0.10, 'V': 0.10
}
with open('target_allocation_policy.json', 'w') as f:
    json.dump(target_allocation_policy, f, indent=2)
print("\nGenerated 'target_allocation_policy.json':")
print(json.dumps(target_allocation_policy, indent=2))
```

### Implementing Core Rebalancing Tools

Here, we define the Python functions that serve as specialized "tools" for our AI agent. These tools perform the exact arithmetic and data operations.

```python
# Tool 1: Get Current Holdings
def get_holdings() -> str:
    """Reads current portfolio holdings from 'portfolio_holdings.csv'."""
    try:
        holdings = pd.read_csv('portfolio_holdings.csv')
        return holdings.to_json(orient='records')
    except FileNotFoundError:
        return json.dumps({"error": "portfolio_holdings.csv not found"})

# Tool 2: Get Current Prices
def get_current_prices(tickers: str) -> str:
    """Fetches current market prices for a comma-separated list of tickers using yfinance."""
    ticker_list = [t.strip() for t in tickers.split(',')]
    prices = {}
    for t in ticker_list:
        try:
            stock = yf.Ticker(t)
            # Use currentPrice or previousClose if currentPrice is not available
            prices[t] = round(stock.info.get('currentPrice', stock.info.get('previousClose', 0)), 2)
            if prices[t] == 0:
                print(f"Warning: Could not fetch price for {t}. Using 0.")
        except Exception as e:
            prices[t] = 0
            print(f"Error fetching price for {t}: {e}. Using 0.")
    return json.dumps(prices)

# Tool 3: Calculate Rebalancing Trades
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
            share_trade = int(dollar_trade / price) # Floor to whole shares
        
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

# Tool 4: Check Constraints
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
```

### Explanation of Tool Design
The separation of concerns is critical here. These tools perform the numerical heavy lifting, ensuring accuracy and consistency. The AI agent's role, as we'll see next, is to reason about *when* and *how* to use these tools, not to perform the calculations itself. This principle addresses the known limitations of LLMs with precise arithmetic.

---

## 3. Orchestrating Rebalancing with an AI Agent

Alex defines the AI agent's role through a detailed system prompt and provides it with the schemas of the specialized tools. The agent's task is to understand Alex's high-level goal (e.g., "Rebalance to equal-weight") and then break it down into steps, using the tools to gather data, calculate trades, and verify constraints.

**Critical Principle: LLMs Cannot Do Reliable Arithmetic.**
The agent's strength lies in its reasoning and ability to orchestrate tools. It *must* delegate all precise arithmetic operations to the Python tools. Asking an LLM to perform calculations like "150 shares × $178.50 per share" can lead to subtle but significant errors. For a 50-position portfolio with dozens of calculations, these errors compound, leading to materially incorrect trade tickets. The tools developed in the previous section guarantee numerical precision.

```python
REBALANCE_SYSTEM_PROMPT = """
You are a portfolio operations agent named "OptiWealth Rebalance Agent". Your primary goal is to compute rebalancing trades for an investment portfolio, ensuring all calculations are precise and constraints are met.

PROCESS:
1. Get current holdings using the 'get_holdings' tool.
2. Get current prices for all holdings using the 'get_current_prices' tool.
3. Determine the target allocation (e.g., from 'target_allocation_policy.json' or user input).
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

REBALANCE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_holdings",
            "description": "Reads current portfolio positions (ticker, shares, avg_cost) from a CSV file.",
            "parameters": {"type": "object", "properties": {}},
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
                    "min_trade_usd": {"type": "number", "description": "Minimum dollar amount for a trade (trades below this might be flagged or skipped)"},
                },
                "required": ["trades_summary_json"],
            },
        },
    },
]

# Mapping of tool names to actual functions
TOOLS = {
    "get_holdings": get_holdings,
    "get_current_prices": get_current_prices,
    "calculate_trades": calculate_trades,
    "check_constraints": check_constraints,
}

def run_rebalancing_agent(goal: str, max_iterations: int = 10) -> dict:
    """Executes the rebalancing agent workflow."""
    messages = [
        {"role": "system", "content": REBALANCE_SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    trace = []

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o for its enhanced tool-calling capabilities
            messages=messages,
            tools=REBALANCE_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0, # Zero temp for deterministic computation
        )
        msg = response.choices[0].message
        messages.append(msg)
        trace.append({"role": "assistant", "message": msg.model_dump_json()})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                function_name = tc.function.name
                function_args = json.loads(tc.function.arguments)

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
```

### Explanation of Agent Setup
The `run_rebalancing_agent` function acts as the central orchestrator. It communicates with the OpenAI API, passing Alex's goal and the available tools. The agent's "thinking process" is visible through the `trace`, which logs each tool call and its output. This transparency is crucial for auditability in financial workflows. The `temperature=0.0` ensures the agent behaves deterministically, which is paramount for high-stakes financial operations.

---

## 4. Executing the Agent and Generating Proposed Trades

Alex now runs the AI agent for a specific rebalancing goal. The agent will autonomously fetch data, compute trades, and identify initial compliance issues using the tools.

```python
# Alex's goal: Rebalance the portfolio to an equal-weight allocation across all positions.
rebalance_goal = "Rebalance the portfolio to equal-weight (10% each) across all positions, then check constraints."

print(f"--- Running OptiWealth Rebalance Agent for: {rebalance_goal} ---")
agent_result = run_rebalancing_agent(goal=rebalance_goal)

print(f"\n--- Agent Execution Completed in {agent_result['iterations']} iterations ---")
print("\n--- Raw Trade Ticket Proposal from Agent ---")
print(agent_result['trade_ticket'])

print("\n--- Agent Reasoning Trace (for auditability) ---")
for entry in agent_result['trace']:
    if entry['role'] == 'assistant':
        print(f"ASSISTANT: {entry['message']}")
    elif entry['role'] == 'tool':
        print(f"TOOL CALL: {entry['name']}({entry['args']}) -> {entry['result']}")
```

### Explanation of Agent Execution
The agent's trace clearly shows its thought process: it first called `get_holdings`, then `get_current_prices` for the identified tickers, followed by `calculate_trades` using the fetched data and the specified target allocation. Finally, it uses `check_constraints` to verify compliance. The raw trade ticket is the agent's proposed output, which Alex will scrutinize. The `trace` is a critical audit log, demonstrating how the agent arrived at its recommendation by showing its tool-call chain and intermediate results.

---

## 5. Verifying Compliance with Portfolio Constraints

After the agent proposes trades, Alex's next step is to get a structured `Constraint Check Report`. Even though the agent used the `check_constraints` tool, Alex wants a clear, unambiguous report detailing any violations. This is a non-negotiable step to ensure regulatory and internal policy compliance.

For this scenario, Alex has the following constraints:
-   **Maximum Individual Position Weight:** No single stock should exceed 12% of the portfolio value after rebalancing.
-   **Maximum Total Portfolio Turnover:** The total rebalancing turnover should not exceed 15% of the portfolio value.
-   **Minimum Trade Size:** Individual trades for any asset should be at least $1,000 to minimize transaction costs for small adjustments.

```python
# Extract the trades summary from the agent's output for explicit constraint checking
# This step ensures Alex can independently verify the constraints even outside the agent's flow
try:
    # Attempt to parse the trade ticket for the trades summary
    # The agent's final output should contain the JSON from calculate_trades tool call
    # We need to find the specific tool call where calculate_trades was used
    trades_summary_json_output = None
    for entry in reversed(agent_result['trace']): # Look at recent tool calls first
        if entry['role'] == 'tool' and entry['name'] == 'calculate_trades':
            trades_summary_json_output = entry['result']
            break
    
    if trades_summary_json_output:
        rebalancing_summary = json.loads(trades_summary_json_output)
        
        # Define constraint parameters
        max_position_weight_pct = 12.0
        max_turnover_pct = 15.0
        min_trade_usd = 1000.0

        print(f"\n--- OptiWealth Compliance Check Report ---")
        print(f"Checking against: Max Position={max_position_weight_pct}%, Max Turnover={max_turnover_pct}%, Min Trade=${min_trade_usd}")

        constraints_report_json = check_constraints(
            trades_summary_json=json.dumps(rebalancing_summary),
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

    else:
        print("Could not find trades summary in agent's trace to perform explicit constraint check.")
        rebalancing_summary = None

except json.JSONDecodeError as e:
    print(f"Error parsing agent's trade ticket or tool output: {e}")
    print("Please ensure the agent's final output or the calculate_trades tool output is valid JSON.")
    rebalancing_summary = None
except Exception as e:
    print(f"An unexpected error occurred during constraint checking: {e}")
    rebalancing_summary = None
```

### Explanation of Constraint Verification
This section explicitly calls the `check_constraints` tool, ensuring that Alex has a clear, programmatic verification of compliance. The output provides a simple pass/fail status for each constraint, along with details for any violations or warnings. This structured report is essential for Alex to quickly identify any issues and maintain trust in the automated workflow.

---

## 6. Human-in-the-Loop Approval and Final Trade Ticket Generation

The most critical step: human approval. Alex, as the CFA Charterholder, makes the final judgment call. The agent merely *proposes*; Alex *disposes*. This human-in-the-loop gate ensures that even if the quantitative analysis is perfect, strategic market insights, client-specific nuances, or unforeseen external factors are considered before any trade instructions are finalized. The trade ticket is marked "PENDING APPROVAL" to clearly indicate its provisional status.

```python
def human_approval_gate(trade_ticket_content: str) -> bool:
    """
    Simulates a human approval process for the trade ticket.
    In a production environment, this would be a UI widget or email notification.
    Includes safety checks before prompting for approval.
    """
    print("\n" + "=" * 60)
    print("TRADE TICKET - PENDING HUMAN APPROVAL")
    print("=" * 60)
    print(trade_ticket_content)
    print("\n" + "=" * 60)

    # Verify key safety checks from the trade ticket content
    checks = {
        'has_pending_status': 'PENDING' in trade_ticket_content.upper(),
        'has_buy_sell_keywords': any(kw in trade_ticket_content.upper() for kw in ('BUY', 'SELL')),
        'has_share_counts': any(char.isdigit() for char in trade_ticket_content) and 'shares' in trade_ticket_content.lower(),
        'no_executed_claim': 'EXECUTED' not in trade_ticket_content.upper(),
        'has_positive_dollar_amounts': all(float(d.split('$')[1].split(' ')[0]) >= 0 for d in trade_ticket_content.split('dollar_amount')[1:] if '$' in d) # Simple check assuming format
    }

    all_safe = all(checks.values())

    print("\n--- SAFETY CHECKS (Pre-Approval) ---")
    for check_name, passed in checks.items():
        print(f" {'PASS' if passed else 'FAIL'}: {check_name}")
    
    if not all_safe:
        print("\nBLOCKED: Safety checks failed. Cannot approve.")
        return False
    
    # In a real system, this would await human input
    # approval = input("\nApprove trades? (yes/no): ").lower()
    
    # Simulate approval for demo purposes
    approval = "yes"
    print(f"\nSimulating human input: '{approval}'")

    if approval == 'yes':
        print("\nSTATUS: APPROVED BY HUMAN")
        print("Trades would now be sent to execution system (if integrated).")
        return True
    else:
        print("\nSTATUS: REJECTED BY HUMAN")
        print("No trades executed.")
        return False

# Execute the human approval gate
approved = False
if rebalancing_summary: # Proceed only if trades_summary_json was successfully processed
    # Create a formatted trade ticket based on rebalancing_summary
    trades_df = pd.DataFrame(rebalancing_summary['trades'])
    trade_ticket_str = f"Portfolio Value: ${rebalancing_summary['total_portfolio_value']:.2f}\n"
    trade_ticket_str += f"Total Turnover: {rebalancing_summary['total_turnover_pct']:.2f}%\n"
    trade_ticket_str += "\nProposed Trades:\n"
    trade_ticket_str += trades_df[['ticker', 'action', 'shares', 'dollar_amount', 'current_weight', 'target_weight']].to_string(index=False)
    trade_ticket_str += "\n\nSTATUS: PENDING HUMAN APPROVAL"

    approved = human_approval_gate(trade_ticket_str)
else:
    print("\nCannot proceed to human approval as rebalancing summary was not generated successfully.")
```

### Explanation of Human Approval
The `human_approval_gate` function first displays the proposed `Trade Ticket` in a clear, easy-to-read format. It then runs crucial programmatic `SAFETY CHECKS` (e.g., ensuring "PENDING" status, presence of BUY/SELL actions, and no "EXECUTED" claims) before even prompting for human input. This layered defense is vital. For this lab, the human approval is simulated, but in production, this would be an interactive step. Alex's explicit "yes" confirms the trades are suitable, transforming information into action.

---

## 7. Visualizing Rebalancing Impact and Turnover

Alex needs to visually assess the rebalancing impact. A clear comparison of current, target, and proposed post-rebalance weights helps confirm alignment. Additionally, a turnover breakdown helps understand liquidity implications and potential transaction costs.

```python
if rebalancing_summary:
    # Get current holdings and prices again for precise current weights for visualization
    current_holdings_df = pd.read_csv('portfolio_holdings.csv')
    tickers = ','.join(current_holdings_df['ticker'].tolist())
    current_prices = json.loads(get_current_prices(tickers))

    current_values_dict = {row['ticker']: row['shares'] * current_prices.get(row['ticker'], 0)
                           for idx, row in current_holdings_df.iterrows()}
    total_current_value = sum(current_values_dict.values())
    current_weights_viz = {t: (v / total_current_value * 100) for t, v in current_values_dict.items()}

    target_weights_viz = {k: v * 100 for k, v in target_allocation_policy.items()}
    post_trade_weights_viz = rebalancing_summary['post_trade_weights']

    # Combine data for plotting
    all_tickers = sorted(list(set(current_weights_viz.keys()) | set(target_weights_viz.keys()) | set(post_trade_weights_viz.keys())))
    
    plot_data = []
    for ticker in all_tickers:
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Current', 'Weight': current_weights_viz.get(ticker, 0)})
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Target', 'Weight': target_weights_viz.get(ticker, 0)})
        plot_data.append({'Ticker': ticker, 'Weight Type': 'Post-Rebalance', 'Weight': post_trade_weights_viz.get(ticker, 0)})
    
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(14, 7))
    sns.barplot(x='Ticker', y='Weight', hue='Weight Type', data=plot_df, palette='viridis')
    plt.title('Pre/Post Rebalance Weight Comparison')
    plt.ylabel('Weight (%)')
    plt.xlabel('Asset Ticker')
    plt.legend(title='Weight Type')
    plt.ylim(0, max(plot_df['Weight'].max() * 1.1, 15)) # Ensure y-axis covers target 10% well
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Turnover Pie Chart
    buy_turnover = sum(t['dollar_amount'] for t in rebalancing_summary['trades'] if t['action'] == 'BUY')
    sell_turnover = sum(t['dollar_amount'] for t in rebalancing_summary['trades'] if t['action'] == 'SELL')
    net_cash_flow = buy_turnover - sell_turnover # Not exactly cash flow, but net change from buys/sells

    turnover_components = {
        'Buy Turnover': buy_turnover,
        'Sell Turnover': sell_turnover,
        # 'Net Cash Flow (Buy-Sell)': net_cash_flow # Can be positive or negative, not ideal for pie chart directly
    }

    # Filter out zero values for pie chart
    filtered_turnover_components = {k: v for k, v in turnover_components.items() if v > 0}

    if filtered_turnover_components:
        labels = filtered_turnover_components.keys()
        sizes = filtered_turnover_components.values()
        colors = sns.color_palette('pastel')[0:len(labels)]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f'${(p/100)*sum(sizes):,.2f}', startangle=90, pctdistance=0.85)
        plt.title('Distribution of Rebalancing Turnover (Dollar Amount)')
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo significant turnover for pie chart visualization.")
else:
    print("\nSkipping visualizations as rebalancing summary was not generated successfully.")
```

### Explanation of Visualizations
The "Pre/Post Rebalance Weight Comparison" bar chart is a vital visual check. Alex can quickly see if the proposed trades bring the portfolio weights in line with the target allocation. The "Turnover Distribution" pie chart helps in understanding the scale of trading activity and potential transaction costs. These visualizations serve as powerful communication tools for Alex to explain the rebalancing rationale and impact to clients or internal stakeholders.

---

## 8. Comparing Agent Performance to Alternatives

Alex and OptiWealth want to understand the true value proposition of the AI agent. This section compares the agent's approach to two alternatives: a manual process (simulated) and a pure Python script. This comparison highlights trade-offs in terms of speed, flexibility, accuracy, and auditability.

```python
# --- Approach A: Manual spreadsheet (simulated timing) ---
# A typical 10-position portfolio rebalance might take a human 25 minutes.
manual_time = 25 * 60 # seconds

# --- Approach B: Pure Python script (deterministic, no LLM) ---
def script_rebalance():
    """Pure Python rebalancing logic, without LLM orchestration."""
    start_time = time.time()

    holdings_json_data = get_holdings()
    holdings = json.loads(holdings_json_data)
    tickers_list = [h['ticker'] for h in holdings]
    tickers_str = ','.join(tickers_list)

    prices_json_data = get_current_prices(tickers_str)
    
    # Load target allocation from the predefined file
    with open('target_allocation_policy.json', 'r') as f:
        target_allocation = json.load(f)
    
    target_weights_json_data = json.dumps(target_allocation)

    trades_summary_json_data = calculate_trades(
        holdings_json=holdings_json_data,
        prices_json=prices_json_data,
        target_weights_json=target_weights_json_data
    )
    
    # Define constraint parameters for the script
    max_position_weight_pct = 12.0
    max_turnover_pct = 15.0
    min_trade_usd = 1000.0

    constraints_json_data = check_constraints(
        trades_summary_json=trades_summary_json_data,
        max_position_pct=max_position_weight_pct,
        max_turnover_pct=max_turnover_pct,
        min_trade_usd=min_trade_usd
    )

    elapsed_time = time.time() - start_time
    return trades_summary_json_data, constraints_json_data, elapsed_time

import time # Ensure time is imported for this section
script_trades, script_constraints, script_time = script_rebalance()

# --- Approach C: LLM Agent (from Step 4, we use its iteration count to estimate time) ---
# Assuming ~3 seconds per iteration for LLM calls + tool execution overhead
agent_estimated_time = agent_result['iterations'] * 3 # seconds

print("\n--- REBALANCING APPROACH COMPARISON ---")
print("=" * 60)
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
```

### Explanation of Comparison
This comparison provides a pragmatic view for Alex. For straightforward, fixed rebalancing rules (like equal-weight), a pure Python script is the fastest and most accurate. Its logic is hardcoded and executes without the overhead of LLM inference. The AI agent, while slower for simple tasks, excels in scenarios requiring *flexibility*, *natural language understanding*, and complex *contextual reasoning* that is difficult to hardcode into a script. Its audit trail also offers higher transparency than a manual process. This insight helps OptiWealth decide when to deploy each solution effectively.

---
