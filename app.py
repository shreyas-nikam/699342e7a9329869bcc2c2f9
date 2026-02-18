import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from source import *

st.set_page_config(
    page_title="QuLab: Lab 33: Portfolio Rebalancing Agent (Simulation)", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 33: Portfolio Rebalancing Agent (Simulation)")
st.divider()

# Initialize Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Welcome & Setup"
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'portfolio_holdings_df' not in st.session_state:
    st.session_state.portfolio_holdings_df = pd.DataFrame(
        columns=['ticker', 'shares', 'avg_cost'])
if 'target_allocation' not in st.session_state:
    st.session_state.target_allocation = {}
if 'rebalance_goal' not in st.session_state:
    st.session_state.rebalance_goal = "Rebalance the portfolio to equal-weight (10% each) across all positions, then check constraints."
if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 12
if 'agent_result' not in st.session_state:
    st.session_state.agent_result = None
if 'rebalancing_summary' not in st.session_state:
    st.session_state.rebalancing_summary = None
if 'max_position_pct' not in st.session_state:
    st.session_state.max_position_pct = 12.0
if 'max_turnover_pct' not in st.session_state:
    st.session_state.max_turnover_pct = 15.0
if 'min_trade_usd' not in st.session_state:
    st.session_state.min_trade_usd = 1000.0
if 'constraints_report' not in st.session_state:
    st.session_state.constraints_report = None
if 'trade_ticket_str' not in st.session_state:
    st.session_state.trade_ticket_str = "No trade ticket generated yet."
if 'is_approved' not in st.session_state:
    st.session_state.is_approved = False
if 'script_rebalance_results' not in st.session_state:
    st.session_state.script_rebalance_results = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.selectbox(
    "Go to",
    ["Welcome & Setup", "Agent Rebalancing", "Constraint Verification",
        "Human Approval Gate", "Rebalancing Impact & Comparison"]
)
st.session_state.current_page = page_selection

# Page: Welcome & Setup
if st.session_state.current_page == "Welcome & Setup":
    st.header("Portfolio Rebalancing Agent: Welcome & Setup")
    st.markdown(f"")
    st.markdown(
        f"**Persona:** Alex Chen, Senior Portfolio Operations Specialist at OptiWealth Asset Management.")
    st.markdown(f"")
    st.markdown(f"As Alex, a CFA Charterholder, your task is to rebalance client portfolios to target asset allocations efficiently and accurately, ensuring compliance and human oversight. This application helps you leverage an AI agent and specialized tools to automate the quantitative aspects of rebalancing.")
    st.markdown(f"")

    st.subheader("0. Configure OpenAI API Key")
    openai_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        value=st.session_state.openai_api_key,
        help="Your API key is stored in session state and is not persisted. It will be cleared when you close the browser."
    )
    if openai_key_input:
        st.session_state.openai_api_key = openai_key_input
        st.success("✅ API key noted!")
    else:
        if not st.session_state.openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key to use the agent.")

    st.markdown(f"")
    st.subheader("1. Load Current Portfolio Holdings")
    if st.button("Load Sample Holdings"):
        sample_holdings = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'BRK-B', 'UNH', 'V'],
            'shares': [50, 40, 30, 20, 25, 35, 45, 15, 10, 60],
            'avg_cost': [150.0, 250.0, 3300.0, 2800.0, 160.0, 170.0, 60.0, 400.0, 450.0, 220.0]
        })
        st.session_state.portfolio_holdings_df = sample_holdings
        st.markdown(f"**Sample holdings loaded!**")
        st.dataframe(st.session_state.portfolio_holdings_df)

    st.subheader("2. Define Target Asset Allocation Policy")
    st.markdown(f"Enter your target asset allocation policy as a JSON object. This defines the desired percentage weight for each ticker.")
    st.markdown(f"")
    if st.session_state.target_allocation:
        target_alloc_df = pd.DataFrame(list(st.session_state.target_allocation.items()),
                                       columns=['Ticker', 'Target Weight'])
        target_alloc_df['Target Weight (%)'] = target_alloc_df['Target Weight'] * 100
    else:
        target_alloc_df = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'BRK-B', 'UNH', 'V'],
            'Target Weight': [0.10] * 10,
            'Target Weight (%)': [10.0] * 10
        })

    edited_target_alloc = st.data_editor(
        target_alloc_df, hide_index=True, key="target_alloc_editor")
    target_allocation_input = json.dumps({row['Ticker']: row['Target Weight']
                                          for idx, row in edited_target_alloc.iterrows()}, indent=2)

    try:
        st.session_state.target_allocation = json.loads(
            target_allocation_input)
    except json.JSONDecodeError:
        st.error("Invalid JSON for target allocation. Please check the format.")
        st.session_state.target_allocation = {}

    if st.button("Load & Process Portfolio"):
        if not st.session_state.portfolio_holdings_df.empty and st.session_state.target_allocation:
            st.session_state.portfolio_holdings_df.to_csv(
                'portfolio_holdings.csv', index=False)
            with open('target_allocation_policy.json', 'w') as f:
                json.dump(st.session_state.target_allocation, f, indent=2)
            st.success(
                "Portfolio holdings and target allocation loaded and saved successfully!")
            st.info("You can now proceed to the 'Agent Rebalancing' step.")
        else:
            st.warning(
                "Please upload holdings and define target allocation before proceeding.")

    st.markdown(f"")
    st.markdown(f"---")
    st.markdown(f"")
    st.markdown(r"**Mathematical Formulation: Rebalancing Trade Calculation**")
    st.markdown(f"The core rebalancing logic involves comparing current portfolio weights to target weights to determine the necessary dollar amount for each trade, then converting that into shares. This process is governed by the formula:")
    st.markdown(
        r"""
$$
\Delta_i = (w_i^{\text{target}} - w_i^{\text{current}}) \times V_{\text{portfolio}}
$$""")
    st.markdown(
        r"where $\Delta_i$ is the dollar amount to trade for asset $i$, $w_i^{\text{target}}$ is the target weight for asset $i$, $w_i^{\text{current}}$ is the current weight for asset $i$, and $V_{\text{portfolio}}$ is the total current market value of the portfolio.")
    st.markdown(f"")
    st.markdown(f"The share quantity for asset $i$ is then calculated as:")
    st.markdown(
        r"""
$$
n_i = \lfloor \frac{{\Delta_i}}{{\text{price}_i}} \rfloor
$$""")
    st.markdown(
        r"where $n_i$ ensures whole shares are traded and $\text{price}_i$ is the current market price of asset $i$.")
    st.markdown(f"")
    st.markdown(f"Constraints are critical. Alex must ensure trades do not violate maximum individual position weights, total portfolio turnover limits, or minimum trade sizes. These checks are also embedded in specialized tools.")

# Page: Agent Rebalancing
elif st.session_state.current_page == "Agent Rebalancing":
    st.header("Orchestrating Rebalancing with an AI Agent")
    st.markdown(f"")
    st.markdown(f"The AI agent's role is to reason about *when* and *how* to use specialized tools, not to perform the calculations itself. This principle addresses the known limitations of LLMs with precise arithmetic. Alex can define a rebalancing goal in natural language, and the agent will orchestrate the process.")

    st.subheader("1. Define Rebalancing Goal")
    rebalance_goal_input = st.text_input(
        "Enter your rebalancing goal:",
        value=st.session_state.rebalance_goal
    )
    st.session_state.rebalance_goal = rebalance_goal_input

    st.subheader("2. Configure Agent Parameters")
    max_iterations_slider = st.slider(
        "Maximum Agent Iterations",
        min_value=10, max_value=20, value=st.session_state.max_iterations, step=1
    )
    st.session_state.max_iterations = max_iterations_slider

    if st.button("Run Rebalancing Agent"):
        if not st.session_state.openai_api_key:
            st.error(
                "❌ Please configure your OpenAI API key in the 'Welcome & Setup' page first.")
        elif st.session_state.portfolio_holdings_df.empty or not st.session_state.target_allocation:
            st.warning(
                "Please complete 'Welcome & Setup' first by loading portfolio and target allocation.")
        else:
            with st.spinner("Agent is thinking and calling tools... This may take a moment."):
                # Convert holdings dataframe to JSON string
                holdings_json_str = st.session_state.portfolio_holdings_df.to_json(
                    orient='records')

                # Convert target allocation to JSON string for the agent
                target_allocation_json_str = json.dumps(
                    st.session_state.target_allocation)

                st.session_state.agent_result = run_rebalancing_agent(
                    openai_api_key=st.session_state.openai_api_key,
                    goal=st.session_state.rebalance_goal,
                    holdings_json_str=holdings_json_str,
                    target_allocation_policy_json_str=target_allocation_json_str,
                    max_iterations=st.session_state.max_iterations
                )

                # Debugging output
                print("Agent Result:", st.session_state.agent_result)

                # Check if we have a structured trade ticket
                if st.session_state.agent_result and st.session_state.agent_result.get('trade_ticket'):
                    trade_ticket = st.session_state.agent_result['trade_ticket']

                    # Convert TradeTicket to rebalancing_summary format for backward compatibility
                    if hasattr(trade_ticket, 'portfolio_value'):  # It's a Pydantic model
                        st.session_state.rebalancing_summary = {
                            'total_portfolio_value': trade_ticket.portfolio_value,
                            'total_turnover_pct': trade_ticket.total_turnover_pct,
                            'n_trades': trade_ticket.n_trades,
                            'trades': [trade.model_dump() for trade in trade_ticket.trades],
                            'post_trade_weights': {}  # Will be populated from trades if needed
                        }
                        st.success(
                            "Rebalancing agent completed its task!")
                    else:
                        st.warning(
                            "Agent did not return a structured trade ticket.")
                        st.session_state.rebalancing_summary = None
                else:
                    st.error(
                        "Agent failed to produce a valid trades summary. Please check the trace.")
                    st.session_state.rebalancing_summary = None

    if st.session_state.agent_result:
        st.subheader("3. Agent Reasoning Trace")
        st.markdown(
            f"The agent's thought process, showing tool calls and outputs for auditability.")
        with st.expander("View Agent Trace"):
            for entry in st.session_state.agent_result['trace']:
                if entry['role'] == 'assistant':
                    st.markdown(
                        f"**ASSISTANT:**")
                    if json.loads(entry['message'])['content']:
                        st.markdown(
                            f"{json.loads(entry['message'])['content']}")
                    else:
                        st.code(f"{entry['message']}", language='markdown')
                elif entry['role'] == 'tool':
                    st.markdown(
                        f"**TOOL CALL:**")
                    st.markdown("Tool called with input:")
                    st.code(f"{entry['name']}({entry['args']})",
                            language='json')
                    st.markdown(f"Tool output:")
                    st.code(entry['result'], language='json')

        st.subheader("4. Raw Trade Ticket Proposal from Agent")
        st.markdown(
            f"This is the agent's initial structured proposal, which Alex will scrutinize.")

        # Display the structured trade ticket
        if st.session_state.agent_result.get('trade_ticket_text'):
            st.code(
                st.session_state.agent_result['trade_ticket_text'], language='json')
        else:
            st.code(
                str(st.session_state.agent_result['trade_ticket']), language='markdown')

    st.markdown(f"")
    st.markdown(f"---")
    st.markdown(f"")
    st.markdown(r"**Explanation of Agent Setup**")
    st.markdown(f"The `run_rebalancing_agent` function acts as the central orchestrator. It communicates with the OpenAI API, passing Alex's goal and the available tools. The agent's \"thinking process\" is visible through the `trace`, which logs each tool call and its output. This transparency is crucial for auditability in financial workflows. The `temperature=0.0` ensures the agent behaves deterministically, which is paramount for high-stakes financial operations.")

# Page: Constraint Verification
elif st.session_state.current_page == "Constraint Verification":
    st.header("Verifying Compliance with Portfolio Constraints")
    st.markdown(f"")
    st.markdown(f"Alex's next step is to get a structured `Constraint Check Report`. This is a non-negotiable step to ensure regulatory and internal policy compliance.")

    st.subheader("1. Define Constraint Parameters")
    st.markdown(f"Adjust the sliders below to define the portfolio constraints:")

    max_position_slider = st.slider(
        "Maximum Individual Position Weight (%)",
        min_value=5.0, max_value=30.0, value=st.session_state.max_position_pct, step=0.5
    )
    st.session_state.max_position_pct = max_position_slider

    max_turnover_slider = st.slider(
        "Maximum Total Portfolio Turnover (%)",
        min_value=5.0, max_value=50.0, value=st.session_state.max_turnover_pct, step=0.5
    )
    st.session_state.max_turnover_pct = max_turnover_slider

    min_trade_slider = st.slider(
        "Minimum Trade Size (USD)",
        min_value=100.0, max_value=5000.0, value=st.session_state.min_trade_usd, step=100.0
    )
    st.session_state.min_trade_usd = min_trade_slider

    if st.button("Check Constraints"):
        if st.session_state.rebalancing_summary is None:
            st.warning(
                "Please run the 'Agent Rebalancing' step first to generate proposed trades.")
        else:
            with st.spinner("Checking constraints..."):
                constraints_report_json = check_constraints(
                    trades_summary_json=json.dumps(
                        st.session_state.rebalancing_summary),
                    max_position_pct=st.session_state.max_position_pct,
                    max_turnover_pct=st.session_state.max_turnover_pct,
                    min_trade_usd=st.session_state.min_trade_usd
                )
                st.session_state.constraints_report = json.loads(
                    constraints_report_json)
                st.success("Constraint check completed!")

    if st.session_state.constraints_report:
        st.subheader("2. Compliance Status")
        if st.session_state.constraints_report['all_constraints_met']:
            st.success("🟢 All key constraints are met!")
        else:
            st.error("🔴 **WARNING: Constraint VIOLATIONS detected!**")
            for violation in st.session_state.constraints_report['violations']:
                st.markdown(f"- **VIOLATION:** {violation}")

        if st.session_state.constraints_report['warnings']:
            st.warning("Warnings/Minor Issues:")
            for warning in st.session_state.constraints_report['warnings']:
                st.markdown(f"- **WARNING:** {warning}")

        with st.expander("Constraints Checked Summary"):
            for k, v in st.session_state.constraints_report['constraints_checked'].items():
                st.write(f"- {k}: {v}")

    st.markdown(f"")
    st.markdown(f"---")
    st.markdown(f"")
    st.markdown(r"**Mathematical Formulation: Constraint Verification**")
    st.markdown(
        f"The `check_constraints` tool verifies post-trade constraints. These include:")
    st.markdown(f"")
    st.markdown(r"**1. Maximum Individual Position Weight:**")
    st.markdown(r"""
$$
w_i^{\text{post}} < w_{\text{max}}
$$""")
    st.markdown(
        r"where $w_i^{\text{post}}$ is the post-rebalance weight of asset $i$, and $w_{\text{max}}$ is the maximum allowed weight for any single position.")
    st.markdown(f"")
    st.markdown(r"**2. Maximum Portfolio Turnover:**")
    st.markdown(r"""
$$
Turnover < Turnover_{\text{max}}
$$""")
    st.markdown(
        r"where $Turnover$ is the total percentage turnover from rebalancing, and $Turnover_{\text{max}}$ is the maximum allowed portfolio turnover.")
    st.markdown(f"")
    st.markdown(r"The total turnover is calculated as:")
    st.markdown(
        r"""
$$
Turnover = \frac{\sum |\Delta_i|}{V_{\text{portfolio}}} \times 100\%
$$""")
    st.markdown(
        r"where $\sum |\Delta_i|$ is the sum of absolute dollar amounts of all trades, and $V_{\text{portfolio}}$ is the total portfolio value.")
    st.markdown(f"")
    st.markdown(r"**3. Minimum Trade Size:**")
    st.markdown(r"""
$$
|\Delta_i| \ge A_{\text{min}}
$$""")
    st.markdown(
        r"where $|\Delta_i|$ is the absolute dollar amount of the trade for asset $i$, and $A_{\text{min}}$ is the minimum acceptable dollar amount for any trade (trades below this might be flagged or skipped).")

# Page: Human Approval Gate
elif st.session_state.current_page == "Human Approval Gate":
    st.header("Human-in-the-Loop Approval")
    st.markdown(f"")
    st.markdown(f"The `human_approval_gate` function first displays the proposed `Trade Ticket` in a clear, easy-to-read format. It then runs crucial programmatic `SAFETY CHECKS` before even prompting for human input. This layered defense is vital. Alex's explicit 'yes' confirms the trades are suitable, transforming information into action.")

    st.subheader("1. Review Proposed Trade Ticket")

    if st.session_state.rebalancing_summary:
        trades_df = pd.DataFrame(
            st.session_state.rebalancing_summary['trades'])
        st.session_state.trade_ticket_str = f"Portfolio Value: ${st.session_state.rebalancing_summary['total_portfolio_value']:.2f}\n"
        st.session_state.trade_ticket_str += f"Total Turnover: {st.session_state.rebalancing_summary['total_turnover_pct']:.2f}%\n"
        st.session_state.trade_ticket_str += "\nProposed Trades:\n"
        st.session_state.trade_ticket_str += trades_df[['ticker', 'action', 'shares',
                                                        'dollar_amount', 'current_weight', 'target_weight']].to_string(index=False)
        st.session_state.trade_ticket_str += "\n\nSTATUS: PENDING HUMAN APPROVAL"

        st.text_area("Trade Ticket Content",
                     value=st.session_state.trade_ticket_str, height=400, disabled=True)
    else:
        st.warning("Please run 'Agent Rebalancing' to generate a trade ticket.")

    st.subheader("2. Human Approval")
    st.markdown(
        f"Alex, critically review the proposed trades. If satisfactory, approve to proceed.")

    if st.button("Approve Trades", key="approve_button"):
        if st.session_state.rebalancing_summary:
            # Safety checks replication for UI
            trades_df_check = pd.DataFrame(
                st.session_state.rebalancing_summary['trades'])
            checks = {
                'has_pending_status': 'PENDING' in st.session_state.trade_ticket_str.upper(),
                'has_buy_sell_keywords': any(kw in st.session_state.trade_ticket_str.upper() for kw in ('BUY', 'SELL')),
                'has_share_counts': any(char.isdigit() for char in st.session_state.trade_ticket_str) and 'shares' in st.session_state.trade_ticket_str.lower(),
                'no_executed_claim': 'EXECUTED' not in st.session_state.trade_ticket_str.upper(),
            }
            try:
                dollar_amounts_present = True
                for trade_val in trades_df_check['dollar_amount']:
                    if trade_val < 0:
                        dollar_amounts_present = False
                        break
                checks['has_positive_dollar_amounts'] = dollar_amounts_present
            except Exception:
                checks['has_positive_dollar_amounts'] = False

            all_safe = all(checks.values())

            st.markdown(f"**--- SAFETY CHECKS (Pre-Approval) ---**")
            for check_name, passed in checks.items():
                st.markdown(f" - {'PASS' if passed else 'FAIL'}: {check_name}")

            if not all_safe:
                st.error("BLOCKED: Safety checks failed. Cannot approve.")
                st.session_state.is_approved = False
            else:
                st.session_state.is_approved = True
                st.success(
                    "STATUS: APPROVED BY HUMAN. Trades would now be sent to execution system.")
                st.info(
                    "The agent's work is approved. Proceed to 'Rebalancing Impact & Comparison' to see the full analysis.")
        else:
            st.warning(
                "No trade ticket to approve. Please run 'Agent Rebalancing' first.")

    if st.session_state.is_approved:
        st.success("Trade ticket has been approved.")
    else:
        st.info("Trade ticket is currently pending approval.")

# Page: Rebalancing Impact & Comparison
elif st.session_state.current_page == "Rebalancing Impact & Comparison":
    st.header("Rebalancing Impact & Performance Comparison")
    st.markdown(f"")
    st.markdown(f"The \"Pre/Post Rebalance Weight Comparison\" bar chart is a vital visual check. Alex can quickly see if the proposed trades bring the portfolio weights in line with the target allocation. The \"Turnover Distribution\" pie chart helps in understanding the scale of trading activity and potential transaction costs. These visualizations serve as powerful communication tools for Alex.")

    if st.session_state.rebalancing_summary:
        st.subheader("1. Pre/Post Rebalance Weight Comparison")

        # Use holdings from session state instead of reading from CSV
        current_holdings_df = st.session_state.portfolio_holdings_df
        tickers_list = current_holdings_df['ticker'].tolist()
        current_prices_json = get_current_prices(','.join(tickers_list))
        current_prices = json.loads(current_prices_json)

        current_values_dict = {row['ticker']: row['shares'] * current_prices.get(row['ticker'], 0)
                               for idx, row in current_holdings_df.iterrows()}
        total_current_value = sum(current_values_dict.values())
        current_weights_viz = {t: (v / total_current_value * 100) if total_current_value > 0 else 0
                               for t, v in current_values_dict.items()}

        target_weights_viz = {k: v * 100 for k,
                              v in st.session_state.target_allocation.items()}
        post_trade_weights_viz = st.session_state.rebalancing_summary['post_trade_weights']

        all_tickers = sorted(list(set(current_weights_viz.keys()) | set(
            target_weights_viz.keys()) | set(post_trade_weights_viz.keys())))

        plot_data = []
        for ticker in all_tickers:
            plot_data.append({'Ticker': ticker, 'Weight Type': 'Current',
                             'Weight': current_weights_viz.get(ticker, 0)})
            plot_data.append({'Ticker': ticker, 'Weight Type': 'Target',
                             'Weight': target_weights_viz.get(ticker, 0)})
            plot_data.append({'Ticker': ticker, 'Weight Type': 'Post-Rebalance',
                             'Weight': post_trade_weights_viz.get(ticker, 0)})

        plot_df = pd.DataFrame(plot_data)

        fig1, ax1 = plt.subplots(figsize=(14, 7))
        sns.barplot(x='Ticker', y='Weight', hue='Weight Type',
                    data=plot_df, palette='viridis', ax=ax1)
        ax1.set_title('Pre/Post Rebalance Weight Comparison')
        ax1.set_ylabel('Weight (%)')
        ax1.set_xlabel('Asset Ticker')
        ax1.legend(title='Weight Type')
        ax1.set_ylim(0, max(plot_df['Weight'].max() * 1.1, 15))
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)

        st.subheader("2. Distribution of Rebalancing Turnover")
        buy_turnover = sum(t['dollar_amount']
                           for t in st.session_state.rebalancing_summary['trades'] if t['action'] == 'BUY')
        sell_turnover = sum(t['dollar_amount']
                            for t in st.session_state.rebalancing_summary['trades'] if t['action'] == 'SELL')

        turnover_components = {
            'Buy Turnover': buy_turnover,
            'Sell Turnover': sell_turnover,
        }

        filtered_turnover_components = {
            k: v for k, v in turnover_components.items() if v > 0}

        if filtered_turnover_components:
            labels = filtered_turnover_components.keys()
            sizes = filtered_turnover_components.values()
            colors = sns.color_palette('pastel')[0:len(labels)]

            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.pie(sizes, labels=labels, colors=colors,
                    autopct=lambda p: f'${(p/100)*sum(sizes):,.2f}', startangle=90, pctdistance=0.85)
            ax2.set_title(
                'Distribution of Rebalancing Turnover (Dollar Amount)')
            ax2.axis('equal')
            st.pyplot(fig2)
        else:
            st.info("No significant turnover for pie chart visualization.")
    else:
        st.warning(
            "Visualizations require a successful rebalancing summary. Please complete previous steps.")

    st.subheader("3. Rebalancing Approach Comparison")
    st.markdown(f"This comparison provides a pragmatic view for Alex. For straightforward, fixed rebalancing rules, a pure Python script is often the fastest and most accurate. The AI agent excels in scenarios requiring *flexibility*, *natural language understanding*, and complex *contextual reasoning*.")

    if st.session_state.script_rebalance_results is None:
        # Ensure we have a fallback if scripted_rebalancing is not directly importable from source.py
        if 'scripted_rebalancing' not in globals():
            def script_rebalance_app_helper():
                start_time = time.time()

                # Convert holdings dataframe to JSON string
                holdings_json_str = st.session_state.portfolio_holdings_df.to_json(
                    orient='records')
                holdings_json_data = get_holdings(
                    holdings_data=holdings_json_str)
                holdings = json.loads(holdings_json_data)
                tickers_list = [h['ticker'] for h in holdings]
                tickers_str = ','.join(tickers_list)

                prices_json_data = get_current_prices(tickers_str)

                target_weights_json_data = json.dumps(
                    st.session_state.target_allocation)

                trades_summary_json_data = calculate_trades(
                    holdings_json=holdings_json_data,
                    prices_json=prices_json_data,
                    target_weights_json=target_weights_json_data
                )

                max_position_weight_pct = st.session_state.max_position_pct
                max_turnover_pct = st.session_state.max_turnover_pct
                min_trade_usd = st.session_state.min_trade_usd

                constraints_json_data = check_constraints(
                    trades_summary_json=trades_summary_json_data,
                    max_position_pct=max_position_weight_pct,
                    max_turnover_pct=max_turnover_pct,
                    min_trade_usd=min_trade_usd
                )
                elapsed_time = time.time() - start_time
                return trades_summary_json_data, constraints_json_data, elapsed_time

            script_trades, script_constraints, script_time = script_rebalance_app_helper()
            st.session_state.script_rebalance_results = (
                script_trades, script_constraints, script_time)
        else:
            # Use the scripted_rebalancing function from source.py
            holdings_json_str = st.session_state.portfolio_holdings_df.to_json(
                orient='records')
            script_trades, script_constraints, script_time = scripted_rebalancing(
                holdings_json_str=holdings_json_str,
                target_allocation_file_path='target_allocation_policy.json',
                constraint_params={
                    'max_position_pct': st.session_state.max_position_pct,
                    'max_turnover_pct': st.session_state.max_turnover_pct,
                    'min_trade_usd': st.session_state.min_trade_usd
                }
            )
            st.session_state.script_rebalance_results = (
                script_trades, script_constraints, script_time)

    manual_time = 25 * 60
    agent_estimated_time = st.session_state.agent_result['iterations'] * \
        3 if st.session_state.agent_result else 0

    comparison_data = {
        'Approach': ['Manual (spreadsheet)', 'Pure Python Script', 'OptiWealth AI Agent'],
        'Time': [f'{manual_time/60:.0f} min', f'{st.session_state.script_rebalance_results[2]:.2f}s', f'~{agent_estimated_time:.0f}s'],
        'Flexibility': ['Low', 'Low', 'High'],
        'Accuracy': ['Medium', 'Perfect', 'High*'],
        'Auditability': ['Low', 'High', 'High']
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    st.markdown(
        f"\n* Agent accuracy depends on tool delegation (arithmetic by tools). Without tools, accuracy would be Low.")
    st.markdown(f"\n**KEY INSIGHT:**")
    st.markdown(
        f"- For FIXED rebalancing rules (e.g., equal-weight to target), a pure Python script is FASTER and more ACCURATE.")
    st.markdown(f"- The AI agent adds value when:")
    st.markdown(f"  - The rebalancing rule is described in NATURAL LANGUAGE (e.g., 'underweight energy, overweight tech, keep cash at 5%').")
    st.markdown(
        f"  - CONSTRAINTS CHANGE dynamically (e.g., 'also minimize tax impact this quarter').")
    st.markdown(
        f"  - HUMAN COMMUNICATION is needed (e.g., explaining the trades to a client).")
    st.markdown(f"\n**Conclusion for OptiWealth:**")
    st.markdown(
        f"  - Use dedicated scripts for standardized, repeatable rebalancing with fixed rules.")
    st.markdown(f"  - Leverage the AI agent for flexible, context-dependent rebalancing that requires natural language interpretation, dynamic constraint handling, and client-ready explanations.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
