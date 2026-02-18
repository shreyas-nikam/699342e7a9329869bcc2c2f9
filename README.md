This is a comprehensive `README.md` file for your Streamlit application lab project.

---

# QuLab: Lab 33: Portfolio Rebalancing Agent (Simulation)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

This Streamlit application, "QuLab: Lab 33: Portfolio Rebalancing Agent (Simulation)", is designed to simulate the workflow of a Senior Portfolio Operations Specialist, Alex Chen, at OptiWealth Asset Management. The application showcases how an AI agent can be leveraged to automate the quantitative aspects of portfolio rebalancing while maintaining crucial human oversight and compliance checks. It emphasizes a "human-in-the-loop" approach, where an AI agent orchestrates tool calls for complex financial calculations, and human experts provide critical approval after thorough verification.

## Table of Contents

1.  [Project Description](#project-description)
2.  [Features](#features)
3.  [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Environment Variables](#environment-variables)
4.  [Usage](#usage)
5.  [Mathematical Formulations & Core Logic](#mathematical-formulations--core-logic)
6.  [Project Structure](#project-structure)
7.  [Technology Stack](#technology-stack)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Contact](#contact)

---

## 1. Project Description

In the fast-paced world of asset management, efficiently rebalancing client portfolios according to predefined target allocations and strict compliance rules is paramount. This application provides a hands-on simulation of such a process. It allows Alex Chen, a CFA Charterholder, to:

*   **Define Portfolio & Policy**: Upload current holdings and specify target asset allocations.
*   **Orchestrate Rebalancing with AI**: Use a natural language prompt to guide an AI agent in orchestrating the rebalancing process, utilizing specialized tools for calculations.
*   **Verify Compliance**: Rigorously check proposed trades against various constraints (max position weight, portfolio turnover, min trade size).
*   **Human Approval**: Implement a critical human approval gate with built-in safety checks before any trades are finalized.
*   **Analyze Impact**: Visualize the pre/post-rebalance portfolio weights and understand the distribution of turnover.
*   **Compare Approaches**: Evaluate the trade-offs between manual, script-based, and AI-agent-orchestrated rebalancing.

The core principle demonstrated is the intelligent delegation of tasks: the AI agent excels at reasoning and orchestrating, while precise financial calculations and critical compliance checks are handled by robust, deterministic tools, with a human expert retaining final authority.

---

## 2. Features

The application is structured into several interactive pages accessible via the sidebar navigation:

*   **Welcome & Setup**:
    *   Upload current portfolio holdings from a CSV file (`ticker`, `shares`, `avg_cost`).
    *   Define target asset allocation as a JSON object (e.g., equal-weighting).
    *   Save loaded data to session state and local files for tool access.
    *   Introduction to the mathematical formulation of rebalancing trade calculation.

*   **Agent Rebalancing**:
    *   Define rebalancing goals in natural language.
    *   Configure maximum agent iterations.
    *   Run an AI agent to interpret the goal and call specialized tools (e.g., `calculate_trades`).
    *   View the agent's detailed reasoning trace, showing tool calls and outputs for auditability.
    *   Display the raw trade ticket proposed by the agent.

*   **Constraint Verification**:
    *   Define critical portfolio constraints: maximum individual position weight (%), maximum total portfolio turnover (%), and minimum trade size (USD).
    *   Run a dedicated `check_constraints` tool to verify the proposed trades against these rules.
    *   Receive a clear compliance status report, highlighting any violations or warnings.
    *   Detailed mathematical formulations for constraint verification.

*   **Human Approval Gate**:
    *   Present the final proposed `Trade Ticket` in a clear, formatted view.
    *   Execute programmatic "SAFETY CHECKS" to ensure the trade ticket meets basic integrity requirements.
    *   Require explicit human approval (Alex Chen) to "finalize" the trades, emphasizing the human-in-the-loop principle.

*   **Rebalancing Impact & Comparison**:
    *   **Pre/Post Rebalance Weight Comparison**: Visualize current, target, and post-rebalance asset weights using an interactive bar chart.
    *   **Distribution of Rebalancing Turnover**: Analyze the proportion of buy vs. sell turnover via a pie chart.
    *   **Rebalancing Approach Comparison**: A textual summary comparing Manual, Pure Python Script, and OptiWealth AI Agent approaches based on time, flexibility, accuracy, and auditability, providing key insights for optimal tool usage.

---

## 3. Getting Started

Follow these instructions to set up and run the Portfolio Rebalancing Agent application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   An OpenAI API Key (for the AI agent functionality)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/quolab-portfolio-rebalancing.git
    cd quolab-portfolio-rebalancing
    ```
    *(Note: Replace `your-username/quolab-portfolio-rebalancing` with the actual repository path if it's hosted online. For a local lab project, you might just `cd` into your project directory.)*

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**:
    Create a `requirements.txt` file in your project root with the following content:
    ```
    streamlit
    pandas
    matplotlib
    seaborn
    openai
    python-dotenv # Recommended for managing API keys
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

The application relies on the OpenAI API. You need to set your API key as an environment variable.

1.  **Create a `.env` file**: In the root directory of your project, create a file named `.env`.
2.  **Add your OpenAI API Key**: Add the following line to the `.env` file:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API Key.
    *(Note: The `source.py` file would typically load this. If `source.py` isn't using `python-dotenv`, you might need to set the environment variable directly in your shell before running Streamlit, e.g., `export OPENAI_API_KEY="sk-..."` on Linux/macOS or `$env:OPENAI_API_KEY="sk-..."` on PowerShell)*

---

## 4. Usage

Once the prerequisites are met and dependencies are installed, you can run the application:

1.  **Run the Streamlit Application**:
    Ensure your virtual environment is active.
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser (usually `http://localhost:8501`).

2.  **Navigate Through the Application**:
    Follow the steps as outlined by the sidebar navigation:

    *   **Welcome & Setup**:
        *   Upload `portfolio_holdings.csv` (a sample CSV with `ticker`, `shares`, `avg_cost` columns is recommended for demonstration).
        *   Adjust or confirm the target allocation policy in JSON format.
        *   Click "Load & Process Portfolio".

    *   **Agent Rebalancing**:
        *   Enter your rebalancing goal in natural language (e.g., "Rebalance the portfolio to equal-weight (10% each) across all positions, then check constraints.").
        *   Set the maximum agent iterations.
        *   Click "Run Rebalancing Agent" to see the agent's thought process and proposed trade ticket.

    *   **Constraint Verification**:
        *   Adjust the sliders to define `Maximum Individual Position Weight (%)`, `Maximum Total Portfolio Turnover (%)`, and `Minimum Trade Size (USD)`.
        *   Click "Check Constraints" to see if the proposed trades meet compliance.

    *   **Human Approval Gate**:
        *   Review the detailed trade ticket.
        *   Observe the programmatic safety checks.
        *   Click "Approve Trades" if everything is satisfactory.

    *   **Rebalancing Impact & Comparison**:
        *   View the visual comparisons of portfolio weights and turnover distribution.
        *   Read the analysis on different rebalancing approaches.

---

## 5. Mathematical Formulations & Core Logic

The application embeds the following key financial formulations and logical processes:

### Rebalancing Trade Calculation

The core rebalancing logic involves comparing current portfolio weights to target weights to determine the necessary dollar amount for each trade, then converting that into shares.

*   **Dollar amount to trade for asset $i$**:
    $$\Delta_i = (w_i^{\text{target}} - w_i^{\text{current}}) \times V_{\text{portfolio}}$$
    where $\Delta_i$ is the dollar amount to trade, $w_i^{\text{target}}$ is the target weight, $w_i^{\text{current}}$ is the current weight, and $V_{\text{portfolio}}$ is the total current market value of the portfolio.

*   **Share quantity for asset $i$**:
    $$n_i = \lfloor \frac{{\Delta_i}}{{\text{price}_i}} \rfloor$$
    where $n_i$ ensures whole shares are traded and $\text{price}_i$ is the current market price of asset $i$.

### Constraint Verification

The `check_constraints` tool verifies post-trade constraints:

*   **Maximum Individual Position Weight**:
    $$w_i^{\text{post}} < w_{\text{max}}$$
    where $w_i^{\text{post}}$ is the post-rebalance weight of asset $i$, and $w_{\text{max}}$ is the maximum allowed weight for any single position.

*   **Maximum Portfolio Turnover**:
    $$Turnover < Turnover_{\text{max}}$$
    where $Turnover$ is the total percentage turnover from rebalancing, and $Turnover_{\text{max}}$ is the maximum allowed portfolio turnover.

    The total turnover is calculated as:
    $$Turnover = \frac{\sum |\Delta_i|}{V_{\text{portfolio}}} \times 100\%$$
    where $\sum |\Delta_i|$ is the sum of absolute dollar amounts of all trades, and $V_{\text{portfolio}}$ is the total portfolio value.

*   **Minimum Trade Size**:
    $$|\Delta_i| \ge A_{\text{min}}$$
    where $|\Delta_i|$ is the absolute dollar amount of the trade for asset $i$, and $A_{\text{min}}$ is the minimum acceptable dollar amount for any trade.

### Agent Orchestration

The AI agent (powered by OpenAI) acts as a flexible planner. It receives a natural language goal and uses a set of predefined tools to achieve that goal. The agent's "thinking process" (`trace`) records each tool call and its output, providing full auditability, which is critical in financial operations. `temperature=0.0` ensures deterministic behavior.

---

## 6. Project Structure

```
.
├── app.py                     # Main Streamlit application file
├── source.py                  # Contains core functions (e.g., agent, tools, data retrieval)
├── .env                       # Environment variables (e.g., OPENAI_API_KEY)
├── requirements.txt           # Python dependencies
├── portfolio_holdings.csv     # Sample/placeholder for current portfolio holdings
└── target_allocation_policy.json # Sample/placeholder for target allocation
```

*   **`app.py`**: This is the entry point for the Streamlit application, handling the UI layout, session state management, and user interactions.
*   **`source.py`**: This file contains the backend logic, including the AI agent implementation (`run_rebalancing_agent`), financial calculation tools (`calculate_trades`, `check_constraints`), and data retrieval/utility functions (`get_holdings`, `get_current_prices`). This separation keeps the Streamlit app clean and focused on presentation.

---

## 7. Technology Stack

*   **Streamlit**: For creating interactive web applications with Python.
*   **Pandas**: For data manipulation and analysis of portfolio holdings.
*   **Matplotlib & Seaborn**: For generating insightful data visualizations (bar charts, pie charts).
*   **OpenAI API**: Powering the AI agent's natural language understanding and tool orchestration capabilities.
*   **Python**: The core programming language.

---

## 8. Contributing

As this is a lab project, contributions are generally not accepted in the form of pull requests to the main repository. However, you are encouraged to:

*   **Fork the repository** for your own experimentation and learning.
*   **Implement new features**: Extend the agent's capabilities, add more sophisticated constraints, or integrate with real-time market data APIs.
*   **Improve existing logic**: Refine the mathematical models or tool implementations.
*   **Enhance UI/UX**: Suggest or implement improvements to the Streamlit interface.

---

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you have one, otherwise state it's for educational purposes).

*(If you don't have a `LICENSE` file, you might replace the above with: "This project is intended for educational and demonstrative purposes as part of the QuLab curriculum. It is not licensed for commercial use without explicit permission.")*

---

## 10. Contact

For questions, feedback, or further information regarding this QuLab project, please refer to the QuantUniversity resources or contact:

*   **QuantUniversity**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **Support/Community**: (e.g., provide a link to a forum or email if applicable)

---

## License

## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
