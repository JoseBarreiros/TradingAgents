import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any
import threading
import copy
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from tradingagents.agents import Toolkit
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
import re
from tradingagents.dataflows.interface import set_config
from tradingagents.agents.utils.memory import FinancialSituationMemory

import chromadb.errors

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


def _sanitize_filename(name):
    # Replace any character that is not alphanumeric or underscore with underscore
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def safe_create_memory(name):
    """Thread-safe memory creation or reuse for ChromaDB."""
    from chromadb import PersistentClient

    client = PersistentClient()  # or however you're creating `chroma_client`
    try:
        collection = client.create_collection(name=name)
    except chromadb.errors.InternalError as e:
        if "already exists" in str(e):
            collection = client.get_collection(name=name)
        else:
            raise

    return FinancialSituationMemory(name, collection=collection)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self._owner_thread = threading.get_ident()
        self.debug = debug
        # Use a deep copy of DEFAULT_CONFIG if no config is provided
        self.config = copy.deepcopy(DEFAULT_CONFIG) if config is None else config

        # Update the interface's config
        set_config(self.config)

        # Create required directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"])
        self.quick_thinking_llm = ChatOpenAI(
            model=self.config["quick_think_llm"], temperature=0.1
        )
        self.toolkit = Toolkit(config=self.config)

        # Thread-safe memory initialization
        self.bull_memory = safe_create_memory("bull_memory")
        self.bear_memory = safe_create_memory("bear_memory")
        self.trader_memory = safe_create_memory("trader_memory")
        self.invest_judge_memory = safe_create_memory("invest_judge_memory")
        self.risk_manager_memory = safe_create_memory("risk_manager_memory")

        # Tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize subcomponents
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            self.config["risk_level"],
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}

        # Setup graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                "TradingAgentsGraph instance is not thread-safe. "
                "Create a separate instance per thread."
            )

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        return {
            "market": ToolNode(
                [
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    self.toolkit.get_stock_news_openai,
                    self.toolkit.get_reddit_stock_info,
                    self.toolkit.get_reddit_stock_info_online,
                ]
            ),
            "news": ToolNode(
                [
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_finnhub_news_online,
                    self.toolkit.get_reddit_news,
                    self.toolkit.get_reddit_news_online,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    self.toolkit.get_fundamentals_openai,
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""
        self._check_thread()
        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk["messages"]:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        safe_ticker = _sanitize_filename(self.ticker)
        directory = Path(f"eval_results/{safe_ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"/full_states_log_{self.curr_state['trade_date']}.json"
        try:
            with open(
                log_path,
                "w",
            ) as f:
                json.dump(self.log_states_dict, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Failed to write log to {log_path}: {e}")

    def reflect_and_remember(self, returns_losses):
        self._check_thread()
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
