import threading
import pytest
from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_trading_agents_graph_thread_safety(monkeypatch):
    """
    Test that TradingAgentsGraph is thread-safe by instantiating it in multiple threads.
    This test monkeypatches `safe_create_memory` to avoid real DB/file operations and
    tracks the creation of memory objects. It asserts that each memory type is created
    exactly once per thread, ensuring no race conditions or shared state issues.
    """
    from tradingagents.graph import trading_graph

    created_names = []

    def fake_safe_create_memory(name):
        created_names.append(name)

        class DummyMemory:
            pass

        return DummyMemory()

    monkeypatch.setattr(trading_graph, "safe_create_memory", fake_safe_create_memory)

    # Function to instantiate TradingAgentsGraph
    def create_graph():
        graph = TradingAgentsGraph()
        assert graph.bull_memory is not None
        assert graph.bear_memory is not None
        assert graph.trader_memory is not None
        assert graph.invest_judge_memory is not None
        assert graph.risk_manager_memory is not None

    threads = [threading.Thread(target=create_graph) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Ensure all memory objects were created for each thread
    assert created_names.count("bull_memory") == 10
    assert created_names.count("bear_memory") == 10
    assert created_names.count("trader_memory") == 10
    assert created_names.count("invest_judge_memory") == 10
    assert created_names.count("risk_manager_memory") == 10
