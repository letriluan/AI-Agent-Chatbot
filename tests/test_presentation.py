from src.agent.presentation import chart_labels, coded_result_answer, remove_unsupported_currency


def test_codes_are_preserved_without_geographic_expansion():
    result = {"ok": True, "columns": ["total_amount"], "rows": [[70.0]]}
    answer = coded_result_answer("Total for WA customers?", "SELECT SUM(amount) FROM orders WHERE state='WA'", result)
    assert "WA" in answer and "70.0" in answer
    assert "Washington" not in answer and "$" not in answer
    assert coded_result_answer("How many SQL queries?", "SELECT COUNT(*) FROM orders", result) is None


def test_unknown_currency_is_removed_but_supplied_currency_is_preserved():
    result = {"columns": ["amount"], "rows": [[35.0]]}
    answer = remove_unsupported_currency("The amount is $35 USD.", "Average amount?", result)
    assert "$" not in answer and "USD" not in answer and "35" in answer
    assert remove_unsupported_currency("USD 35", "Show amount in USD", result) == "USD 35"
    assert remove_unsupported_currency("USD 35", "Average amount?", {"columns": ["amount_usd"], "rows": [[35]]}) == "USD 35"


def test_generic_plot_aliases_get_meaningful_labels():
    title, x, y = chart_labels(["x", "y"], [["delivered", 4]], "Plot number of orders by status")
    assert (title, x, y) == ("Order count by order status", "Order status", "Order count")
    title, x, y = chart_labels(["x", "y"], [["2026-01", 30]], "Plot monthly total amount")
    assert (title, x, y) == ("Total amount by month", "Month", "Total amount")


def test_column_aliases_are_displayed_without_inventing_units():
    title, x, y = chart_labels(["purchase_month", "total_item_price"], [["2026-01", 10]], "Monthly prices")
    assert (title, x, y) == ("Total item price by purchase month", "Purchase month", "Total item price")
    assert "$" not in title + x + y


def test_month_line_preserves_calendar_gaps():
    from unittest.mock import patch
    from src.agent import plot_tools
    with patch.object(plot_tools.plt, "plot", wraps=plot_tools.plt.plot) as draw:
        encoded = plot_tools.plot_line_png_base64(["2016-10", "2016-12"], [1, 2], title="Order count by month", xlabel="Month", ylabel="Order count")
        dates = draw.call_args.args[0]
        assert (dates[1] - dates[0]).days == 61
        assert encoded


def test_month_axis_does_not_repeat_month_labels():
    from datetime import datetime
    from src.agent import plot_tools
    fig, ax = plot_tools.plt.subplots()
    try:
        ax.plot([datetime(2026, 1, 1), datetime(2026, 3, 1)], [30, 110])
        plot_tools._finalize_plot("Amount by month", "Month", "Amount", temporal=True)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert len(labels) == len(set(labels))
        assert "2026-02" in labels
    finally:
        plot_tools.plt.close(fig)
