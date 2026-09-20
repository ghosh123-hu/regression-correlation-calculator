import math
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Regression & Correlation Calculator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Regression & Correlation Calculator")
st.caption("Enter X and Y values, then press Calculate to see means, correlation, regression equations and prediction.")


# ---------- Helpers ----------
def parse_values(text):
    text = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]

    if not parts:
        raise ValueError("Please enter values.")

    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError("Use numbers only, separated by commas.")

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Values must be finite numbers (no NaN or infinity).")

    return values


def fmt(value, decimals=3):
    if abs(value) < 1e-10:
        value = 0
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def signed_number(value):
    return f"{fmt(value)}" if value >= 0 else f"− {fmt(abs(value))}"


def signed_term(value, variable):
    sign = "+" if value >= 0 else "−"
    return f"{sign} {fmt(abs(value))}{variable}"


# ---------- Input ----------
# No pre-filled values and no remembered values:
#  - fields start empty (only grey placeholder examples are shown)
#  - autocomplete="off" stops the browser from suggesting/remembering old entries
#  - the Clear button wipes whatever is typed, and the results with it
def clear_inputs():
    st.session_state["x_text"] = ""
    st.session_state["y_text"] = ""
    st.session_state["prediction_x"] = None


with st.sidebar:
    with st.form("input_form"):
        st.header("1. Enter Data")
        st.write("Enter comma-separated values. X and Y must have the same number of values.")

        x_text = st.text_input(
            "X values", key="x_text", placeholder="e.g. 10, 20, 30, 40, 50", autocomplete="off"
        )
        y_text = st.text_input(
            "Y values", key="y_text", placeholder="e.g. 15, 25, 28, 38, 45", autocomplete="off"
        )

        st.divider()

        st.header("2. Prediction (optional)")
        prediction_x = st.number_input(
            "X for prediction", key="prediction_x", value=None, step=1.0, placeholder="e.g. 35"
        )

        calculate = st.form_submit_button("Calculate", type="primary", use_container_width=True)

    st.button("Clear all inputs", on_click=clear_inputs, use_container_width=True)


# ---------- Calculation ----------
# No caching and no session_state: results exist only for the run in which
# Calculate was pressed, so old results can never be shown.
results = None
error = None

if calculate:
    try:
        x = parse_values(x_text)
        y = parse_values(y_text)

        if len(x) != len(y):
            raise ValueError("X and Y must contain the same number of values.")
        if len(x) < 2:
            raise ValueError("Please enter at least two paired observations.")

        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        dx = [value - x_mean for value in x]
        dy = [value - y_mean for value in y]

        sxx = sum(d * d for d in dx)
        syy = sum(d * d for d in dy)
        sxy = sum(a * b for a, b in zip(dx, dy))

        if sxx == 0:
            raise ValueError("X values must not all be the same.")
        if syy == 0:
            raise ValueError("Y values must not all be the same.")

        r = sxy / math.sqrt(sxx * syy)
        r = max(-1.0, min(1.0, r))  # guard against float rounding, e.g. r = 1.0000000000000002
        r2 = r * r

        # Regression of Y on X: Y = a + bX
        b_yx = sxy / sxx
        a_yx = y_mean - b_yx * x_mean

        # Regression of X on Y: X = a + bY
        b_xy = sxy / syy
        a_xy = x_mean - b_xy * y_mean

        predicted_y = None if prediction_x is None else a_yx + b_yx * prediction_x

        results = {
            "x": x, "y": y, "n": len(x),
            "x_mean": x_mean, "y_mean": y_mean,
            "sxx": sxx, "syy": syy, "sxy": sxy,
            "r": r, "r2": r2,
            "a_yx": a_yx, "b_yx": b_yx,
            "a_xy": a_xy, "b_xy": b_xy,
            "prediction_x": prediction_x,
            "predicted_y": predicted_y
        }

    except ValueError as e:
        error = str(e)

if error:
    st.error(error)

if not calculate:
    st.info("👈 Enter your X and Y values in the sidebar and press **Calculate** to see the results.")


# ---------- Results ----------
if results:
    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (n)", results["n"])
    c2.metric("X̄ (Mean of X)", fmt(results["x_mean"]))
    c3.metric("Ȳ (Mean of Y)", fmt(results["y_mean"]))
    c4.metric("Correlation (r)", fmt(results["r"], 4))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("r²", fmt(results["r2"], 4))
    c6.metric("Σ(X − X̄)²", fmt(results["sxx"]))
    c7.metric("Σ(Y − Ȳ)²", fmt(results["syy"]))
    c8.metric("Σ(X − X̄)(Y − Ȳ)", fmt(results["sxy"]))

    st.subheader("Regression Equations")

    st.info(
        f"**Y on X:**  Y = {signed_number(results['a_yx'])} "
        f"{signed_term(results['b_yx'], 'X')}"
    )

    st.info(
        f"**X on Y:**  X = {signed_number(results['a_xy'])} "
        f"{signed_term(results['b_xy'], 'Y')}"
    )

    if results["predicted_y"] is not None:
        st.success(
            f"**Prediction:** For X = {fmt(results['prediction_x'])}, "
            f"predicted Y = {fmt(results['predicted_y'])}"
        )

    st.subheader("Scatter Plot + Regression Line")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(results["x"], results["y"], s=55, label="Data points")

    x_min, x_max = min(results["x"]), max(results["x"])
    if math.isclose(x_min, x_max):
        x_line = [x_min, x_max + 1]
    else:
        padding = (x_max - x_min) * 0.08
        x_line = [x_min - padding, x_max + padding]

    y_line = [results["a_yx"] + results["b_yx"] * v for v in x_line]

    ax.plot(x_line, y_line, linewidth=2, label="Regression line (Y on X)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scatter Plot (r = {fmt(results['r'], 4)})")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
