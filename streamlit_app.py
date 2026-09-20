import math
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Regression & Correlation Calculator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Regression & Correlation Calculator")
st.caption("Enter X and Y values to calculate means, correlation, regression equations and prediction.")

# ---------- Helpers ----------
def parse_values(text):
    text = text.replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]

    if not parts:
        raise ValueError("Please enter values.")

    try:
        return [float(p) for p in parts]
    except ValueError:
        raise ValueError("Use numbers only, separated by commas.")


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
with st.sidebar:
    st.header("1. Enter Data")
    st.write("Enter comma-separated values. X and Y must have the same number of values.")

    x_text = st.text_input(
        "X values",
        value="10, 20, 30, 40, 50"
    )

    y_text = st.text_input(
        "Y values",
        value="15, 25, 28, 38, 45"
    )

    calculate_clicked = st.button("Calculate", type="primary", use_container_width=True)

    st.divider()

    st.header("2. Prediction")
    prediction_x = st.number_input(
        "X for prediction",
        value=35.0,
        step=1.0
    )

# Calculate automatically on first load or when button is clicked.
if "results" not in st.session_state or calculate_clicked:
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

        # Correlation coefficient
        r = sxy / math.sqrt(sxx * syy)
        r2 = r * r

        # Regression of Y on X: Y = a + bX
        b_yx = sxy / sxx
        a_yx = y_mean - b_yx * x_mean

        # Regression of X on Y: X = a + bY
        b_xy = sxy / syy
        a_xy = x_mean - b_xy * y_mean

        predicted_y = a_yx + b_yx * prediction_x

        st.session_state.results = {
            "x": x,
            "y": y,
            "n": len(x),
            "x_mean": x_mean,
            "y_mean": y_mean,
            "sxx": sxx,
            "syy": syy,
            "sxy": sxy,
            "r": r,
            "r2": r2,
            "a_yx": a_yx,
            "b_yx": b_yx,
            "a_xy": a_xy,
            "b_xy": b_xy,
            "prediction_x": prediction_x,
            "predicted_y": predicted_y
        }

    except ValueError as e:
        st.error(str(e))

# ---------- Results ----------
if "results" in st.session_state:
    result = st.session_state.results

    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations (n)", result["n"])
    c2.metric("X̄ (Mean of X)", fmt(result["x_mean"]))
    c3.metric("Ȳ (Mean of Y)", fmt(result["y_mean"]))
    c4.metric("Correlation (r)", fmt(result["r"], 4))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("r²", fmt(result["r2"], 4))
    c6.metric("Σ(X − X̄)²", fmt(result["sxx"]))
    c7.metric("Σ(Y − Ȳ)²", fmt(result["syy"]))
    c8.metric("Σ(X − X̄)(Y − Ȳ)", fmt(result["sxy"]))

    st.subheader("Regression Equations")

    st.info(
        f"**Y on X:**  Y = {signed_number(result['a_yx'])} "
        f"{signed_term(result['b_yx'], 'X')}"
    )

    st.info(
        f"**X on Y:**  X = {signed_number(result['a_xy'])} "
        f"{signed_term(result['b_xy'], 'Y')}"
    )

    st.success(
        f"**Prediction:** For X = {fmt(result['prediction_x'])}, "
        f"predicted Y = {fmt(result['predicted_y'])}"
    )

    st.subheader("Scatter Plot + Regression Line")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(result["x"], result["y"], s=55, label="Data points")

    x_min, x_max = min(result["x"]), max(result["x"])
    if math.isclose(x_min, x_max):
        x_line = [x_min, x_max + 1]
    else:
        padding = (x_max - x_min) * 0.08
        x_line = [x_min - padding, x_max + padding]

    y_line = [
        result["a_yx"] + result["b_yx"] * value
        for value in x_line
    ]

    ax.plot(x_line, y_line, linewidth=2, label="Regression line (Y on X)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scatter Plot (r = {fmt(result['r'], 4)})")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Change the values and click Calculate to demonstrate how r, the regression equations, "
        "prediction and graph change."
    )
