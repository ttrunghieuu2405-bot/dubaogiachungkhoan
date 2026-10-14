
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# 1. CẤU HÌNH
# =====================================================

st.set_page_config(
    page_title="AI Stock Price Forecast",
    page_icon="📈",
    layout="wide"
)

SEED = 42
WINDOW_SIZE = 60

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #1677c8;
    text-align: center;
}
.sub-title {
    text-align: center;
    color: #777;
    font-size: 17px;
}
[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.07);
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📈 AI STOCK PRICE FORECAST</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">'
    'Ứng dụng dự báo giá cổ phiếu bằng trí tuệ nhân tạo LSTM'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# 2. SIDEBAR
# =====================================================

st.sidebar.header("⚙️ CẤU HÌNH DỰ BÁO")

stock_options = {
    "Vinamilk (VNM)": "VNM.VN",
    "FPT Corporation (FPT)": "FPT.VN",
    "Hòa Phát (HPG)": "HPG.VN",
    "Vietcombank (VCB)": "VCB.VN",
    "Vingroup (VIC)": "VIC.VN",
    "Thế Giới Di Động (MWG)": "MWG.VN",
    "Techcombank (TCB)": "TCB.VN",
    "SSI Securities (SSI)": "SSI.VN",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Tự nhập mã": "CUSTOM"
}

stock_name = st.sidebar.selectbox(
    "Chọn cổ phiếu",
    list(stock_options.keys())
)

ticker = stock_options[stock_name]

if ticker == "CUSTOM":
    ticker = st.sidebar.text_input(
        "Nhập mã cổ phiếu",
        value="VNM.VN"
    ).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=date.today(),
    max_value=date.today()
)

forecast_days = st.sidebar.slider(
    "Số phiên muốn dự báo",
    min_value=1,
    max_value=30,
    value=7
)

epochs = st.sidebar.slider(
    "Số Epochs",
    min_value=5,
    max_value=100,
    value=10,
    step=5
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64],
    index=1
)

run_button = st.sidebar.button(
    "🚀 BẮT ĐẦU DỰ BÁO",
    type="primary",
    use_container_width=True
)

st.sidebar.info(
    "Mô hình sử dụng 60 phiên giao dịch gần nhất "
    "để dự báo giá đóng cửa phiên tiếp theo."
)

# =====================================================
# 3. HÀM TẢI DỮ LIỆU
# =====================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start, end):
    # yfinance sử dụng end độc quyền
    end_inclusive = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=str(start),
        end=end_inclusive,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xử lý MultiIndex của yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    return df


# =====================================================
# 4. TẠO SEQUENCES
# =====================================================

def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =====================================================
# 5. XÂY DỰNG MÔ HÌNH LSTM
# =====================================================

def build_lstm_model(window_size=60):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# =====================================================
# 6. HUẤN LUYỆN VÀ KIỂM THỬ
# =====================================================

def train_and_evaluate(df, epochs, batch_size):

    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = df[["Close"]].values.astype("float32")

    train_size = int(0.8 * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(
        train_scaled, WINDOW_SIZE
    )

    test_combined = np.vstack((
        train_scaled[-WINDOW_SIZE:],
        test_scaled
    ))

    X_test, y_test = create_sequences(
        test_combined, WINDOW_SIZE
    )

    X_train = X_train.reshape(
        X_train.shape[0], WINDOW_SIZE, 1
    )

    X_test = X_test.reshape(
        X_test.shape[0], WINDOW_SIZE, 1
    )

    # Validation lấy 10% cuối của train
    val_size = max(1, int(0.1 * len(X_train)))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    model = build_lstm_model(WINDOW_SIZE)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    # Dự báo trên tập kiểm thử
    test_pred_scaled = model.predict(
        X_test, verbose=0
    )

    test_pred = scaler.inverse_transform(
        test_pred_scaled
    )

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_real, test_pred)
    )

    mae = mean_absolute_error(
        y_test_real, test_pred
    )

    r2 = r2_score(
        y_test_real, test_pred
    )

    # Chỉ số phần trăm sai số
    actual = y_test_real.flatten()
    predicted = test_pred.flatten()

    nonzero = np.abs(actual) > 1e-8

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

    test_dates = df.index[train_size:]

    comparison_df = pd.DataFrame({
        "Ngày": test_dates,
        "Giá thực tế": actual,
        "Giá dự báo": predicted
    })

    comparison_df = comparison_df.set_index("Ngày")

    all_scaled = scaler.transform(data)

    last_60_days = all_scaled[-WINDOW_SIZE:]

    return (
        model,
        scaler,
        history,
        metrics,
        comparison_df,
        last_60_days
    )


# =====================================================
# 7. DỰ BÁO NHIỀU PHIÊN TƯƠNG LAI
# =====================================================

def forecast_future(
    model,
    scaler,
    last_60_days,
    forecast_days
):

    current_window = last_60_days.copy()
    predictions_scaled = []

    for _ in range(forecast_days):

        X_input = current_window.reshape(
            1, WINDOW_SIZE, 1
        )

        next_prediction = model.predict(
            X_input, verbose=0
        )[0, 0]

        predictions_scaled.append(next_prediction)

        current_window = np.vstack((
            current_window[1:],
            [[next_prediction]]
        ))

    predictions = scaler.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    )

    return predictions.flatten()


# =====================================================
# 8. VẼ BIỂU ĐỒ
# =====================================================

def plot_price_history(df, ticker):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df.index,
        df["Close"],
        label="Giá đóng cửa",
        color="#1677c8",
        linewidth=1.7
    )

    ax.set_title(
        f"Lịch sử giá cổ phiếu {ticker}",
        fontsize=14
    )

    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_training_loss(history):

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        history.history["loss"],
        label="Train Loss",
        linewidth=2
    )

    ax.plot(
        history.history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    ax.set_title("Train Loss vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def plot_test_predictions(comparison_df):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        comparison_df.index,
        comparison_df["Giá thực tế"],
        label="Giá thực tế",
        color="#1677c8"
    )

    ax.plot(
        comparison_df.index,
        comparison_df["Giá dự báo"],
        label="Giá dự báo LSTM",
        color="#e67e22"
    )

    ax.set_title("So sánh giá thực tế và dự báo")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def plot_future_forecast(
    df,
    forecast_df,
    ticker
):

    fig, ax = plt.subplots(figsize=(12, 5))

    historical = df["Close"].tail(90)

    ax.plot(
        historical.index,
        historical.values,
        label="Giá lịch sử",
        color="#1677c8",
        linewidth=2
    )

    # Nối đường dự báo với giá cuối cùng
    future_dates = [
        historical.index[-1]
    ] + list(forecast_df.index)

    future_values = [
        float(historical.iloc[-1])
    ] + forecast_df["Giá dự báo"].tolist()

    ax.plot(
        future_dates,
        future_values,
        label="Dự báo tương lai",
        color="#e74c3c",
        marker="o",
        markersize=4,
        linestyle="--",
        linewidth=2
    )

    ax.axvline(
        x=historical.index[-1],
        color="gray",
        linestyle=":",
        label="Mốc bắt đầu dự báo"
    )

    ax.set_title(
        f"Dự báo giá tương lai - {ticker}"
    )

    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


# =====================================================
# 9. GIAO DIỆN CHÍNH
# =====================================================

st.subheader("📊 Tổng quan ứng dụng")

st.write(
    "Ứng dụng sử dụng mạng nơ-ron LSTM để học "
    "xu hướng biến động giá đóng cửa trong quá khứ "
    "và ước lượng giá cho các phiên tiếp theo."
)

st.info(
    "Chọn mã cổ phiếu, thời gian dữ liệu và số phiên "
    "dự báo ở thanh bên trái. Sau đó nhấn "
    "'BẮT ĐẦU DỰ BÁO'."
)

if run_button:

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu hợp lệ.")
        st.stop()

    if start_date >= end_date:
        st.error(
            "Ngày bắt đầu phải nhỏ hơn ngày kết thúc."
        )
        st.stop()

    with st.spinner("Đang tải dữ liệu cổ phiếu..."):
        try:
            df = load_data(
                ticker,
                start_date,
                end_date
            )
        except Exception as e:
            st.error(f"Lỗi tải dữ liệu: {e}")
            st.stop()

    if df.empty:
        st.error(
            "Không tìm thấy dữ liệu. "
            "Hãy kiểm tra mã cổ phiếu hoặc khoảng ngày."
        )
        st.stop()

    if len(df) < 150:
        st.error(
            "Dữ liệu quá ít để huấn luyện. "
            "Vui lòng chọn khoảng thời gian dài hơn "
            "(ít nhất 150 phiên giao dịch)."
        )
        st.stop()

    # ---------------------------------------------
    # TỔNG QUAN DỮ LIỆU
    # ---------------------------------------------

    st.success(
        f"Đã tải thành công {len(df)} phiên "
        f"giao dịch của {ticker}."
    )

    last_price = float(df["Close"].iloc[-1])
    first_price = float(df["Close"].iloc[0])

    price_change = (
        (last_price - first_price) / first_price * 100
    ) if first_price != 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mã cổ phiếu",
            ticker
        )

    with col2:
        st.metric(
            "Giá đóng cửa gần nhất",
            f"{last_price:,.2f}"
        )

    with col3:
        st.metric(
            "Số phiên dữ liệu",
            f"{len(df):,}"
        )

    with col4:
        st.metric(
            "Biến động toàn kỳ",
            f"{price_change:+.2f}%"
        )

    st.caption(
        "Đơn vị giá theo dữ liệu Yahoo Finance. "
        "Dữ liệu được tải với auto_adjust=True, "
        "nên giá đóng cửa đã được điều chỉnh."
    )

    st.subheader("1. Dữ liệu giá lịch sử")

    st.pyplot(
        plot_price_history(df, ticker)
    )

    with st.expander("Xem dữ liệu cổ phiếu"):
        st.dataframe(
            df.tail(100),
            use_container_width=True
        )

    # ---------------------------------------------
    # HUẤN LUYỆN
    # ---------------------------------------------

    st.subheader("2. Huấn luyện mô hình LSTM")

    with st.spinner(
        "AI đang huấn luyện mô hình. "
        "Quá trình có thể mất vài phút..."
    ):
        try:
            (
                model,
                scaler,
                history,
                metrics,
                comparison_df,
                last_60_days
            ) = train_and_evaluate(
                df,
                epochs,
                batch_size
            )
        except Exception as e:
            st.error(f"Lỗi huấn luyện: {e}")
            st.stop()

    st.success("Huấn luyện mô hình hoàn tất!")

    st.write(
        f"Số Epoch thực tế: "
        f"**{len(history.history['loss'])}**"
    )

    st.pyplot(
        plot_training_loss(history)
    )

    # ---------------------------------------------
    # ĐÁNH GIÁ
    # ---------------------------------------------

    st.subheader("3. Đánh giá độ chính xác")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "RMSE",
            f"{metrics['RMSE']:,.2f}"
        )

    with m2:
        st.metric(
            "MAE",
            f"{metrics['MAE']:,.2f}"
        )

    with m3:
        st.metric(
            "R²",
            f"{metrics['R2']:.4f}"
        )

    with m4:
        st.metric(
            "MAPE",
            f"{metrics['MAPE']:.2f}%"
        )

    st.markdown("""
    **Ý nghĩa các chỉ số:**

    - **RMSE:** Sai số bình phương trung bình căn bậc hai.
      Càng thấp càng tốt.
    - **MAE:** Sai số tuyệt đối trung bình.
      Càng thấp càng tốt.
    - **R²:** Mức độ phù hợp trên tập kiểm thử.
      Càng gần 1 càng tốt.
    - **MAPE:** Sai số phần trăm tuyệt đối trung bình.
      Càng thấp càng tốt.
    """)

    st.pyplot(
        plot_test_predictions(comparison_df)
    )

    with st.expander(
        "Xem bảng so sánh thực tế và dự báo"
    ):
        st.dataframe(
            comparison_df,
            use_container_width=True
        )

    # ---------------------------------------------
    # DỰ BÁO TƯƠNG LAI
    # ---------------------------------------------

    st.subheader("4. Dự báo giá cổ phiếu tương lai")

    future_predictions = forecast_future(
        model,
        scaler,
        last_60_days,
        forecast_days
    )

    # Dùng ngày làm việc T2-T6 để ước lượng
    # Các phiên tương lai; chưa loại trừ ngày lễ
    last_date = pd.Timestamp(df.index[-1])

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Giá dự báo": future_predictions
    }, index=future_dates)

    forecast_df.index.name = "Ngày dự kiến"

    forecast_df["Thay đổi so với giá cuối (%)"] = (
        (forecast_df["Giá dự báo"] / last_price) - 1
    ) * 100

    next_price = float(future_predictions[0])
    final_price = float(future_predictions[-1])

    next_change = (
        (next_price / last_price) - 1
    ) * 100

    total_change = (
        (final_price / last_price) - 1
    ) * 100

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Giá phiên tiếp theo",
            f"{next_price:,.2f}",
            delta=f"{next_change:+.2f}%"
        )

    with b:
        st.metric(
            f"Giá sau {forecast_days} phiên",
            f"{final_price:,.2f}",
            delta=f"{total_change:+.2f}%"
        )

    with c:
        direction = (
            "TĂNG" if total_change > 0
            else "GIẢM" if total_change < 0
            else "ĐI NGANG"
        )

        st.metric(
            "Xu hướng mô hình dự báo",
            direction
        )

    st.pyplot(
        plot_future_forecast(
            df,
            forecast_df,
            ticker
        )
    )

    st.markdown("#### Bảng dự báo chi tiết")

    display_df = forecast_df.copy()
    display_df.index = display_df.index.strftime(
        "%d/%m/%Y"
    )

    st.dataframe(
        display_df.style.format({
            "Giá dự báo": "{:,.2f}",
            "Thay đổi so với giá cuối (%)": "{:+.2f}%"
        }),
        use_container_width=True
    )

    # ---------------------------------------------
    # TẢI KẾT QUẢ
    # ---------------------------------------------

    st.subheader("5. Xuất kết quả dự báo")

    csv_data = forecast_df.to_csv(
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 Tải bảng dự báo CSV",
        data=csv_data,
        file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    comparison_csv = comparison_df.to_csv(
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 Tải kết quả kiểm thử CSV",
        data=comparison_csv,
        file_name=f"kiem_thu_{ticker.replace('.', '_')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ---------------------------------------------
    # NHẬN XÉT TỰ ĐỘNG
    # ---------------------------------------------

    st.subheader("6. Nhận xét từ mô hình AI")

    if total_change > 2:
        st.info(
            f"Mô hình LSTM dự báo {ticker} "
            f"có xu hướng tăng khoảng "
            f"{total_change:.2f}% trong "
            f"{forecast_days} phiên tới."
        )

    elif total_change < -2:
        st.warning(
            f"Mô hình LSTM dự báo {ticker} "
            f"có xu hướng giảm khoảng "
            f"{abs(total_change):.2f}% trong "
            f"{forecast_days} phiên tới."
        )

    else:
        st.info(
            f"Mô hình dự báo {ticker} "
            f"có xu hướng tương đối ổn định "
            f"trong {forecast_days} phiên tới."
        )

    if metrics["R2"] < 0:
        st.warning(
            "R² âm cho thấy mô hình kiểm thử kém hơn "
            "dự báo hằng số bằng giá trung bình "
            "của tập kiểm thử."
        )

    st.caption(
        "Lưu ý: Dự báo nhiều phiên sử dụng phương pháp "
        "đệ quy, tức là dự báo phiên trước được dùng làm "
        "đầu vào cho phiên sau. Sai số có thể tích lũy. "
        "Ngày dự kiến chưa loại trừ ngày nghỉ lễ "
        "của từng thị trường."
    )

    st.warning(
        "Ứng dụng chỉ phục vụ mục đích học tập, "
        "nghiên cứu và tham khảo. Dự báo AI "
        "không bảo đảm phản ánh giá thị trường "
        "trong tương lai và không phải khuyến nghị "
        "mua hoặc bán cổ phiếu."
    )

else:
    st.markdown("### 🧠 Quy trình hoạt động")

    st.markdown("""
    **Bước 1:** Chọn mã cổ phiếu và khoảng thời gian.

    **Bước 2:** Tải dữ liệu giá đóng cửa từ Yahoo Finance.

    **Bước 3:** Chuẩn hóa dữ liệu bằng MinMaxScaler.

    **Bước 4:** Tạo các chuỗi dữ liệu gồm 60 phiên.

    **Bước 5:** Huấn luyện mô hình LSTM hai tầng.

    **Bước 6:** Kiểm thử và đánh giá độ chính xác.

    **Bước 7:** Dự báo giá trong tương lai và xuất kết quả.
    """)

st.divider()

st.caption(
    "AI Stock Forecast | Streamlit + TensorFlow LSTM "
    "| Data: Yahoo Finance"
)
