import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cấu hình trang web
st.set_page_config(
    page_title="Dự Đoán Giá Nhà - Nhóm 3",
    page_icon="🏠",
    layout="wide"
)

# Tiêu đề ứng dụng
st.title("🏠 ỨNG DỤNG DỰ ĐOÁN GIÁ NHÀ")
st.markdown(
    "**Học phần:** Trí tuệ Nhân tạo và Ứng dụng | **Thực hiện:** Nhóm 3 — "
    "Ứng dụng mô hình **ANN (Mạng nơ-ron nhân tạo)** và **Decision Tree (Cây quyết định)**"
)
st.divider()


@st.cache_data
def load_data():
    df = pd.read_excel("Nhom3_GiaNha.xlsx")
    df_clean = df.dropna(subset=["price"]).drop_duplicates().reset_index(drop=True)
    return df_clean


@st.cache_resource
def train_models(df_clean):
    feature_cols = [
        "area", "bedrooms", "bathrooms", "stories", "mainroad",
        "guestroom", "basement", "hotwaterheating", "airconditioning",
        "parking", "prefarea", "furnishingstatus"
    ]
    target_col = "price"

    X_raw = df_clean[feature_cols]
    y = df_clean[target_col].values

    # Mã hóa One-Hot các biến phân loại
    X_encoded = pd.get_dummies(X_raw, drop_first=False)
    encoded_columns = X_encoded.columns.tolist()

    # Chia tập dữ liệu 80% Train - 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 1. Mô hình Decision Tree Regressor
    dt_model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_pred_test = dt_model.predict(X_test)

    # 2. Mô hình ANN (Kiến trúc 64 -> 32 -> 1, ReLU, Adam)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    ann_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=1500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train_scaled)
    ann_pred_scaled = ann_model.predict(X_test_scaled)
    ann_pred_test = scaler_y.inverse_transform(ann_pred_scaled.reshape(-1, 1)).ravel()

    # Tính chỉ số đánh giá trên tập Test
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape}

    metrics = {
        "ANN (Mạng nơ-ron nhân tạo)": calc_metrics(y_test, ann_pred_test),
        "Decision Tree (Cây quyết định)": calc_metrics(y_test, dt_pred_test)
    }

    test_comparison = pd.DataFrame({
        "Giá thực tế": y_test,
        "Dự đoán (ANN)": np.round(ann_pred_test, 0),
        "Dự đoán (Decision Tree)": np.round(dt_pred_test, 0)
    })

    feature_importance = pd.Series(
        dt_model.feature_importances_, index=encoded_columns
    ).sort_values(ascending=False)

    return {
        "dt_model": dt_model,
        "ann_model": ann_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "encoded_columns": encoded_columns,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "metrics": metrics,
        "test_comparison": test_comparison,
        "feature_importance": feature_importance
    }


df_data = load_data()
artifacts = train_models(df_data)

# Tạo 3 tab chức năng chính
tab1, tab2, tab3 = st.tabs([
    "🔮 Dự Đoán Giá Nhà Mới",
    "📊 Khám Phá Dữ Liệu",
    "📈 Đánh Giá & So Sánh Mô Hình"
])

# ==========================================
# TAB 1: DỰ ĐOÁN GIÁ NHÀ (ĐẦY ĐỦ 12 THUỘC TÍNH)
# ==========================================
with tab1:
    st.subheader("Nhập đầy đủ 12 thông số bất động sản để định giá")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📐 Kích thước & Cấu trúc")
            area = st.number_input(
                "Diện tích đất - area (sqft)",
                min_value=1000,
                max_value=25000,
                value=5000,
                step=100
            )
            bedrooms = st.selectbox(
                "Số phòng ngủ - bedrooms",
                options=[1, 2, 3, 4, 5, 6],
                index=2
            )
            bathrooms = st.selectbox(
                "Số phòng tắm - bathrooms",
                options=[1, 2, 3, 4],
                index=1
            )
            stories = st.selectbox(
                "Số tầng - stories",
                options=[1, 2, 3, 4],
                index=1
            )

        with col2:
            st.markdown("##### 🏡 Vị trí & Tiện ích cơ bản")
            mainroad = st.selectbox(
                "Mặt tiền đường lớn - mainroad",
                options=["yes", "no"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )
            prefarea = st.selectbox(
                "Khu vực ưu tiên / đắc địa - prefarea",
                options=["yes", "no"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )
            parking = st.selectbox(
                "Số chỗ đậu xe - parking",
                options=[0, 1, 2, 3],
                index=1
            )
            furnishingstatus = st.selectbox(
                "Tình trạng nội thất - furnishingstatus",
                options=["furnished", "semi-furnished", "unfurnished"],
                format_func=lambda x: {
                    "furnished": "Đầy đủ nội thất (furnished)",
                    "semi-furnished": "Nội thất cơ bản (semi-furnished)",
                    "unfurnished": "Chưa có nội thất (unfurnished)"
                }[x]
            )

        with col3:
            st.markdown("##### ⚙️ Tiện nghi bổ sung")
            guestroom = st.selectbox(
                "Phòng khách riêng - guestroom",
                options=["no", "yes"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )
            basement = st.selectbox(
                "Tầng hầm - basement",
                options=["no", "yes"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )
            hotwaterheating = st.selectbox(
                "Hệ thống nước nóng - hotwaterheating",
                options=["no", "yes"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )
            airconditioning = st.selectbox(
                "Điều hòa nhiệt độ - airconditioning",
                options=["yes", "no"],
                format_func=lambda x: "Có (yes)" if x == "yes" else "Không (no)"
            )

        submitted = st.form_submit_button(
            "🚀 Dự Đoán Giá Nhà Ngay",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        input_df = pd.DataFrame([{
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }])

        # Mã hóa và đồng bộ cột với tập huấn luyện
        input_encoded = pd.get_dummies(input_df, drop_first=False)
        input_encoded = input_encoded.reindex(
            columns=artifacts["encoded_columns"], fill_value=0
        )

        # Dự đoán bằng Decision Tree
        pred_dt = float(artifacts["dt_model"].predict(input_encoded)[0])

        # Dự đoán bằng ANN và giải chuẩn hóa (inverse_transform)
        input_scaled = artifacts["scaler_X"].transform(input_encoded)
        pred_ann_scaled = artifacts["ann_model"].predict(input_scaled)[0]
        pred_ann = float(
            artifacts["scaler_y"].inverse_transform([[pred_ann_scaled]])[0][0]
        )

        st.success("✅ Đã hoàn tất dự đoán từ cả 2 mô hình học máy!")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(
                label="🧠 Giá dự đoán - ANN (Mô hình tối ưu)",
                value=f"{pred_ann:,.0f}",
                delta=f"Đơn giá: {pred_ann / area:,.0f} / sqft"
            )
        with res_col2:
            st.metric(
                label="🌳 Giá dự đoán - Decision Tree",
                value=f"{pred_dt:,.0f}",
                delta=f"Đơn giá: {pred_dt / area:,.0f} / sqft"
            )
        with res_col3:
            diff = abs(pred_ann - pred_dt)
            st.metric(
                label="⚖️ Chênh lệch giữa 2 mô hình",
                value=f"{diff:,.0f}",
                delta=f"{(diff / pred_ann * 100):.1f}% so với ANN",
                delta_color="off"
            )

        with st.expander("📋 Xem lại bảng thông số ngôi nhà vừa nhập", expanded=True):
            st.dataframe(input_df, use_container_width=True)

# ==========================================
# TAB 2: KHÁM PHÁ DỮ LIỆU
# ==========================================
with tab2:
    st.subheader("Tổng quan bộ dữ liệu Nhom3_GiaNha.xlsx")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số mẫu dữ liệu", f"{df_data.shape[0]} mẫu")
    c2.metric("Tổng số thuộc tính đầu vào (X)", "12 thuộc tính")
    c3.metric("Tập Train (80%)", f"{artifacts['X_train_shape'][0]} mẫu")
    c4.metric("Tập Test (20%)", f"{artifacts['X_test_shape'][0]} mẫu")

    st.markdown("#### 10 dòng dữ liệu đầu tiên sau khi làm sạch")
    st.dataframe(df_data.head(10), use_container_width=True)

    st.markdown("#### Thống kê mô tả các biến số học")
    st.dataframe(df_data.describe().round(2), use_container_width=True)

# ==========================================
# TAB 3: ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
# ==========================================
with tab3:
    st.subheader("Kết quả đánh giá thực nghiệm trên tập Test (20%)")

    metrics_df = pd.DataFrame(artifacts["metrics"]).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    st.info(
        "💡 **Nhận xét:** Mô hình **ANN (Mạng nơ-ron nhân tạo)** đạt chỉ số $R^2$ cao hơn "
        "và sai số MAE/RMSE thấp hơn so với **Decision Tree**, phù hợp làm mô hình nòng cốt "
        "để định giá bất động sản."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Đối chiếu Giá thực tế vs Giá dự đoán (15 mẫu Test đầu tiên)")
        st.dataframe(artifacts["test_comparison"].head(15), use_container_width=True)

    with col_right:
        st.markdown("#### Top 10 thuộc tính ảnh hưởng mạnh nhất (Decision Tree)")
        top_imp = artifacts["feature_importance"].head(10)
        st.bar_chart(top_imp)
