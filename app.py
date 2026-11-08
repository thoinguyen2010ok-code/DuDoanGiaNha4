import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Phân tích & Dự đoán Giá Nhà - Nhóm 4",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Ứng Dụng Phân Tích & Dự Đoán Giá Nhà")
st.write(
    "Thay đổi thông số ở thanh bên trái để xem kết quả cập nhật tự động ngay"
    " lập tức."
)

# Sidebar cho phép chọn mô hình
st.sidebar.header("⚙️ Cấu hình mô hình")
model_choice = st.sidebar.selectbox(
    "Chọn thuật toán:",
    ["Hồi quy tuyến tính (Linear Regression)", "K-Means (Phân cụm nhà)"],
)


# Hàm tải dữ liệu và huấn luyện mô hình
@st.cache_resource
def load_and_train_model():
  df = pd.read_excel("Nhom4_gia_nha.xlsx")

  features = [
      "OverallQual",
      "GrLivArea",
      "GarageCars",
      "TotalBsmtSF",
      "FullBath",
      "YearBuilt",
  ]
  target = "SalePrice"

  data_model = df[features + [target]].dropna()
  X = data_model[features]
  y = data_model[target]

  # Huấn luyện Linear Regression
  lr_model = LinearRegression()
  lr_model.fit(X, y)

  # Huấn luyện K-Means (3 cụm)
  kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
  kmeans_model.fit(X)
  data_model["Cluster"] = kmeans_model.labels_
  cluster_prices = data_model.groupby("Cluster")[target].mean().to_dict()

  return df, lr_model, kmeans_model, cluster_prices, features


try:
  df_raw, lr_model, kmeans_model, cluster_prices, features = (
      load_and_train_model()
  )
except Exception as e:
  st.error(
      f"Lỗi khi đọc file dữ liệu 'Nhom4_gia_nha.xlsx': {e}. Hãy chắc chắn file"
      " nằm cùng thư mục!"
  )
  st.stop()

st.sidebar.header("🏠 Thông số nhập vào nhà")


# Tạo thanh trượt nhập thông số (Tự động cập nhật realtime)
def user_input_features():
  overall_qual = st.sidebar.slider(
      "Chất lượng tổng thể (OverallQual - Thang 1-10)",
      int(df_raw["OverallQual"].min()),
      int(df_raw["OverallQual"].max()),
      int(df_raw["OverallQual"].median()),
  )
  gr_liv_area = st.sidebar.slider(
      "Diện tích sinh hoạt (GrLivArea - sqft)",
      int(df_raw["GrLivArea"].min()),
      int(df_raw["GrLivArea"].max()),
      int(df_raw["GrLivArea"].median()),
  )
  garage_cars = st.sidebar.slider(
      "Sức chứa Gara (số ô tô)",
      int(df_raw["GarageCars"].min()),
      int(df_raw["GarageCars"].max()),
      int(df_raw["GarageCars"].median()),
  )
  total_bsmt_sf = st.sidebar.slider(
      "Diện tích tầng hầm (TotalBsmtSF - sqft)",
      int(df_raw["TotalBsmtSF"].min()),
      int(df_raw["TotalBsmtSF"].max()),
      int(df_raw["TotalBsmtSF"].median()),
  )
  full_bath = st.sidebar.slider(
      "Số phòng tắm đầy đủ (FullBath)",
      int(df_raw["FullBath"].min()),
      int(df_raw["FullBath"].max()),
      int(df_raw["FullBath"].median()),
  )
  year_built = st.sidebar.slider(
      "Năm xây dựng (YearBuilt)",
      int(df_raw["YearBuilt"].min()),
      int(df_raw["YearBuilt"].max()),
      int(df_raw["YearBuilt"].median()),
  )

  data = {
      "OverallQual": overall_qual,
      "GrLivArea": gr_liv_area,
      "GarageCars": garage_cars,
      "TotalBsmtSF": total_bsmt_sf,
      "FullBath": full_bath,
      "YearBuilt": year_built,
  }
  return pd.DataFrame(data, index=[0])


input_df = user_input_features()

st.subheader("📋 Thông số ngôi nhà bạn chọn:")
st.dataframe(input_df)

st.markdown("---")
st.subheader("📊 Kết quả phân tích tự động:")

# Xử lý hiển thị kết quả ngay lập tức mà không cần bấm nút
if model_choice == "Hồi quy tuyến tính (Linear Regression)":
  prediction = lr_model.predict(input_df)[0]
  st.success(
      f"💰 **Giá nhà dự kiến ước tính (Linear Regression):**"
      f" **${prediction:,.2f} USD** (khoảng {(prediction * 25400):,.0f} VNĐ)"
  )
else:
  cluster_id = kmeans_model.predict(input_df)[0]
  avg_price = cluster_prices.get(cluster_id, 0)

  # Tính khoảng cách đến các tâm cụm để thấy sự dao động khi đổi thông số
  distances = kmeans_model.transform(input_df)[0]
  current_distance = distances[cluster_id]

  st.markdown(
      f"""
        <div style="padding: 15px; background-color: #d4edda; border-radius: 5px; color: #155724; border: 1px solid #c3e6cb;">
            <b>🔍 Phân khúc phân nhóm: Cụm số {cluster_id + 1} (K-Means)</b><br>
            📊 <b>Mức giá trung bình của phân khúc này:</b> ${avg_price:,.2f} USD (khoảng {(avg_price * 25400):,.0f} VNĐ)<br>
            📐 <i>Khoảng cách tâm cụm (Distance): {current_distance:.2f} (Thay đổi diện tích/chất lượng lớn sẽ làm đổi cụm hoặc thay đổi khoảng cách này)</i>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.markdown("---")
st.info(f"💡 Thuật toán đang hoạt động: **{model_choice}**.")