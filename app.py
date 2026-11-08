import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Dự đoán Giá Nhà - Nhóm 4", page_icon="🏠", layout="centered"
)

st.title("🏠 Ứng Dụng Dự Đoán Giá Nhà")
st.write(
    "Nhập các thông số kỹ thuật của ngôi nhà để hệ thống dự đoán mức giá bán"
    " phù hợp dựa trên dữ liệu lịch sử."
)

# Sidebar cho phép chọn mô hình
st.sidebar.header("⚙️ Cấu hình mô hình")
model_choice = st.sidebar.selectbox(
    "Chọn thuật toán dự đoán:",
    ["Hồi quy tuyến tính (Linear Regression)", "Random Forest"],
)


# Hàm tải dữ liệu và huấn luyện mô hình tương ứng
@st.cache_resource
def load_and_train_model(choice):
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

  # Lựa chọn mô hình theo yêu cầu
  if choice == "Hồi quy tuyến tính (Linear Regression)":
    model = LinearRegression()
  else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

  model.fit(X, y)
  return model, df


try:
  model, df_raw = load_and_train_model(model_choice)
except Exception as e:
  st.error(
      f"Lỗi khi đọc file dữ liệu 'Nhom4_gia_nha.xlsx': {e}. Hãy chắc chắn file"
      " nằm cùng thư mục!"
  )
  st.stop()

st.sidebar.header("🏠 Thông số nhập vào nhà")


# Tạo thanh trượt nhập thông số
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

# Nút dự đoán
if st.button("🚀 Dự đoán Giá Nhà", type="primary"):
  prediction = model.predict(input_df)[0]
  st.success(
      f"💰 Giá nhà dự kiến ước tính: **${prediction:,.2f} USD** (khoảng"
      f" {(prediction * 25400):,.0f} VNĐ)"
  )

st.markdown("---")
st.info(f"💡 Ứng dụng hiện đang sử dụng mô hình: **{model_choice}**.")