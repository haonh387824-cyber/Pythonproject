import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Cấu hình giao diện trang web
st.set_page_config(page_title="Quản lý điểm sinh viên", layout="wide")

# 1. Khởi tạo dữ liệu
@st.cache_data
def load_data():
    data = {
        'Họ và tên': [
            'Nguyễn Văn An', 'Trần Thị Bích', 'Lê Hoàng Cường', 'Phạm Minh Đức',
            'Vũ Thị Hoa', 'Hoàng Văn Khoa', 'Đặng Lan Huệ', 'Bùi Quốc Nam',
            'Đỗ Thu Phương', 'Ngô Quang Vinh'
        ],
        'Điểm chuyên cần': [10.0, 9.0, 8.5, 7.0, 10.0, 6.5, 9.5, 8.0, 9.0, 7.5],
        'Điểm giữa kỳ':    [8.5,  7.5, 9.0, 6.0,  8.5, 5.5, 9.5, 7.0, 8.5, 6.5],
        'Điểm cuối kỳ':     [9.0,  8.0, 8.5, 7.5,  9.0, 6.0, 9.0, 7.5, 9.0, 8.0]
    }
    df = pd.DataFrame(data)
    
    # Tính điểm tổng kết
    df['Điểm tổng kết'] = (
        df['Điểm chuyên cần'] * 0.1 + 
        df['Điểm giữa kỳ'] * 0.3 + 
        df['Điểm cuối kỳ'] * 0.6
    ).round(2)

    # Hàm xếp loại
    def xep_loai(diem):
        if diem >= 8.5:
            return 'Giỏi'
        elif diem >= 7.0:
            return 'Khá'
        elif diem >= 5.0:
            return 'Trung bình'
        else:
            return 'Yếu'

    df['Xếp loại'] = df['Điểm tổng kết'].apply(xep_loai)
    return df

df = load_data()

# 2. Tiêu đề ứng dụng
st.title("QUẢN LÝ ĐIỂM SINH VIÊN")

# 3. Hiển thị bảng điểm
st.subheader("📋 Bảng điểm sinh viên")
st.dataframe(df, use_container_width=True)

# 4. Hiển thị các chỉ số thống kê
st.subheader("📊 Thống kê chung")
col1, col2, col3, col4 = st.columns(4)

diem_tb = df['Điểm tổng kết'].mean()
max_sv = df.loc[df['Điểm tổng kết'].idxmax()]
min_sv = df.loc[df['Điểm tổng kết'].idxmin()]
so_sv_dat = (df['Điểm tổng kết'] >= 5.0).sum()

col1.metric("Điểm TB cả lớp", f"{diem_tb:.2f}")
col2.metric("Điểm cao nhất", f"{max_sv['Điểm tổng kết']}", f"SV: {max_sv['Họ và tên']}")
col3.metric("Điểm thấp nhất", f"{min_sv['Điểm tổng kết']}", f"SV: {min_sv['Họ và tên']}")
col4.metric("Số SV đạt (≥ 5.0)", f"{so_sv_dat}/{len(df)}")

st.divider()

# 5. Danh sách chọn sinh viên (Selectbox)
st.subheader("🔍 Tra cứu chi tiết từng sinh viên")
selected_student = st.selectbox("Chọn sinh viên từ danh sách:", df['Họ và tên'])

# Lọc thông tin của sinh viên được chọn
sv_info = df[df['Họ và tên'] == selected_student].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Điểm chuyên cần", sv_info['Điểm chuyên cần'])
c2.metric("Điểm giữa kỳ", sv_info['Điểm giữa kỳ'])
c3.metric("Điểm cuối kỳ", sv_info['Điểm cuối kỳ'])
c4.metric("Điểm tổng kết", sv_info['Điểm tổng kết'])
c5.metric("Xếp loại", sv_info['Xếp loại'])

st.divider()

# 6. Biểu đồ cột thể hiện điểm tổng kết
st.subheader("📈 Biểu đồ điểm tổng kết")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(df['Họ và tên'], df['Điểm tổng kết'], color='#2b5c8f')
ax.set_title('ĐIỂM TỔNG KẾT CỦA 10 SINH VIÊN', fontsize=12, fontweight='bold')
ax.set_xlabel('Họ và tên sinh viên', fontsize=10)
ax.set_ylabel('Điểm tổng kết', fontsize=10)
ax.set_ylim(0, 10.5)
plt.xticks(rotation=45, ha='right')

# Hiển thị số điểm trên đỉnh từng cột
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f'{yval}', ha='center', va='bottom', fontsize=8)

st.pyplot(fig)