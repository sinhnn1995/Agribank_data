import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import time # Thêm thư viện time để mô phỏng độ trễ (nếu cần)

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)
st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Khởi tạo Gemini Client (Dùng chung cho cả Phân tích và Chat) ---
# Lấy Khóa API từ Streamlit Secrets
API_KEY = st.secrets.get("GEMINI_API_KEY")
GEMINI_CLIENT = None

if API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(api_key=API_KEY)
    except Exception as e:
        # Hiển thị cảnh báo nếu không thể khởi tạo client nhưng không làm dừng app
        st.warning(f"Lỗi khởi tạo Gemini Client: {e}. Vui lòng kiểm tra API Key trong Secrets.")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************

    return df

# --- Hàm gọi API Gemini ---
def get_ai_analysis(data_for_ai, client):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not client:
        return "Lỗi: Gemini Client chưa được khởi tạo. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets."

    try:
        model_name = 'gemini-2.5-flash'
        prompt = f"""
Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
Tuyệt đối không sử dụng các ký tự đặc biệt như /-strong, /-heart, :>, :o, :-((, :-h trong phản hồi.

Dữ liệu thô và chỉ số: {data_for_ai}
"""
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']

        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)

            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                st.warning("Nợ Ngắn Hạn bằng 0, không thể tính chỉ số Thanh toán Hiện hành.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"

            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")

            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)',
                    'Tăng trưởng Tài sản ngắn hạn (%)',
                    'Thanh toán hiện hành (N-1)',
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    (f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" 
                     if 'TÀI SẢN NGẮN HẠN' in df_processed['Chỉ tiêu'].str.upper().str.replace(' ', '').values else "N/A"),
                    f"{thanh_toan_hien_hanh_N_1}",
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)

            if st.button("Yêu cầu AI Phân tích"):
                if GEMINI_CLIENT:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, GEMINI_CLIENT)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API hoặc Client chưa được khởi tạo. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")


# ******************************************************************************
# --- PHẦN BỔ SUNG: CHỨC NĂNG CHAT VỚI GEMINI ---
# ******************************************************************************
st.markdown("---")
st.subheader("6. Hỗ trợ Hỏi Đáp với Gemini (gemini-2.5-flash)")

# Khởi tạo lịch sử chat trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Thêm tin nhắn chào mừng ban đầu
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Chào bạn! Tôi là trợ lý AI. Hãy hỏi tôi về bất kỳ thuật ngữ tài chính nào hoặc yêu cầu tôi giải thích thêm về kết quả phân tích nếu bạn đã tải file lên nhé."
    })

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input chat mới
if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
    # 1. Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Gọi Gemini API
    if GEMINI_CLIENT:
        try:
            # Xây dựng nội dung cho API call, bao gồm lịch sử
            # Chuyển định dạng lịch sử Streamlit sang định dạng API
            contents = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in st.session_state.messages
            ]
            
            with st.chat_message("assistant"):
                # Dùng spinner để hiển thị trạng thái đang chờ
                with st.spinner('Đang chờ phản hồi từ Gemini...'):
                    # Sử dụng model hỗ trợ chat
                    response = GEMINI_CLIENT.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=contents
                    )
                
                ai_response = response.text
                
                # 4. Thêm phản hồi của AI vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

                # 5. Hiển thị phản hồi của AI
                st.markdown(ai_response)
        
        except APIError as e:
            error_msg = f"Lỗi gọi Gemini API trong Chat: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
        except Exception as e:
            error_msg = f"Đã xảy ra lỗi không xác định trong Chat: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
    else:
        error_msg = "Không thể kết nối với Gemini. Vui lòng kiểm tra Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)
