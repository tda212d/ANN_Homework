import streamlit as st
import pandas as pd

# ==========================================
# 1. ส่วนตั้งค่าหน้าจอและฟอนต์ (UI & Typography)
# ==========================================
st.set_page_config(page_title="HR Screening Neural Network", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap');

html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6 {
    font-family: 'Kanit', sans-serif !important;
}
/* ตกแต่งกรอบของการพยากรณ์ให้ดูโดดเด่น */
div[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #00E5FF;
}
</style>
""", unsafe_allow_html=True)

# ส่วนหัวของเว็บไซต์ (Header)
st.title(" Neural Network: ระบบคัดกรองพนักงาน (HR Screening)")
st.markdown("จำลองโครงข่ายประสาทเทียม **Single Layer Perceptron** เพื่อพยากรณ์โอกาสในการเรียกสัมภาษณ์งาน")
st.divider() # เส้นคั่นเพิ่ม White Space

# ==========================================
# 2. แถบตั้งค่าด้านข้าง (Sidebar Layout) - จัดกลุ่มลดความแออัด
# ==========================================
with st.sidebar:
    st.header("ตั้งค่าพารามิเตอร์")
    st.markdown("ปรับแต่งค่าของโมเดล (Hyperparameters)")
    
    w1 = st.number_input("Weight 1 (ประสบการณ์)", value=0.5, step=0.1)
    w2 = st.number_input("Weight 2 (ทักษะ)", value=0.5, step=0.1)
    theta = st.number_input("Threshold (θ เกณฑ์)", value=1.0, step=0.1)
    alpha = st.number_input("Learning Rate (α)", value=0.2, step=0.1)
    epochs = st.slider("รอบการเรียนรู้ (Epochs)", min_value=1, max_value=20, value=5)
    
    st.info("โมเดลจะนำค่าเหล่านี้ไปใช้เป็นจุดเริ่มต้นในการเรียนรู้")

# ==========================================
# 3. พื้นที่หลัก: นำเข้าข้อมูล (Main Content: Data Import)
# ==========================================
st.header("1. 📂 นำเข้าชุดข้อมูล (Data Import)")
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV (hr_dataset.csv) เพื่อเริ่มต้น", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = ['ลำดับ', 'Exp_x1', 'Skill_x2', 'Target_y', 'คำอธิบาย']
        
        # แสดงตารางแบบซ่อนได้ (Expander) เพื่อประหยัดพื้นที่
        with st.dataframe(df, use_container_width=True)
            
        st.divider()

        # ==========================================
        # 4. พื้นที่หลัก: กระบวนการเรียนรู้ (Main Content: Training)
        # ==========================================
        def threshold_function(net):
            return 1 if net >= 0 else 0

        st.header("2. 🚀 กระบวนการเรียนรู้ (Training Process)")
        
        if st.button("เริ่มฝึกสอนโมเดล (Train Model)", use_container_width=True):
            history = []
            
            # ลูปจำลองการเรียนรู้
            for epoch in range(epochs):
                for index, row in df.iterrows():
                    x1, x2, y = row['Exp_x1'], row['Skill_x2'], row['Target_y']
                    
                    net = (w1 * x1 + w2 * x2) - theta
                    y_hat = threshold_function(net)
                    error = y - y_hat
                    
                    if error != 0:
                        w1 += (alpha * error * x1)
                        w2 += (alpha * error * x2)
                        theta -= (alpha * error) 
                    
                    history.append({
                        "Epoch": epoch + 1, "Row": index + 1, 
                        "x1": x1, "x2": x2, "Target": y, "Predict": y_hat, 
                        "Error": error, "New w1": round(w1,4), "New w2": round(w2,4), "New θ": round(theta,4)
                    })
            
            # จัด Layout แสดงผลลัพธ์ค่าน้ำหนักด้วย st.metric ให้ดูเป็น Dashboard มืออาชีพ
            st.success("🎉 โมเดลเรียนรู้เสร็จสิ้น! ได้ค่าน้ำหนักที่เหมาะสมที่สุดดังนี้:")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Weight 1 (w1)", round(w1,4))
            col_m2.metric("Weight 2 (w2)", round(w2,4))
            col_m3.metric("Threshold (θ)", round(theta,4))
            
            st.markdown("**ตารางบันทึกการปรับค่าน้ำหนักในแต่ละรอบ (History):**")
            st.dataframe(pd.DataFrame(history), height=250, use_container_width=True)

            # เก็บค่าไว้พยากรณ์
            st.session_state['trained_w1'] = w1
            st.session_state['trained_w2'] = w2
            st.session_state['trained_theta'] = theta

        st.divider()

        # ==========================================
        # 5. พื้นที่หลัก: การพยากรณ์ (Main Content: Prediction)
        # ==========================================
        st.header("3. 🔮 ทดสอบพยากรณ์ (Prediction)")
        st.markdown("จำลองปรับค่าคุณสมบัติของผู้สมัคร เพื่อให้ AI ทำนายผลการคัดกรอง")
        
        # จัด Layout แบ่งซ้าย-ขวา สำหรับส่วนกรอกข้อมูล
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            test_x1 = st.slider("ประสบการณ์ทำงาน (Exp)", 0.0, 1.0, 0.5, help="0.0 = ไม่มีประสบการณ์, 1.0 = มีประสบการณ์สูง")
        with col_pred2:
            test_x2 = st.slider("คะแนนทักษะ (Skill)", 0.0, 1.0, 0.5, help="0.0 = ทักษะต่ำ, 1.0 = ทักษะดีเยี่ยม")

        st.write("") # เคาะบรรทัดเพิ่ม White space

        if st.button("ประมวลผลการตัดสินใจ (Predict)", type="primary"):
            if 'trained_w1' in st.session_state:
                tw1 = st.session_state['trained_w1']
                tw2 = st.session_state['trained_w2']
                ttheta = st.session_state['trained_theta']
                
                net_result = (tw1 * test_x1 + tw2 * test_x2) - ttheta
                prediction = threshold_function(net_result)
                
                if prediction == 1:
                    st.success(f"### 🎉 ผลการพยากรณ์: 'เรียกสัมภาษณ์งาน' (Pass)\nค่า Net = {round(net_result, 4)} ซึ่งผ่านเกณฑ์ (>= 0)")
                else:
                    st.error(f"### ❌ ผลการพยากรณ์: 'ไม่ผ่านเกณฑ์' (Fail)\nค่า Net = {round(net_result, 4)} ซึ่งต่ำกว่าเกณฑ์ (< 0)")
            else:
                st.warning("⚠️ กรุณากดปุ่ม 'เริ่มฝึกสอนโมเดล' ในข้อ 2 ก่อน เพื่อให้ AI เรียนรู้ค่าน้ำหนักครับ!")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

