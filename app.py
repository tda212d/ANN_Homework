import streamlit as st
import pandas as pd

# ==========================================
# 1. ส่วนตั้งค่าหน้าจอและ UI
# ==========================================
st.set_page_config(page_title="HR Screening Neural Network", layout="wide")
st.title("Neural Network เบื้องต้น: ระบบคัดกรองพนักงาน (HR Screening)")
st.markdown("โปรแกรมจำลอง **Single Layer Perceptron** เพื่อพยากรณ์การเรียกสัมภาษณ์งาน")

#-------------------โค้ดฟอนต์---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

html, body, [class*="css"], [class*="st-"], h1, h2, h3, h4, h5, h6 {
    font-family: 'Kanit', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
#----------------------------------------------------

# ==========================================
# 2. กระบวนการเตรียมข้อมูล (Data Preparation - แบบอัปโหลดผ่านเว็บ)
# ==========================================
st.header("1. นำเข้าชุดข้อมูล (Import Dataset)")

# สร้างวิดเจ็ตสำหรับอัปโหลดไฟล์
uploaded_file = st.file_uploader("📂 กรุณาอัปโหลดไฟล์ CSV ของคุณเพื่อเริ่มต้นการทำงาน", type=['csv'])

# ตรวจสอบว่ามีการอัปโหลดไฟล์เข้ามาหรือไม่
if uploaded_file is not None:
    try:
        # อ่านไฟล์ที่ผู้ใช้อัปโหลด
        df = pd.read_csv(uploaded_file)
        
        # เปลี่ยนชื่อคอลัมน์ให้เรียกใช้งานง่ายในโค้ด (อ้างอิงจากโครงสร้างข้อมูล HR Screening)
        df.columns = ['ลำดับ', 'Exp_x1', 'Skill_x2', 'Target_y', 'คำอธิบาย']
        
        st.success("โหลดข้อมูลสำเร็จ!")
        st.dataframe(df, use_container_width=True)
        
        # ==========================================
        # 3. กำหนดค่าพารามิเตอร์ (Hyperparameters)
        # ==========================================
        st.header("2. กำหนดค่าเริ่มต้นของโมเดล")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w1 = st.number_input("Weight 1 (w1)", value=0.5, step=0.1)
        with col2:
            w2 = st.number_input("Weight 2 (w2)", value=0.5, step=0.1)
        with col3:
            theta = st.number_input("Threshold (θ)", value=1.0, step=0.1)
        with col4:
            alpha = st.number_input("Learning Rate (α)", value=0.2, step=0.1)

        epochs = st.slider("จำนวนรอบการเรียนรู้ (Epochs)", min_value=1, max_value=20, value=5)

        # ==========================================
        # 4. ฟังก์ชันกระตุ้น (Activation Function)
        # ==========================================
        def threshold_function(net):
            return 1 if net >= 0 else 0

        # ==========================================
        # 5. กระบวนการเรียนรู้ (Training Process)
        # ==========================================
        st.header("3. กระบวนการเรียนรู้ของโมเดล")
        if st.button("เริ่มฝึกสอนโมเดล (Train Model)"):
            
            history = []
            
            for epoch in range(epochs):
                for index, row in df.iterrows():
                    x1 = row['Exp_x1']
                    x2 = row['Skill_x2']
                    y = row['Target_y']
                    
                    net = (w1 * x1 + w2 * x2) - theta
                    y_hat = threshold_function(net)
                    
                    error = y - y_hat
                    
                    if error != 0:
                        w1 = w1 + (alpha * error * x1)
                        w2 = w2 + (alpha * error * x2)
                        theta = theta - (alpha * error) 
                    
                    history.append({
                        "Epoch": epoch + 1, "Data": index + 1, 
                        "x1": x1, "x2": x2, "y": y, "y^": y_hat, 
                        "Error": error, "New w1": round(w1,4), "New w2": round(w2,4), "New θ": round(theta,4)
                    })
                    
            st.success(f"ฝึกสอนเสร็จสิ้น! ค่าน้ำหนักสุดท้าย: w1 = {round(w1,4)}, w2 = {round(w2,4)}, θ = {round(theta,4)}")
            st.dataframe(pd.DataFrame(history))

            st.session_state['trained_w1'] = w1
            st.session_state['trained_w2'] = w2
            st.session_state['trained_theta'] = theta

        # ==========================================
        # 6. การพยากรณ์ (Prediction)
        # ==========================================
        st.header("4. ทดสอบพยากรณ์ผลลัพธ์")
        colA, colB = st.columns(2)
        with colA:
            test_x1 = st.slider("ระบุประสบการณ์การทำงาน (Exp: 0.0 - 1.0)", 0.0, 1.0, 0.5)
        with colB:
            test_x2 = st.slider("ระบุคะแนนทักษะ (Skill: 0.0 - 1.0)", 0.0, 1.0, 0.5)

        if st.button("🔮 พยากรณ์ผล (Predict)"):
            if 'trained_w1' in st.session_state:
                tw1 = st.session_state['trained_w1']
                tw2 = st.session_state['trained_w2']
                ttheta = st.session_state['trained_theta']
                
                net_result = (tw1 * test_x1 + tw2 * test_x2) - ttheta
                prediction = threshold_function(net_result)
                
                if prediction == 1:
                    st.success(f" ผลการพยากรณ์: **1 (เรียกสัมภาษณ์งาน)** (ค่า Net = {round(net_result, 4)})")
                else:
                    st.error(f"❌ ผลการพยากรณ์: **0 (ไม่ผ่านเกณฑ์)** (ค่า Net = {round(net_result, 4)})")
            else:
                st.warning("กรุณากดปุ่ม 'เริ่มฝึกสอนโมเดล' ก่อนครับ!")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
else:

    st.info(" โปรดลากไฟล์ CSV มาวาง หรือคลิกที่ปุ่ม Browse files เพื่อเริ่มต้น")


