import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# 設定 matplotlib 支援中文
rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體（Windows 系統）
rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

# 載入資料
df = pd.read_csv('df_exported.csv')
df_exported = df[df['DIABETES_SELF'] == 1].copy()
df['Spine_Tot_T_adj'] = df['Spine_Tot_T']
df['HipL_Tot_T_adj'] = df['HipL_Tot_T']
df['HipR_Tot_T_adj'] = df['HipR_Tot_T']

# 模型定義
def predict_spine_t(AGE, SEX, BODY_HEIGHT, BODY_WEIGHT, QUS):
    sex_coef = 2 if SEX == '女性' else 1
    return -13.2094 + 0.0483 * AGE + 3.7831 * sex_coef + 0.019 * BODY_HEIGHT + 0.0009 * BODY_WEIGHT + 0.0348 * QUS - 0.0582 * AGE * sex_coef + 0.0006 * AGE * BODY_WEIGHT

def predict_hipR_t(BODY_FAT_RATE, AGE, SEX, BODY_HEIGHT, BODY_WEIGHT, QUS):
    sex_coef = 2 if SEX == '女性' else 1
    return 11.1267 - 0.2742 * AGE - 0.4183 * sex_coef - 0.0995 * BODY_HEIGHT + 0.0419 * BODY_WEIGHT + 0.0244 * BODY_FAT_RATE + 0.0148 * QUS + 0.0017 * AGE * BODY_HEIGHT + 0.0071 * sex_coef * QUS - 0.0003 * BODY_WEIGHT * BODY_FAT_RATE

def predict_hipL_t(AGE, SEX, BODY_WEIGHT, QUS):
    sex_coef = 2 if SEX == '女性' else 1
    return -7.45 + 0.0355 * AGE + 1.4007 * sex_coef + 0.0366 * BODY_WEIGHT + 0.0187 * QUS - 0.026 * AGE * sex_coef + 0.005 * sex_coef * QUS

# Streamlit UI
# 在網頁最上方放置圖片
st.image(r"C:\Users\shjm9\Downloads\UR_Kenny\dia.jpg", use_container_width=True)
st.title("第二型糖尿病患 DXA 骨質密度校正")
st.markdown("""
本工具利用臨床資料預測第二型糖尿病患者在 **脊椎**、**右髖關節** 和 **左髖關節** 的 T 分數，協助使用者初步評估骨質健康狀態，並提供圖表與結論參考。
""")
st.markdown("""
1. 輸入 **年齡、性別、身高、體重、體脂率與超音波 T 分數，若不確定，可勾選「我不知道」將使用參考群體平均值**。    
2. 可選擇輸入各部位已知 DXA T 值，與模型預測做比較。  
""")



# 使用者輸入
AGE = st.slider("年齡", 18, 75, 46)
SEX = st.selectbox("生理性別", ["男性", "女性"])

# 體重輸入與選擇
temp_height = st.number_input("身高 (cm)", value=160.0, step=0.1)
use_default_height = st.checkbox("我不知道身高")
if use_default_height:
    BODY_HEIGHT = df_exported['BODY_HEIGHT'].mean()
    st.info(f"已使用平均身高：{BODY_HEIGHT:.1f} cm")
else:
    BODY_HEIGHT = temp_height

temp_weight = st.number_input("體重 (kg)", value=60.0, step=0.1)
use_default_weight = st.checkbox("我不知道體重")
if use_default_weight:
    BODY_WEIGHT = df_exported['BODY_WEIGHT'].mean()
    st.info(f"已使用平均體重：{BODY_WEIGHT:.1f} kg")
else:
    BODY_WEIGHT = temp_weight

# 體脂率輸入與選擇
temp_bf = st.number_input("體脂率 (%)", value=25.0, step=0.1)
use_default_bf = st.checkbox("我不知道體脂率")
if use_default_bf:
    BODY_FAT_RATE = df_exported['BODY_FAT_RATE'].mean()
    st.info(f"已使用平均體脂率：{BODY_FAT_RATE:.1f} %")
else:
    BODY_FAT_RATE = temp_bf

# 超音波輸入與選擇
temp_qus = st.number_input("超音波骨密度", value=80.0, step=0.1)
use_default_qus = st.checkbox("我不知道超音波骨密度")
if use_default_qus:
    QUS = df_exported['QUS'].mean()
    st.info(f"已使用平均超音波骨密度：{QUS:.1f}")
else:
    QUS = temp_qus

temp_hba1c = st.number_input("糖化血紅素", value=7.0, step=0.1)
use_default_hba1c = st.checkbox("我不知道糖化血紅素")
if use_default_hba1c:
    HBA1C = df_exported['HBA1C'].mean()
    st.info(f"已使用平均糖化血紅素：{HBA1C:.1f}")
else:
    HBA1C = temp_hba1c

# Spine T 值輸入與選擇
temp_spine = st.number_input("你輸入的脊椎 T 分數", value=0.0, step=0.1)
use_default_spine = st.checkbox("我不知道脊椎 T 分數")
if use_default_spine:
    your_spine_t = None
    st.info("已選擇：不輸入脊椎 T 分數")
else:
    your_spine_t = temp_spine

# HipR T 值輸入與選擇
temp_hipr = st.number_input("你輸入的右髖關節 T 分數", value=0.0, step=0.1)
use_default_hipr = st.checkbox("我不知道右髖關節 T 分數")
if use_default_hipr:
    your_hipr_t = None
    st.info("已選擇：不輸入右髖關節 T 分數")
else:
    your_hipr_t = temp_hipr

# HipL T 值輸入與選擇
temp_hipl = st.number_input("你輸入的左髖關節 T 分數", value=0.0, step=0.1)
use_default_hipl = st.checkbox("我不知道左髖關節 T 分數")
if use_default_hipl:
    your_hipl_t = None
    st.info("已選擇：不輸入左髖關節 T 分數")
else:
    your_hipl_t = temp_hipl




if st.button("預測 T 值並加入圖表"):
    # 預測三個部位
    pred_spine = predict_spine_t(AGE, SEX, BODY_HEIGHT, BODY_WEIGHT, QUS)
    pred_hipR = predict_hipR_t(BODY_FAT_RATE, AGE, SEX, BODY_HEIGHT, BODY_WEIGHT, QUS)
    pred_hipL = predict_hipL_t(AGE, SEX, BODY_WEIGHT, QUS)

    # 對應欄位與值
    regions = {
        "脊椎": ("Spine_Tot_T", pred_spine, your_spine_t),
        "右髖關節": ("HipR_Tot_T", pred_hipR, your_hipr_t),
        "左髖關節": ("HipL_Tot_T", pred_hipL, your_hipl_t)
    }

    # 三欄佈局顯示圖表
    st.markdown('RC:')
    cols = st.columns(3)
    for i, (region, (y_col, pred_bmd, user_input)) in enumerate(regions.items()):
        with cols[i]:
            fig, ax = plt.subplots()
            ax.scatter(df_exported['AGE'], df_exported[y_col], color='blue', s=4, alpha=0.3, label='原始資料')
            ax.scatter(AGE, pred_bmd, color='red', label='模型預測值', s=100)
            ax.scatter(AGE, user_input, color='black', label='你的輸入值', s=100)
            ax.set_xlabel("年齡")
            ax.set_ylabel("T 分數")
            ax.set_title(region)
            ax.legend()
            st.pyplot(fig)
            st.markdown(f"**{region}**  \n預測 T 分數：**{pred_bmd:.2f}**")

    # 1. 取最低
    min_t = min(pred_spine, pred_hipR, pred_hipL)

    # 2. 給結論
    if min_t >= -1.0:
        conclusion = "正常骨質"
        emoji = "🟢"
    elif min_t >= -2.5:
        conclusion = "骨質缺乏"
        emoji = "🟡"
    else:
        conclusion = "骨質疏鬆"
        emoji = "🔴"

    # 3. 顯示
    st.write(f"🔻 **最低 T 分數預測值**: {min_t:.2f}")
    if conclusion == "正常骨質":
        st.success(f"**預測結果**: {conclusion}{emoji}")
    elif conclusion == "骨質缺乏":
        st.warning(f"**預測結果**: {conclusion}{emoji}")
    else:
        st.error(f"**預測結果**: {conclusion}{emoji}")
    st.markdown("---")

    mi_row = {
    'Spine_Tot_T_adj':np.nan,
    'HipL_Tot_T_adj': np.nan,
    'HipR_Tot_T_adj': np.nan,
    'AGE': AGE,
    'SEX': 2 if SEX == "女性" else 1,
    'BODY_HEIGHT': BODY_HEIGHT,
    'BODY_WEIGHT': BODY_WEIGHT,
    'BODY_FAT_RATE': BODY_FAT_RATE,
    'QUS': QUS,
    'HBA1C': HBA1C
    }
    # ===== 脊椎插補（使用 diabetes_self == 0 的資料）=====
    features_spine = ['Spine_Tot_T_adj', 'AGE', 'SEX', 'BODY_HEIGHT', 'BODY_WEIGHT', 'QUS']
    df_spine = df[df['DIABETES_SELF'] == 0][features_spine].copy()
    df_spine = pd.concat([df_spine, pd.DataFrame([{k: mi_row[k] for k in features_spine}])], ignore_index=True)

    imputer_spine = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=10, sample_posterior=True)
    imputed_spine = imputer_spine.fit_transform(df_spine)
    mi_pred_spine = imputed_spine[-1][0]

    # ===== 髖部插補（使用 diabetes_self == 0 的資料）=====
    features_hip = ['HipL_Tot_T_adj', 'HipR_Tot_T_adj', 'AGE', 'SEX', 'BODY_HEIGHT', 'BODY_WEIGHT', 'BODY_FAT_RATE', 'QUS', 'HBA1C']
    df_hip = df[df['DIABETES_SELF'] == 0][features_hip].copy()
    df_hip = pd.concat([df_hip, pd.DataFrame([{k: mi_row[k] for k in features_hip}])], ignore_index=True)

    imputer_hip = IterativeImputer(estimator=BayesianRidge(), random_state=0, max_iter=10, sample_posterior=True)
    imputed_hip = imputer_hip.fit_transform(df_hip)
    mi_pred_hipL = imputed_hip[-1][0]
    mi_pred_hipR = imputed_hip[-1][1]

    # 對應欄位與值
    regions = {
        "脊椎": ("Spine_Tot_T", mi_pred_spine, your_spine_t),
        "右髖關節": ("HipR_Tot_T", mi_pred_hipR, your_hipr_t),
        "左髖關節": ("HipL_Tot_T", mi_pred_hipL, your_hipl_t)
    }

    # 三欄佈局顯示圖表
    st.markdown('MI:')
    cols = st.columns(3)
    for i, (region, (y_col, mi_pred_bmd, user_input)) in enumerate(regions.items()):
        with cols[i]:
            fig, ax = plt.subplots()
            ax.scatter(df_exported['AGE'], df_exported[y_col], color='blue', s=4, alpha=0.3, label='原始資料')
            ax.scatter(AGE, mi_pred_bmd, color='red', label='模型預測值', s=100)
            ax.scatter(AGE, user_input, color='black', label='你的輸入值', s=100)
            ax.set_xlabel("年齡")
            ax.set_ylabel("T 分數")
            ax.set_title(region)
            ax.legend()
            st.pyplot(fig)
            st.markdown(f"**{region}**  \n預測 T 分數：**{mi_pred_bmd:.2f}**")

    # 1. 取最低
    min_t = min(mi_pred_spine, mi_pred_hipR, mi_pred_hipL)

    # 2. 給結論
    if min_t >= -1.0:
        conclusion = "正常骨質"
        emoji = "🟢"
    elif min_t >= -2.5:
        conclusion = "骨質缺乏"
        emoji = "🟡"
    else:
        conclusion = "骨質疏鬆"
        emoji = "🔴"

    # 3. 顯示
    st.write(f"🔻 **最低 T 分數預測值**: {min_t:.2f}")
    if conclusion == "正常骨質":
        st.success(f"**預測結果**: {conclusion}{emoji}")
    elif conclusion == "骨質缺乏":
        st.warning(f"**預測結果**: {conclusion}{emoji}")
    else:
        st.error(f"**預測結果**: {conclusion}{emoji}")
    st.markdown("---")

    # 加上圖示 legend
    st.markdown("""
    #### 判斷標準說明
    - 🟢 **正常骨質**：最低 T 分數 ≥ -1.0  
    - 🟡 **骨質減少**：-2.5 < 最低 T 分數 < -1.0  
    - 🔴 **骨質疏鬆**：最低 T 分數 ≤ -2.5  
    """)