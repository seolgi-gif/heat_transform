import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import time

# --- 1. 한글 폰트 설정 (안정적인 방식) ---
try:
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic = next((f for f in font_path if 'NanumGothic' in f), None)
    if nanum_gothic:
        font_prop = fm.FontProperties(fname=nanum_gothic)
        plt.rc('font', family='NanumGothic')
    else:
        font_prop = fm.FontProperties(size=12)
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    st.warning("폰트 로딩 중 문제가 발생했습니다. 기본 폰트로 표시됩니다.")
    font_prop = fm.FontProperties(size=12)

# --- 2. 2D 열전달 시뮬레이션 함수 (NumPy 벡터화로 속도 개선) ---
@st.cache_data
def run_2d_heat_simulation(k, L_x, rho, cp=1000, T_hot=1000+273.15, T_initial=20+273.15, sim_time_minutes=15):
    sim_time_seconds = sim_time_minutes * 60
    L_y = 0.1
    alpha = k / (rho * cp)
    nx, ny = 50, 25
    dx = L_x / (nx - 1)
    dy = L_y / (ny - 1)
    dt = 0.2 * (1 / (alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None

    time_points = np.linspace(0, sim_time_seconds, nt)
    temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial
    TARGET_TEMP_KELVIN = 120 + 273.15
    time_to_target = None

    for t_step in range(nt):
        T_old = T.copy()
        
        # 경계 조건 적용
        T[:, 0] = T_hot; T[:, -1] = T[:, -2]; T[0, :] = T[1, :]; T[-1, :] = T[-2, :]
        
        # --- 핵심 개선: NumPy 벡터화를 통해 내부 온도 전체를 한 번에 계산 ---
        T_inner = T_old[1:-1, 1:-1]
        laplacian_x = (T_old[1:-1, 2:] - 2 * T_inner + T_old[1:-1, :-2]) / dx**2
        laplacian_y = (T_old[2:, 1:-1] - 2 * T_inner + T_old[:-2, 1:-1]) / dy**2
        T[1:-1, 1:-1] = T_inner + alpha * dt * (laplacian_x + laplacian_y)
        # --- 여기까지가 기존의 2중 for문을 대체합니다 ---

        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        if time_to_target is None and current_inner_temp_k >= TARGET_TEMP_KELVIN:
            time_to_target = time_points[t_step] / 60
            
    return time_points, temp_history_celsius, T - 273.15, time_to_target

# --- 3. 시나리오(재료) 정의 ---
scenarios = {
    '에어로겔': {'k': 0.02, 'rho': 80, 'cp': 1000, 'cost': 500},
    '고강도 경량 단열 타일': {'k': 0.06, 'rho': 145, 'cp': 1000, 'cost': 350},
    '세라믹 섬유': {'k': 0.1, 'rho': 150, 'cp': 1000, 'cost': 100},
    '알루미늄': {'k': 200.0, 'rho': 2700, 'cp': 900, 'cost': 20},
}

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("🌡️ 2D 열전달 시뮬레이션 및 최적화 분석")
st.markdown("외부 1000°C 환경에서 **15분** 동안, 재료의 **두께**에 따라 내부 온도가 어떻게 변하는지 관찰하고, 주어진 조건에 가장 적합한 재료를 분석합니다.")

st.sidebar.header("⚙️ 시뮬레이션 설정")
selected_material_name = st.sidebar.selectbox("1. 개별 재료 선택", options=list(scenarios.keys()))
thickness_mm = st.sidebar.slider("2. 재료 두께 (mm)", min_value=10.0, max_value=200.0, value=50.0, step=1.0)

thickness_m = thickness_mm / 1000.0
SIMULATION_TIME_MINUTES = 15

# --- 개별 시뮬레이션 섹션 ---
st.subheader(f"1. '{selected_material_name}' 개별 시뮬레이션")
material_props = scenarios[selected_material_name]
k = material_props['k']; rho = material_props['rho']; cp = material_props['cp']

if st.sidebar.button("🚀 개별 시뮬레이션 실행"):
    with st.spinner(f"'{selected_material_name}'(두께: {thickness_mm}mm) 시뮬레이션 중..."):
        time_pts, temp_hist, _, _ = run_2d_heat_simulation(k=k, L_x=thickness_m, rho=rho, cp=cp)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_pts / 60, temp_hist, label=f"{selected_material_name} ({thickness_mm}mm)", lw=2.5)
    ax.axhline(y=120, color='r', linestyle='--', label='목표 최대 온도 (120°C)')
    ax.set_title(f'내부 표면 온도 변화', fontproperties=font_prop, fontsize=16)
    ax.set_xlabel('시간 (분)', fontproperties=font_prop); ax.set_ylabel('평균 온도 (°C)', fontproperties=font_prop)
    ax.legend(prop=font_prop); ax.grid(True, linestyle=':'); ax.set_xlim(0, SIMULATION_TIME_MINUTES)
    ax.set_ylim(15, max(150, max(temp_hist) * 1.2))
    st.pyplot(fig)

st.divider()

# --- 최적화 분석 섹션 (진행 상황 바 추가) ---
st.subheader(f"2. 전 재료 최적화 분석 (두께: {thickness_mm}mm)")
if st.button("📊 최적화 분석 실행"):
    results = []
    materials_to_run = list(scenarios.items())
    
    # 진행 상황 바 초기화
    progress_bar = st.progress(0, text="분석 시작...")
    
    for i, (name, props) in enumerate(materials_to_run):
        # 진행 상황 바 텍스트 업데이트
        progress_bar.text(f"({i+1}/{len(materials_to_run)}) '{name}' 시뮬레이션 중...")
        
        _, temp_hist, _, _ = run_2d_heat_simulation(
            k=props['k'], L_x=thickness_m, rho=props['rho'], cp=props['cp']
        )
        if temp_hist is not None:
            final_temp = temp_hist[-1]
            results.append({'name': name, 'final_temp': final_temp, **props})
        
        # 진행 상황 바 업데이트
        progress_bar.progress((i + 1) / len(materials_to_run))
        
    progress_bar.text("분석 완료!")
    time.sleep(1) # 완료 메시지를 보여주기 위해 1초 대기
    progress_bar.empty() # 진행 상황 바 숨기기

    passed_scenarios = [r for r in results if r['final_temp'] < 120]

    if not passed_scenarios:
        st.warning(f"두께 {thickness_mm}mm 조건에서는 120°C 목표를 만족하는 재료가 없습니다. 두께를 늘려보세요.")
    else:
        for r in passed_scenarios:
            safety_margin = 120 - r['final_temp']
            r['perf_per_thickness'] = safety_margin / thickness_m
            r['perf_per_weight'] = safety_margin / (thickness_m * r['rho'])
            r['perf_per_cost'] = safety_margin / r['cost']

        best_performance = min(passed_scenarios, key=lambda x: x['final_temp'])
        best_thickness_eff = max(passed_scenarios, key=lambda x: x['perf_per_thickness'])
        best_weight_eff = max(passed_scenarios, key=lambda x: x['perf_per_weight'])
        best_cost_eff = max(passed_scenarios, key=lambda x: x['perf_per_cost'])

        st.markdown("#### ✨ 최적 재료 추천")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🥇 절대 성능", best_performance['name'], f"{best_performance['final_temp']:.1f} °C")
        with col2:
            st.metric("🚀 두께 효율", best_thickness_eff['name'], "소형화 최적")
        with col3:
            st.metric("🕊️ 중량 효율", best_weight_eff['name'], "경량화 최적")
        with col4:
            st.metric("💰 비용 효율", best_cost_eff['name'], "가성비 최적")

        with st.expander("자세한 분석 결과 보기"):
            df = pd.DataFrame(results)
            df['최종 온도 (°C)'] = df['final_temp'].round(1)
            df_display = df[['name', '최종 온도 (°C)', 'k', 'rho', 'cost']]
            df_display = df_display.rename(columns={'name':'재료', 'k':'열전도율', 'rho':'밀도', 'cost':'상대 비용'})
            st.dataframe(df_display, use_container_width=True, hide_index=True)
