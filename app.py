import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

# --- 1. 한글 폰트 설정 ---
@st.cache_data
def font_setup():
    """Streamlit 환경에 맞는 한글 폰트를 설정합니다."""
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    nanum_gothic_files = [f for f in font_files if 'NanumGothic' in f]
    
    if nanum_gothic_files:
        plt.rc('font', family='NanumGothic')
        font_prop = fm.FontProperties(fname=nanum_gothic_files[0])
    else:
        st.warning("나눔고딕 폰트를 찾을 수 없습니다. 기본 폰트로 표시되며 글자가 깨질 수 있습니다.")
        font_prop = fm.FontProperties(size=12)
        
    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

font_prop = font_setup()


# --- 2. 시나리오(재료) 정의 ---
SCENARIOS = {
    '에어로겔': {'k': 0.02, 'rho': 80, 'cp': 1000},
    '세라믹 섬유': {'k': 0.1, 'rho': 150, 'cp': 1000},
    'PCM (고체상태)': {'k': 0.25, 'rho': 900, 'cp': 2100},
    '강철 (Steel)': {'k': 50.0, 'rho': 7850, 'cp': 490},
    '알루미늄': {'k': 200.0, 'rho': 2700, 'cp': 900},
}

MATERIALS_DB_DETAILED = {
    "Aerogel": {"rho": 150, "k": lambda T_K: 0.02 + 5e-5 * (T_K - 273.15), "cp": lambda T_K: 1000 + 0.5 * (T_K - 273.15)},
    "Ceramic_Fiber": {"rho": 2500, "k": lambda T_K: 1.5 + 2e-4 * (T_K - 273.15), "cp": lambda T_K: 800 + 0.4 * (T_K - 273.15)},
    "PCM": {
        "rho": 800, "k": lambda T_K, T_melt: np.where(T_K < T_melt, 0.22, 0.18),
        "cp": lambda T_K, T_melt: 2000 if T_K < T_melt else 2200,
        "T_melt_start_C": 140.0, "T_melt_end_C": 160.0, "L_h": 250000,
    }
}
NAME_MAP = {'에어로겔': 'Aerogel', '세라믹 섬유': 'Ceramic_Fiber', 'PCM (고체상태)': 'PCM'}


# --- 3. 시뮬레이션 함수 (이전과 동일) ---
@st.cache_data
def run_multilayer_simulation(materials, thicknesses_m, material_names=None, T_hot_c=1000, T_initial_c=20, T_target_c=120, sim_time_minutes=15, stop_at_target=False):
    T_hot = T_hot_c + 273.15; T_initial = T_initial_c + 273.15; T_target_kelvin = T_target_c + 273.15
    sim_time_seconds = sim_time_minutes * 60
    L_x = sum(thicknesses_m)
    if L_x == 0: return None, None, None, None
    L_y = 0.1; nx, ny = 60, 6; dx = L_x / (nx - 1); dy = L_y / (ny - 1)
    alphas = [mat['k'] / (mat['rho'] * mat['cp']) for mat in materials]
    alpha_map = np.zeros(nx); current_pos_m = 0; start_idx = 0
    for i, thick_m in enumerate(thicknesses_m):
        current_pos_m += thick_m
        end_idx = int(current_pos_m / L_x * (nx - 1))
        alpha_map[start_idx : end_idx + 1] = alphas[i]
        start_idx = end_idx
    max_alpha = max(alphas); dt = 0.2 * (1 / (max_alpha * (1/dx**2 + 1/dy**2)))
    if dt > 0.5: dt = 0.5
    nt = int(sim_time_seconds / dt)
    if nt <= 0: return None, None, None, None
    time_points = np.linspace(0, sim_time_seconds, nt); temp_history_celsius = np.zeros(nt)
    T = np.ones((ny, nx)) * T_initial; time_to_target = None
    for t_step in range(nt):
        T_old = T.copy()
        laplacian_x = (T_old[1:-1, 2:] - 2 * T_old[1:-1, 1:-1] + T_old[1:-1, :-2]) / dx**2
        laplacian_y = (T_old[2:, 1:-1] - 2 * T_old[1:-1, 1:-1] + T_old[:-2, 1:-1]) / dy**2
        alpha_slice = alpha_map[1:-1]
        change_in_T = alpha_slice * dt * (laplacian_x + laplacian_y)
        T[1:-1, 1:-1] = T_old[1:-1, 1:-1] + change_in_T
        T[:, 0] = T_hot; T[:, -1] = T[:, -2]; T[0, :] = T[1, :]; T[-1, :] = T[-2, :]
        current_inner_temp_k = np.mean(T[:, -1])
        temp_history_celsius[t_step] = current_inner_temp_k - 273.15
        if time_to_target is None and current_inner_temp_k >= T_target_kelvin:
            time_to_target = time_points[t_step] / 60
            if stop_at_target:
                return time_points[:t_step+1], temp_history_celsius[:t_step+1], T - 273.15, time_to_target
    return time_points, temp_history_celsius, T - 273.15, time_to_target

def temp_to_enthalpy(T_C, material, mat_props):
    T_ref_C = 0.0
    if material != "PCM":
        T_avg_K = (T_C + T_ref_C) / 2.0 + 273.15; cp_avg = mat_props["cp"](T_avg_K)
        return mat_props["rho"] * cp_avg * (T_C - T_ref_C)
    else:
        cp_s = mat_props["cp"](mat_props["T_melt_start_C"]-10+273.15,0); cp_l = mat_props["cp"](mat_props["T_melt_end_C"]+10+273.15,0)
        T_ms_C, T_me_C, L_h, rho = mat_props["T_melt_start_C"], mat_props["T_melt_end_C"], mat_props["L_h"], mat_props["rho"]
        if T_C < T_ms_C: return rho * cp_s * (T_C - T_ref_C)
        elif T_C < T_me_C: return rho * cp_s * (T_ms_C-T_ref_C) + rho*L_h*((T_C-T_ms_C)/(T_me_C-T_ms_C))
        else: return rho * cp_s * (T_ms_C-T_ref_C) + rho*L_h + rho*cp_l*(T_C-T_me_C)
def enthalpy_to_temp(H, material, mat_props):
    T_ref_C = 0.0
    if material != "PCM":
        T_guess_K = 300; cp_avg = mat_props["cp"](T_guess_K)
        return T_ref_C + H / (mat_props["rho"] * cp_avg)
    else:
        cp_s = mat_props["cp"](mat_props["T_melt_start_C"]-10+273.15,0); cp_l = mat_props["cp"](mat_props["T_melt_end_C"]+10+273.15,0)
        T_ms_C, T_me_C, L_h, rho = mat_props["T_melt_start_C"], mat_props["T_melt_end_C"], mat_props["L_h"], mat_props["rho"]
        H_solid_max = rho * cp_s * (T_ms_C-T_ref_C); H_liquid_min = H_solid_max + rho*L_h
        if H < H_solid_max: return T_ref_C + H / (rho*cp_s)
        elif H < H_liquid_min: return T_ms_C + ((H-H_solid_max)/(rho*L_h))*(T_me_C-T_ms_C)
        else: return T_me_C + (H-H_liquid_min)/(rho*cp_l)

@st.cache_data
def run_detailed_single_material_simulation(material_name, total_thickness_mm):
    NX, NY = 51, 51; LX = total_thickness_mm / 1000.0; LY = 0.1
    dx = LX / (NX - 1); dy = LY / (NY - 1)
    INITIAL_TEMP_C = 25.0; TOTAL_SIM_TIME = 300; HEAT_SOURCE_TEMP_C = 800.0
    T_AMBIENT_C = 25.0; h_conv = 10.0; epsilon = 0.8; SIGMA = 5.67e-8
    mat_props = MATERIALS_DB_DETAILED[material_name]
    T_C = np.full((NY, NX), INITIAL_TEMP_C)
    H = np.full((NY, NX), temp_to_enthalpy(INITIAL_TEMP_C, material_name, mat_props))
    k_hot = mat_props['k'](1200, mat_props.get('T_melt_start_C',0)+273.15) if material_name == "PCM" else mat_props['k'](1200)
    cp_hot = mat_props['cp'](1200, mat_props.get('T_melt_start_C',0)+273.15) if material_name == "PCM" else mat_props['cp'](1200)
    est_alpha = k_hot / (mat_props['rho'] * cp_hot); dt = 0.2 * (dx**2) / (2 * est_alpha)
    n_steps = int(TOTAL_SIM_TIME / dt)
    for step in range(1, n_steps + 1):
        T_K = T_C + 273.15
        k_val_func = mat_props['k']
        k = k_val_func(T_K, mat_props.get('T_melt_start_C',0)+273.15) if material_name == "PCM" else k_val_func(T_K)
        H[:, 0] = temp_to_enthalpy(HEAT_SOURCE_TEMP_C, material_name, mat_props)
        T_ambient_K = T_AMBIENT_C + 273.15
        q_out_right = h_conv*(T_K[:,-1] - T_ambient_K) + epsilon*SIGMA*(T_K[:,-1]**4 - T_ambient_K**4); H[:,-1] -= (q_out_right / dx) * dt
        q_out_top = h_conv*(T_K[-1,:] - T_ambient_K) + epsilon*SIGMA*(T_K[-1,:]**4 - T_ambient_K**4); H[-1,:] -= (q_out_top / dy) * dt
        q_out_bottom = h_conv*(T_K[0,:] - T_ambient_K) + epsilon*SIGMA*(T_K[0,:]**4 - T_ambient_K**4); H[0,:] -= (q_out_bottom / dy) * dt
        k_x_interface = 0.5 * (k[:, 1:] + k[:, :-1]); k_y_interface = 0.5 * (k[1:, :] + k[:-1, :])
        q_x = -k_x_interface * (T_C[:, 1:] - T_C[:, :-1]) / dx; q_y = -k_y_interface * (T_C[1:, :] - T_C[:-1, :]) / dy
        H[1:-1, 1:-1] -= (((q_x[1:-1, 1:] - q_x[1:-1, :-1])/dx) + ((q_y[1:, 1:-1] - q_y[:-1, 1:-1])/dy)) * dt
        T_C = np.array([[enthalpy_to_temp(H[i,j], material_name, mat_props) for j in range(NX)] for i in range(NY)])
    return T_C, TOTAL_SIM_TIME, INITIAL_TEMP_C, HEAT_SOURCE_TEMP_C, LX, LY

# --- 4. Streamlit UI 구성 ---
st.set_page_config(layout="wide")
st.title("🚗 자동차 배터리 열차폐 시스템 설계 시뮬레이션")
st.markdown("이 앱은 **다층(Multi-layer) 구조**의 열 차폐 성능을 분석하고, 단일 재료와 비교하여 최적의 설계를 찾는 데 도움을 줍니다.")

st.sidebar.header("⚙️ 1. 기본 조건 설정")
max_thickness_mm = st.sidebar.number_input("최대 허용 두께 (mm)", 5.0, 100.0, 50.0, 1.0)
target_delay_min = st.sidebar.number_input("목표 지연 시간 (분)", 1.0, 30.0, 5.0, 0.5)

# --- 1단계 ---
st.header("📊 1단계: 단일 재료 성능 분석")
st.markdown(f"각 재료를 **{max_thickness_mm}mm** 두께로 단독 사용했을 때의 기본 성능과 열 확산 특성을 확인합니다.")

if 'single_analysis_done' not in st.session_state:
    st.session_state.single_analysis_done = False

if st.button("단일 재료 분석 시작"):
    results = []
    st.info("각 재료의 성능을 분석 중입니다...")
    progress_bar = st.progress(0, text="분석 시작...")
    sorted_scenarios = sorted(SCENARIOS.items(), key=lambda item: item[1]['k'])
    for i, (name, props) in enumerate(sorted_scenarios):
        progress_bar.progress((i + 1) / len(SCENARIOS), text=f"분석 중: {name}")
        _, _, _, time_to_target = run_multilayer_simulation(
            materials=[props], thicknesses_m=[max_thickness_mm / 1000.0], material_names=[name],
            sim_time_minutes=target_delay_min * 3, stop_at_target=True
        )
        delay_str = f"{time_to_target:.2f} 분" if time_to_target else f"{target_delay_min * 3}분 이상"
        is_success = time_to_target is None or time_to_target >= target_delay_min
        results.append({"재료": name, "120°C 도달 시간": delay_str, f"목표({target_delay_min}분) 달성": "✅" if is_success else "❌"})
    progress_bar.empty()
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.subheader(f"주요 단열재 300초 후 온도 분포 비교 (두께: {max_thickness_mm}mm)")
    materials_for_heatmap_ui = ['에어로겔', '세라믹 섬유', 'PCM (고체상태)']
    cols = st.columns(len(materials_for_heatmap_ui))
    for i, ui_name in enumerate(materials_for_heatmap_ui):
        with cols[i]:
            with st.spinner(f"'{ui_name}' 히트맵 생성 중..."):
                db_name = NAME_MAP[ui_name]
                final_map, sim_time, initial_temp, heat_source_temp, lx, ly = run_detailed_single_material_simulation(db_name, max_thickness_mm)
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(final_map, cmap='inferno', vmin=initial_temp, vmax=heat_source_temp, extent=[0, lx*100, 0, ly*100], origin='lower')
                ax.set_title(ui_name, fontproperties=font_prop)
                ax.set_xlabel('X-position (cm)', fontproperties=font_prop)
                if i == 0: ax.set_ylabel('Y-position (cm)', fontproperties=font_prop)
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Temperature (°C)', fontproperties=font_prop)
                st.pyplot(fig)
    st.session_state.single_analysis_done = True
    st.session_state.recommended_materials = ['세라믹 섬유', 'PCM (고체상태)', '에어로겔']

# --- 2단계 (새로 추가) ---
if st.session_state.single_analysis_done:
    st.header("💡 2단계: 최적 조합 추천")
    st.markdown("""
    1단계 분석 결과와 열 차폐 원리를 바탕으로 가장 효율적인 다층 구조 조합을 추천합니다.
    - **Layer 1 (외부)**: 고온의 열원에 직접 노출되므로, 내열성이 강한 **세라믹 섬유**가 적합합니다.
    - **Layer 2 (중간)**: 상변화물질(**PCM**)을 배치하여 녹는 과정에서 많은 열(잠열)을 흡수시켜 온도 상승을 효과적으로 지연시킵니다.
    - **Layer 3 (내부)**: 최종적으로 배터리를 보호하기 위해, 단열 성능이 가장 뛰어난 **에어로겔**을 사용합니다.
    """)
    recommended_str = " -> ".join(st.session_state.recommended_materials)
    st.success(f"**추천 조합 (외부 -> 내부):** {recommended_str}")

# --- 3단계 (기존 2단계) ---
st.header("🛠️ 3단계: 다층 구조 설계 및 성능 비교")
if not st.session_state.single_analysis_done:
    st.info("먼저 1단계 분석을 실행하여 각 재료의 기본 성능을 확인하세요.")
else:
    st.markdown("2단계에서 추천된 조합을 바탕으로 두께를 조절하며 성능을 확인하거나, 직접 새로운 조합을 만들어보세요.")
    
    material_options = list(SCENARIOS.keys())
    # 2단계의 추천 조합을 기본값으로 사용
    default_selection = st.session_state.get('recommended_materials', [])
    selected_materials = st.multiselect("3개의 재료를 선택하세요 (외부 -> 내부 순서)", 
                                      material_options, 
                                      default=default_selection, 
                                      max_selections=3)

    if len(selected_materials) == 3:
        st.subheader("두께 분배")
        cols = st.columns(3)
        thicknesses = []
        for i, mat_name in enumerate(selected_materials):
            with cols[i]:
                thicknesses.append(st.slider(f"Layer {i+1}: {mat_name} (mm)", 0.0, max_thickness_mm, max_thickness_mm / 3, 0.5, key=f"thick_{i}_{mat_name}"))

        total_selected_thickness = sum(thicknesses)
        if total_selected_thickness > max_thickness_mm:
            st.error(f"선택한 두께의 총합({total_selected_thickness:.1f}mm)이 최대 허용 두께({max_thickness_mm}mm)를 초과했습니다.")
        else:
            st.info(f"현재 총 두께: {total_selected_thickness:.1f} mm / {max_thickness_mm} mm")

        if st.button("다층 구조 시뮬레이션 및 성능 비교", key="run_multilayer"):
            if total_selected_thickness <= 0:
                st.error("두께를 0보다 크게 설정해야 시뮬레이션이 가능합니다.")
            else:
                with st.spinner("다층 구조 및 비교군(단일 구조) 시뮬레이션을 진행 중입니다..."):
                    materials_multi = [SCENARIOS[name] for name in selected_materials]
                    thicknesses_multi_m = [t / 1000.0 for t in thicknesses]
                    time_pts_multi, temp_hist_multi, _, time_to_target_multi = run_multilayer_simulation(
                        materials=materials_multi, thicknesses_m=thicknesses_multi_m, material_names=selected_materials,
                        sim_time_minutes=target_delay_min * 2
                    )
                    comparison_results = {}
                    for name in selected_materials:
                        time_pts_single, temp_hist_single, _, time_to_target_single = run_multilayer_simulation(
                            materials=[SCENARIOS[name]], thicknesses_m=[total_selected_thickness / 1000.0], material_names=[f"single_{name}"],
                            sim_time_minutes=target_delay_min * 2
                        )
                        comparison_results[name] = {"time_pts": time_pts_single, "temp_hist": temp_hist_single, "delay": time_to_target_single}

                st.subheader("🚀 시뮬레이션 결과")
                st.markdown("##### 성능 요약")
                delay_multi = time_to_target_multi if time_to_target_multi is not None else (target_delay_min * 2)
                best_single_name = ""
                best_single_delay = -1
                for name, result in comparison_results.items():
                    current_delay = result['delay'] if result['delay'] is not None else (target_delay_min * 2)
                    if current_delay > best_single_delay:
                        best_single_delay = current_delay
                        best_single_name = name
                col1, col2, col3 = st.columns(3)
                col1.metric("다층 구조 지연 시간", f"{delay_multi:.2f} 분")
                col2.metric(f"최고 성능 단일 구조 ({best_single_name})", f"{best_single_delay:.2f} 분")
                if delay_multi > best_single_delay:
                    improvement = delay_multi - best_single_delay
                    col3.metric("성능 향상", f"✅ +{improvement:.2f} 분", help="다층 구조가 가장 좋은 단일 구조보다 지연 시간이 더 깁니다.")
                else:
                    decline = best_single_delay - delay_multi
                    col3.metric("성능 저하", f"❌ -{decline:.2f} 분", help="다층 구조가 가장 좋은 단일 구조보다 성능이 낮습니다. 조합을 재고하세요.")

                st.markdown("##### 온도 변화 그래프")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(time_pts_multi / 60, temp_hist_multi, label=f"다층 구조 ({total_selected_thickness:.1f}mm)", lw=3, color='crimson')
                for name, result in comparison_results.items():
                    ax.plot(result['time_pts'] / 60, result['temp_hist'], label=f"{name} 단일 ({total_selected_thickness:.1f}mm)", linestyle='--', alpha=0.8)
                ax.axhline(y=120, color='k', linestyle=':', label='목표 최대 온도 (120°C)')
                ax.axvline(x=target_delay_min, color='g', linestyle=':', label=f'목표 지연 시간 ({target_delay_min}분)')
                ax.set_title('다층 구조 vs 단일 구조 성능 비교', fontproperties=font_prop, fontsize=16)
                ax.set_xlabel('시간 (분)', fontproperties=font_prop)
                ax.set_ylabel('온도 (°C)', fontproperties=font_prop)
                ax.legend(prop=font_prop, loc='best'); ax.grid(True, linestyle=':'); ax.set_xlim(0, target_delay_min * 2)
                ax.set_ylim(15, max(150, np.max(temp_hist_multi) * 1.2) if len(temp_hist_multi) > 0 else 150)
                st.pyplot(fig)
    else:
        st.warning("먼저 3개의 재료를 선택해주세요.")
