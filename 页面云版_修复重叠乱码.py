import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from html import escape
from pathlib import Path

# Model Library
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("请先在终端安装 XGBoost：pip install xgboost")
    st.stop()
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import shap

# ==========================================
# Page & Global Font Configuration
# ==========================================
st.set_page_config(page_title="矿山尘肺病智能预警系统", layout="wide")

st.markdown(
    """
<style>
    :root {
        --bg-panel: rgba(10, 29, 54, 0.84);
        --line: rgba(66, 178, 255, 0.32);
        --line-strong: rgba(80, 205, 255, 0.62);
        --text-main: #eaf6ff;
        --text-muted: #91b6cf;
        --blue: #1e9bff;
        --cyan: #37e4ff;
        --green: #35d07f;
        --yellow: #f2c94c;
        --orange: #ff8a3d;
        --red: #ff4f5e;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        font-family: "Microsoft YaHei", "SimHei", Arial, Helvetica, sans-serif;
        color: var(--text-main);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 15% 8%, rgba(39, 159, 255, 0.26), transparent 28%),
            radial-gradient(circle at 82% 10%, rgba(55, 228, 255, 0.18), transparent 30%),
            linear-gradient(135deg, #06111f 0%, #0a1e38 52%, #06111f 100%);
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(80, 205, 255, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(80, 205, 255, 0.06) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.75), transparent 78%);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span,
    div[data-testid="stMarkdownContainer"],
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSlider"] label p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    input[type="number"], input, textarea, select {
        font-family: "Microsoft YaHei", "SimHei", Arial, Helvetica, sans-serif !important;
    }

    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stSlider"] label p {
        color: var(--text-main) !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    input[type="number"], div[data-baseweb="select"] > div {
        background: rgba(6, 20, 39, 0.88) !important;
        color: var(--text-main) !important;
        border: 1px solid rgba(80, 205, 255, 0.28) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 0 18px rgba(30, 155, 255, 0.08);
    }

    input[type="number"] {
        font-size: 19px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stNumberInput"] button {
        color: var(--text-main) !important;
        background: rgba(27, 78, 126, 0.5) !important;
        border: 1px solid rgba(80, 205, 255, 0.18) !important;
    }

    div[data-testid="stButton"] button {
        height: 46px;
        border-radius: 8px;
        border: 1px solid rgba(92, 217, 255, 0.85);
        background: linear-gradient(90deg, #0f7cff, #20d4ff) !important;
        color: #031120 !important;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0;
        box-shadow: 0 0 24px rgba(32, 212, 255, 0.35);
    }

    .page-title {
        position: relative;
        padding: 20px 24px 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(80, 205, 255, 0.38);
        border-radius: 10px;
        background: linear-gradient(90deg, rgba(10, 31, 58, 0.88), rgba(6, 21, 39, 0.62));
        box-shadow: 0 0 30px rgba(30, 155, 255, 0.18);
        overflow: hidden;
    }

    .page-title::after {
        content: "";
        position: absolute;
        left: 18px;
        right: 18px;
        bottom: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    }

    .page-title h1 {
        margin: 0;
        font-size: 42px;
        line-height: 1.08;
        font-weight: 800;
        color: #f3fbff;
        text-shadow: 0 0 18px rgba(55, 228, 255, 0.45);
        white-space: nowrap;
    }

    .section-card {
        padding: 20px 22px;
        margin: 14px 0 18px;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--bg-panel);
        box-shadow: 0 18px 46px rgba(0, 0, 0, 0.25), inset 0 0 28px rgba(30, 155, 255, 0.05);
    }

    .section-title {
        margin: 0 0 16px;
        color: #f1fbff;
        font-size: 24px;
        font-weight: 800;
    }

    .hint-box {
        padding: 13px 15px;
        border: 1px solid rgba(80, 205, 255, 0.26);
        border-left: 3px solid var(--cyan);
        border-radius: 8px;
        background: rgba(9, 35, 64, 0.7);
        color: var(--text-muted);
        line-height: 1.65;
        min-height: 82px;
    }

    .result-panel {
        padding: 20px 22px;
        border: 1px solid var(--line-strong);
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(11, 37, 68, 0.94), rgba(6, 20, 39, 0.88));
        box-shadow: 0 0 34px rgba(30, 155, 255, 0.22);
    }

    .risk-label {
        color: var(--text-muted);
        font-size: 14px;
        margin-bottom: 8px;
    }

    .risk-level {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 34px;
        font-weight: 900;
        margin: 0;
    }

    .risk-dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        box-shadow: 0 0 20px currentColor;
        background: currentColor;
    }

    .risk-value {
        font-size: 42px;
        line-height: 1;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 16px rgba(55, 228, 255, 0.25);
    }

    .gauge-track {
        height: 12px;
        border-radius: 999px;
        background: rgba(145, 182, 207, 0.18);
        overflow: hidden;
        margin-top: 18px;
    }

    .gauge-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--green), var(--yellow), var(--orange), var(--red));
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }

    .mini-metric {
        padding: 13px 14px;
        border: 1px solid rgba(80, 205, 255, 0.24);
        border-radius: 8px;
        background: rgba(9, 30, 55, 0.72);
    }

    .mini-metric .label {
        color: var(--text-muted);
        font-size: 13px;
        margin-bottom: 7px;
    }

    .mini-metric .value {
        color: #ffffff;
        font-size: 22px;
        font-weight: 850;
    }

    .advice-list {
        margin: 0;
        padding-left: 1.2rem;
        color: var(--text-main);
        line-height: 1.78;
    }

    .history-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-main);
        font-size: 14px;
    }

    .history-table th,
    .history-table td {
        border-bottom: 1px solid rgba(80, 205, 255, 0.18);
        padding: 10px 8px;
        text-align: left;
    }

    .history-table th {
        color: var(--cyan);
        font-weight: 800;
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: 0.95fr 1.25fr 1fr;
        gap: 14px;
        align-items: stretch;
    }

    .dash-panel {
        min-height: 210px;
        padding: 16px;
        border: 1px solid rgba(80, 205, 255, 0.24);
        border-radius: 8px;
        background: rgba(7, 24, 46, 0.76);
        box-shadow: inset 0 0 24px rgba(30, 155, 255, 0.06);
    }

    .dash-title {
        color: #f1fbff;
        font-size: 17px;
        font-weight: 850;
        margin-bottom: 12px;
        padding-bottom: 9px;
        border-bottom: 1px solid rgba(80, 205, 255, 0.18);
    }

    .profile-line {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 9px 0;
        border-bottom: 1px dashed rgba(145, 182, 207, 0.18);
        color: var(--text-muted);
        font-size: 14px;
    }

    .profile-line strong {
        color: var(--text-main);
        font-weight: 800;
        text-align: right;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 62px;
        padding: 4px 9px;
        border-radius: 999px;
        color: #041320;
        font-size: 13px;
        font-weight: 850;
    }

    .bar-row {
        display: grid;
        grid-template-columns: 84px 1fr 54px;
        gap: 10px;
        align-items: center;
        margin: 12px 0;
        color: var(--text-muted);
        font-size: 13px;
    }

    .bar-track {
        height: 8px;
        border-radius: 999px;
        background: rgba(145, 182, 207, 0.16);
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #1e9bff, #37e4ff);
    }

    .tunnel-map {
        position: relative;
        height: 210px;
        border: 1px solid rgba(80, 205, 255, 0.22);
        border-radius: 8px;
        background:
            linear-gradient(90deg, rgba(55, 228, 255, 0.18) 1px, transparent 1px),
            linear-gradient(rgba(55, 228, 255, 0.1) 1px, transparent 1px),
            rgba(4, 15, 30, 0.7);
        background-size: 38px 38px;
        overflow: hidden;
    }

    .tunnel-line {
        position: absolute;
        height: 2px;
        background: rgba(55, 228, 255, 0.55);
        box-shadow: 0 0 12px rgba(55, 228, 255, 0.45);
    }

    .tunnel-node {
        position: absolute;
        width: 16px;
        height: 16px;
        margin: -8px 0 0 -8px;
        border-radius: 50%;
        border: 2px solid rgba(255, 255, 255, 0.72);
        box-shadow: 0 0 18px currentColor;
        background: currentColor;
    }

    .node-label {
        position: absolute;
        transform: translate(-50%, 14px);
        color: var(--text-muted);
        font-size: 12px;
        white-space: nowrap;
    }

    .alarm-list {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .alarm-item {
        padding: 10px 12px;
        border-left: 3px solid currentColor;
        border-radius: 7px;
        background: rgba(9, 35, 64, 0.66);
        color: var(--text-main);
        line-height: 1.55;
        font-size: 14px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(80, 205, 255, 0.26) !important;
        background: rgba(8, 25, 47, 0.74) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary p {
        color: var(--text-main) !important;
        font-weight: 800 !important;
    }

    [data-testid="stAlert"] {
        background: rgba(9, 35, 64, 0.8);
        color: var(--text-main);
        border-color: rgba(80, 205, 255, 0.28);
    }

    /* 保留 Streamlit 图标字体，避免菜单和展开箭头变成文字 */
    .material-icons,
    .material-icons-round,
    .material-icons-outlined,
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: 'Material Icons', 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    }

    /* 公共页面可隐藏右上角工具栏，避免移动端/窄屏设置菜单重叠 */
    [data-testid="stToolbar"] {
        display: none;
    }

    @media (max-width: 900px) {
        .page-title h1 { font-size: 30px; white-space: normal; }
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .dashboard-grid { grid-template-columns: 1fr; }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="page-title">
    <h1>矿山尘肺病智能预警系统</h1>
</div>
""",
    unsafe_allow_html=True,
)

COLOR_BLUE = "#316395"
COLOR_RED = "#B82E2E"


# ==========================================
# 工具函数：渲染带 CSS 颜色劫持的 JS 力图
# ==========================================
def st_shap(plot, height=320, min_width=1800):
    shap_html = f"""
    <head>
        {shap.getjs()}
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: white;
                font-family: "Microsoft YaHei", "SimHei", Arial, Helvetica, sans-serif !important;
            }}
            .shap-wrap {{
                width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
                padding: 6px 0 0 0;
                background: white;
            }}
            .shap-inner {{
                min-width: {min_width}px;
                width: {min_width}px;
                padding-right: 12px;
            }}
            .shap-inner svg {{
                overflow: visible;
            }}
            body, div, span, text, g {{
                font-family: "Microsoft YaHei", "SimHei", Arial, Helvetica, sans-serif !important;
            }}
            path[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[stroke="#ff0052"] {{ stroke: {COLOR_RED} !important; }}
            text[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
            path[stroke="#008bfb"] {{ stroke: {COLOR_BLUE} !important; }}
            text[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
        </style>
    </head>
    <body>
        <div class="shap-wrap">
            <div class="shap-inner">{plot.html()}</div>
        </div>
    </body>
    """
    components.html(shap_html, height=height, scrolling=False)


def get_risk_info(prediction):
    if prediction <= 5:
        return {
            "level": "低风险",
            "color": "#35d07f",
            "summary": "当前模型预测风险较低，建议维持现有防护并持续监测。",
        }
    if prediction <= 12:
        return {
            "level": "中风险",
            "color": "#f2c94c",
            "summary": "当前存在一定职业暴露风险，建议加强粉尘控制和个体防护。",
        }
    if prediction <= 20:
        return {
            "level": "高风险",
            "color": "#ff8a3d",
            "summary": "当前风险偏高，应尽快排查高暴露环节并优化作业组织。",
        }
    return {
        "level": "极高风险",
        "color": "#ff4f5e",
        "summary": "当前风险处于警戒区间，建议立即开展防尘整改与职业健康复核。",
    }


def get_job_tip(job_key):
    tips = {
        "blasting": "爆破岗位粉尘峰值波动大，应重点关注爆破前后湿式作业、通风排尘和人员撤离时间。",
        "transport": "运输岗位容易出现转载点二次扬尘，应重点关注封闭转载、喷雾降尘和巷道积尘清理。",
        "extract": "采掘岗位接触粉尘时间长，应重点关注采煤机内外喷雾、煤层注水和局部通风效果。",
        "support": "支护岗位可能涉及钻孔、喷浆等高尘环节，应重点关注湿式钻孔和喷浆除尘。",
        "repair": "检修岗位常进入高尘残留区域，应重点关注作业前冲洗、短时暴露控制和便携式防护。",
        "other": "其他岗位应结合现场粉尘来源识别暴露环节，重点落实个人防护和定期监测。",
    }
    return tips.get(job_key, tips["other"])


def build_protection_advice(risk_info, prediction, data, job_key):
    advice = []
    if risk_info["level"] == "低风险":
        advice.append("维持现有通风、喷雾降尘和个人防护制度，保持岗位粉尘浓度定期检测。")
        advice.append("建议按职业健康管理要求进行周期性体检，并保存历次检测结果用于趋势对比。")
    elif risk_info["level"] == "中风险":
        advice.append("加强作业点通风与湿式降尘，重点复核粉尘浓度较高的班次和工序。")
        advice.append("提高防尘口罩佩戴规范性，检查密合性、滤棉更换频率和现场监督记录。")
    elif risk_info["level"] == "高风险":
        advice.append("优先降低粉尘峰值和时间加权平均浓度，必要时缩短连续暴露时间并安排轮岗。")
        advice.append("建议增加职业健康体检频次，对长期接尘人员建立重点随访台账。")
    else:
        advice.append("建议立即开展现场防尘专项排查，暂停或限制高尘作业直至通风除尘措施达标。")
        advice.append("对相关人员进行职业健康复核，必要时调整至低尘岗位并开展医学随访。")

    if data["max"] >= 8:
        advice.append("峰值粉尘浓度偏高，建议在主要产尘点增设喷雾、捕尘或封闭除尘设施。")
    if data["Ctwa"] >= 4:
        advice.append("时间加权平均浓度偏高，建议优化通风量和除尘设备运行时段。")
    if data["SiO2"] >= 20:
        advice.append("游离二氧化硅含量较高，应提高呼吸防护等级并强化高硅粉尘岗位管控。")
    if data["time/week"] >= 50:
        advice.append("每周作业时间较长，建议控制接尘时长并设置离尘休息。")
    if data["protect"] < 1:
        advice.append("防护措施有效性偏低，建议开展防护用品适配性检查和现场佩戴培训。")

    advice.append(get_job_tip(job_key))
    return advice[:6]


def render_result_panel(title, prediction, risk_info):
    gauge_width = min(max(prediction / 25 * 100, 2), 100)
    st.markdown(
        f"""
<div class="result-panel">
    <div class="risk-label">{title}</div>
    <div class="risk-level" style="color:{risk_info['color']};">
        <span class="risk-dot"></span>
        <span>{risk_info['level']}</span>
    </div>
    <div style="height:18px;"></div>
    <div class="risk-label">模型预测异常风险值</div>
    <div class="risk-value">{prediction:.4f}%</div>
    <div class="gauge-track">
        <div class="gauge-fill" style="width:{gauge_width:.1f}%;"></div>
    </div>
    <div style="margin-top:14px;color:#91b6cf;line-height:1.65;">{risk_info['summary']}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_advice(advice):
    items = "".join(f"<li>{escape(item)}</li>" for item in advice)
    st.markdown(f'<ul class="advice-list">{items}</ul>', unsafe_allow_html=True)


def render_history(history):
    rows = ""
    for item in history:
        rows += (
            "<tr>"
            f"<td>{escape(item['time'])}</td>"
            f"<td>{escape(item['job'])}</td>"
            f"<td>{escape(item['level'])}</td>"
            f"<td>{item['prediction']:.4f}%</td>"
            f"<td>{item['years']:.1f}</td>"
            f"<td>{item['max']:.2f}</td>"
            f"<td>{item['protect']:.2f}</td>"
            "</tr>"
        )
    st.markdown(
        f"""
<table class="history-table">
    <thead>
        <tr>
            <th>时间</th>
            <th>工种</th>
            <th>风险等级</th>
            <th>预测值</th>
            <th>工龄</th>
            <th>峰值粉尘</th>
            <th>防护有效性</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
</table>
""",
        unsafe_allow_html=True,
    )


def status_color(value, low, high):
    if value <= low:
        return "#35d07f"
    if value <= high:
        return "#f2c94c"
    return "#ff4f5e"


def render_bar(label, value, max_value, color=None):
    width = min(max(value / max_value * 100, 3), 100)
    bar_color = color or status_color(value, max_value * 0.45, max_value * 0.72)
    return (
        f'<div class="bar-row"><span>{escape(label)}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{bar_color};"></div></div>'
        f'<strong>{value:.1f}</strong></div>'
    )


def build_alarm_items(data, risk_info):
    alarms = []
    if risk_info["level"] in ["高风险", "极高风险"]:
        alarms.append(("重点预警", risk_info["color"], f"当前预测为{risk_info['level']}，建议启动现场复核和防尘整改。"))
    if data["max"] >= 8:
        alarms.append(("粉尘峰值", "#ff8a3d", "峰值粉尘浓度偏高，请检查喷雾、捕尘和通风设施运行状态。"))
    if data["Ctwa"] >= 4:
        alarms.append(("平均浓度", "#f2c94c", "时间加权平均浓度偏高，建议优化班次暴露时长和通风量。"))
    if data["SiO2"] >= 20:
        alarms.append(("高硅粉尘", "#ff4f5e", "游离二氧化硅含量较高，请提高呼吸防护等级并加强医学随访。"))
    if data["protect"] < 1:
        alarms.append(("防护不足", "#ff4f5e", "防护有效性评分偏低，请复查口罩密合性和防护用品更换记录。"))
    if not alarms:
        alarms.append(("运行正常", "#35d07f", "当前参数未触发重点报警，建议保持常规监测。"))
    return alarms[:4]


def render_monitor_dashboard(worker_id, team_name, work_area, job_name, data, prediction, risk_info):
    dust_index = min(data["max"] * 8 + data["Ctwa"] * 10 + data["SiO2"] * 0.25, 100)
    ventilation_index = max(100 - data["Ctwa"] * 12 - data["time/week"] * 0.25, 0)
    protection_index = min(data["protect"] / 5 * 100, 100)
    exposure_index = min(data["years"] * 1.35 + data["time/week"] * 0.45, 100)

    device_status = "正常" if protection_index >= 40 and dust_index < 70 else "待复核"
    device_color = "#35d07f" if device_status == "正常" else "#f2c94c"
    health_status = "重点关注" if risk_info["level"] in ["高风险", "极高风险"] else "常规随访"
    health_color = risk_info["color"] if health_status == "重点关注" else "#35d07f"
    alarms = build_alarm_items(data, risk_info)
    alarm_html = "".join(
        f'<div class="alarm-item" style="color:{color};"><b>{escape(title)}</b><br>{escape(text)}</div>'
        for title, color, text in alarms
    )

    st.markdown(
        f"""
<div class="dashboard-grid">
    <div class="dash-panel">
        <div class="dash-title">人员健康档案</div>
        <div class="profile-line"><span>人员编号</span><strong>{escape(worker_id)}</strong></div>
        <div class="profile-line"><span>所属班组</span><strong>{escape(team_name)}</strong></div>
        <div class="profile-line"><span>作业区域</span><strong>{escape(work_area)}</strong></div>
        <div class="profile-line"><span>工种类型</span><strong>{escape(job_name)}</strong></div>
        <div class="profile-line"><span>工龄</span><strong>{data['years']:.1f} 年</strong></div>
        <div class="profile-line"><span>健康状态</span><strong><span class="status-pill" style="background:{health_color};">{health_status}</span></strong></div>
    </div>
    <div class="dash-panel">
        <div class="dash-title">井下作业区风险态势</div>
        <div class="tunnel-map">
            <div class="tunnel-line" style="left:9%;right:9%;top:52%;"></div>
            <div class="tunnel-line" style="left:28%;width:2px;height:120px;top:22%;"></div>
            <div class="tunnel-line" style="left:55%;width:2px;height:138px;top:16%;"></div>
            <div class="tunnel-line" style="left:72%;width:2px;height:92px;top:34%;"></div>
            <div class="tunnel-node" style="left:18%;top:52%;color:#35d07f;"></div>
            <div class="node-label" style="left:18%;top:52%;">通风正常</div>
            <div class="tunnel-node" style="left:44%;top:52%;color:{risk_info['color']};"></div>
            <div class="node-label" style="left:44%;top:52%;">当前岗位</div>
            <div class="tunnel-node" style="left:63%;top:52%;color:{status_color(data['max'], 6, 10)};"></div>
            <div class="node-label" style="left:63%;top:52%;">粉尘监测</div>
            <div class="tunnel-node" style="left:82%;top:52%;color:{device_color};"></div>
            <div class="node-label" style="left:82%;top:52%;">设备状态</div>
        </div>
    </div>
    <div class="dash-panel">
        <div class="dash-title">智能预警中心</div>
        <div class="alarm-list">{alarm_html}</div>
    </div>
</div>
<div style="height:14px;"></div>
<div class="dashboard-grid">
    <div class="dash-panel">
        <div class="dash-title">环境监测指标</div>
        {render_bar("粉尘指数", dust_index, 100, status_color(dust_index, 45, 72))}
        {render_bar("通风指数", ventilation_index, 100, "#35d07f" if ventilation_index >= 65 else "#f2c94c")}
        {render_bar("暴露指数", exposure_index, 100, status_color(exposure_index, 45, 72))}
        {render_bar("防护指数", protection_index, 100, "#35d07f" if protection_index >= 60 else "#ff8a3d")}
    </div>
    <div class="dash-panel">
        <div class="dash-title">设备与防护状态</div>
        <div class="profile-line"><span>通风系统</span><strong><span class="status-pill" style="background:{'#35d07f' if ventilation_index >= 65 else '#f2c94c'};">{'正常' if ventilation_index >= 65 else '需复核'}</span></strong></div>
        <div class="profile-line"><span>喷雾降尘</span><strong><span class="status-pill" style="background:{'#35d07f' if data['max'] < 8 else '#ff8a3d'};">{'有效' if data['max'] < 8 else '偏弱'}</span></strong></div>
        <div class="profile-line"><span>个体防护</span><strong><span class="status-pill" style="background:{'#35d07f' if protection_index >= 60 else '#ff8a3d'};">{'达标' if protection_index >= 60 else '加强'}</span></strong></div>
        <div class="profile-line"><span>监测频次</span><strong>{'加密监测' if risk_info['level'] in ['高风险', '极高风险'] else '常规监测'}</strong></div>
        <div class="profile-line"><span>处置建议</span><strong>{'立即复核' if risk_info['level'] == '极高风险' else '跟踪整改' if risk_info['level'] == '高风险' else '持续观察'}</strong></div>
    </div>
    <div class="dash-panel">
        <div class="dash-title">班组风险概览</div>
        <div class="profile-line"><span>当前预测值</span><strong>{prediction:.4f}%</strong></div>
        <div class="profile-line"><span>当前风险等级</span><strong><span class="status-pill" style="background:{risk_info['color']};">{risk_info['level']}</span></strong></div>
        <div class="profile-line"><span>近 7 日复查建议</span><strong>{'建议复查' if risk_info['level'] in ['高风险', '极高风险'] else '无需加急'}</strong></div>
        <div class="profile-line"><span>重点因素</span><strong>{'峰值粉尘' if data['max'] >= data['Ctwa'] else '平均浓度'}</strong></div>
        <div class="profile-line"><span>职业健康建议</span><strong>{'安排体检' if risk_info['level'] in ['高风险', '极高风险'] else '按期体检'}</strong></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# Matplotlib 基础配置（专供图形使用）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial', 'Helvetica', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# Core Logic & 5-Stack Model Loading
# ==========================================
@st.cache_resource
def load_and_train_model():
    file_path = Path(__file__).with_name("煤矿数据.xlsx")
    features = ['years', 'time/week', 'blasting', 'transport', 'extract', 'support', 'repair', 'other', 'max', 'Ctwa',
                'SiO2', 'protect']

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        st.warning("未在脚本目录找到“煤矿数据.xlsx”，当前使用模拟数据完成页面演示。")
        df = pd.DataFrame(np.random.rand(200, 13), columns=features + ['abnormal'])
        df['years'] = df['years'] * 30
        df['SiO2'] = df['SiO2'] * 100

    x = df[features].copy()
    y = df['abnormal'].copy()
    x = x.fillna(x.median(numeric_only=True))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    base_models = [
        RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=6, random_state=100),
        RidgeCV(alphas=[0.1, 1, 10, 50]),
        KNeighborsRegressor(n_neighbors=5),
        SVR(kernel="rbf", C=10, gamma=0.1),
        XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=100, n_jobs=-1)
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    xtrain_meta = np.zeros((x_scaled.shape[0], len(base_models)))

    for m, model in enumerate(base_models):
        for train_idx, val_idx in kf.split(x_scaled):
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            model.fit(x_scaled[train_idx], y_train_fold)
            xtrain_meta[val_idx, m] = model.predict(x_scaled[val_idx])
        model.fit(x_scaled, y)

    meta_model = RidgeCV()
    meta_model.fit(xtrain_meta, y)
    background = shap.sample(pd.DataFrame(x_scaled, columns=features), min(200, len(x_scaled)), random_state=42)

    def stacking_predict(X_input):
        base_preds = [model.predict(X_input) for model in base_models]
        return meta_model.predict(np.column_stack(base_preds))

    explainer = shap.KernelExplainer(stacking_predict, background)
    return base_models, meta_model, scaler, explainer, features, stacking_predict


with st.spinner("正在加载数据并训练模型，请稍候..."):
    base_models, meta_model, scaler, explainer, features, stacking_predict = load_and_train_model()

# ==========================================
# Part 1: Input Section
# ==========================================
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

st.markdown('<div class="section-card"><div class="section-title">一、预测参数录入</div>', unsafe_allow_html=True)

input_data = {}
col1, col2 = st.columns(2)
with col1:
    input_data['years'] = st.number_input("工龄（年）", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    input_data['max'] = st.number_input("峰值粉尘浓度（mg/m³）", min_value=0.0, value=5.0, step=0.5)
    input_data['SiO2'] = st.number_input("游离二氧化硅含量（%）", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
with col2:
    input_data['time/week'] = st.number_input("每周工作时长（小时）", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    input_data['Ctwa'] = st.number_input("时间加权平均浓度（mg/m³）", min_value=0.0, value=2.0, step=0.5)
    input_data['protect'] = st.number_input("防护措施有效性评分", min_value=0.0, max_value=5.0, value=1.0, step=0.5)

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    worker_id = st.text_input("人员编号", value="MINE-001")
with info_col2:
    team_name = st.selectbox("所属班组", ["综采一队", "掘进二队", "运输队", "通风队", "机电检修队"])
with info_col3:
    work_area = st.selectbox("作业区域", ["一采区", "二采区", "主运输巷", "回风巷", "检修硐室"])

job_options = {
    "爆破作业": "blasting",
    "运输作业": "transport",
    "采掘作业": "extract",
    "支护作业": "support",
    "检修作业": "repair",
    "其他岗位": "other"
}
selected_job_zh = st.selectbox("请选择工种类型", list(job_options.keys()))
selected_job_en = job_options[selected_job_zh]
for en_name in job_options.values():
    input_data[en_name] = 1.0 if en_name == selected_job_en else 0.0

input_df = pd.DataFrame([input_data])[features]
current_prediction = float(stacking_predict(scaler.transform(input_df))[0])
current_risk_info = get_risk_info(current_prediction)

st.markdown(
    f"""
<div class="metric-grid">
    <div class="mini-metric"><div class="label">当前工种</div><div class="value">{selected_job_zh}</div></div>
    <div class="mini-metric"><div class="label">预估风险等级</div><div class="value" style="color:{current_risk_info['color']};">{current_risk_info['level']}</div></div>
    <div class="mini-metric"><div class="label">预估风险值</div><div class="value">{current_prediction:.4f}%</div></div>
    <div class="mini-metric"><div class="label">防护有效性</div><div class="value">{input_data['protect']:.1f}</div></div>
</div>
<div class="hint-box"><b>职业类型说明：</b>{get_job_tip(selected_job_en)}</div>
""",
    unsafe_allow_html=True,
)

run_prediction = st.button("开始风险预测", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

display_names_plot = {
    'years': '工龄', 'time/week': '每周工时', 'blasting': '爆破作业',
    'transport': '运输作业', 'extract': '采掘作业', 'support': '支护作业',
    'repair': '检修作业', 'other': '其他岗位', 'max': '峰值粉尘浓度',
    'Ctwa': '时间加权平均浓度', 'SiO2': '游离二氧化硅', 'protect': '防护有效性'
}

display_names_force = display_names_plot.copy()

if run_prediction:
    st.session_state["latest_prediction"] = {
        "prediction": current_prediction,
        "risk_info": current_risk_info,
        "input_df": input_df.copy(),
        "input_data": input_data.copy(),
        "job_key": selected_job_en,
        "job_name": selected_job_zh,
        "worker_id": worker_id,
        "team_name": team_name,
        "work_area": work_area,
    }
    st.session_state["prediction_history"].insert(
        0,
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "job": selected_job_zh,
            "level": current_risk_info["level"],
            "prediction": current_prediction,
            "years": input_data["years"],
            "max": input_data["max"],
            "protect": input_data["protect"],
        },
    )
    st.session_state["prediction_history"] = st.session_state["prediction_history"][:6]


# ==========================================
# Part 2: Prediction & Analysis
# ==========================================
st.markdown('<div class="section-card"><div class="section-title">二、预测结果与防护建议</div>', unsafe_allow_html=True)

latest = st.session_state.get("latest_prediction")
if latest:
    result_col1, result_col2 = st.columns([1.05, 1])
    with result_col1:
        render_result_panel("尘肺病风险等级", latest["prediction"], latest["risk_info"])
    with result_col2:
        st.markdown('<div class="hint-box"><b>个性化防护建议</b></div>', unsafe_allow_html=True)
        render_advice(build_protection_advice(latest["risk_info"], latest["prediction"], latest["input_data"], latest["job_key"]))
else:
    st.info("请先录入参数并点击“开始风险预测”，系统会生成风险等级、预测值和防护建议。")
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">三、风险趋势模拟</div>', unsafe_allow_html=True)
sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    sim_years = st.slider("模拟工龄（年）", 0.0, 50.0, float(input_data["years"]), 1.0)
with sim_col2:
    sim_max = st.slider("模拟峰值粉尘浓度（mg/m³）", 0.0, 30.0, float(input_data["max"]), 0.5)
with sim_col3:
    sim_protect = st.slider("模拟防护措施有效性", 0.0, 5.0, float(input_data["protect"]), 0.5)

sim_data = input_data.copy()
sim_data["years"] = sim_years
sim_data["max"] = sim_max
sim_data["protect"] = sim_protect
sim_df = pd.DataFrame([sim_data])[features]
sim_prediction = float(stacking_predict(scaler.transform(sim_df))[0])
sim_risk_info = get_risk_info(sim_prediction)

trend_col1, trend_col2 = st.columns([1, 1])
with trend_col1:
    render_result_panel("模拟风险等级", sim_prediction, sim_risk_info)
with trend_col2:
    delta = sim_prediction - current_prediction
    st.markdown(
        f"""
<div class="hint-box">
    <b>模拟变化对比</b><br>
    当前输入风险值：{current_prediction:.4f}%<br>
    模拟条件风险值：{sim_prediction:.4f}%<br>
    风险变化：{delta:+.4f}%<br>
    调整工龄、粉尘峰值或防护有效性后，系统会自动刷新模拟结果。
</div>
""",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">四、监测预警驾驶舱</div>', unsafe_allow_html=True)
dashboard_source = latest or {
    "prediction": current_prediction,
    "risk_info": current_risk_info,
    "input_data": input_data,
    "job_name": selected_job_zh,
    "worker_id": worker_id,
    "team_name": team_name,
    "work_area": work_area,
}
render_monitor_dashboard(
    dashboard_source["worker_id"],
    dashboard_source["team_name"],
    dashboard_source["work_area"],
    dashboard_source["job_name"],
    dashboard_source["input_data"],
    dashboard_source["prediction"],
    dashboard_source["risk_info"],
)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="section-card"><div class="section-title">五、最近预测记录</div>', unsafe_allow_html=True)
if st.session_state.get("prediction_history", []):
    render_history(st.session_state["prediction_history"])
else:
    st.info("暂无历史记录。完成预测后，系统会保留最近 6 次结果用于对比。")
st.markdown("</div>", unsafe_allow_html=True)


if latest:
    input_df_for_analysis = latest["input_df"]
    input_scaled = scaler.transform(input_df_for_analysis)
    selected_job_en_for_analysis = latest["job_key"]

    with st.expander("查看可解释性分析图和归因整改建议", expanded=False):
        with st.spinner("正在生成可解释性分析图..."):
            shap_values_raw = explainer.shap_values(input_scaled, nsamples=100)
            unselected_jobs = [job for job in job_options.values() if job != selected_job_en_for_analysis]
            keep_indices = [i for i, f in enumerate(features) if f not in unselected_jobs]
            unselected_indices = [i for i, f in enumerate(features) if f in unselected_jobs]
            unselected_shap_sum = np.sum(shap_values_raw[0][unselected_indices])
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = expected_val[0]

            adjusted_base_value = float(expected_val) + float(unselected_shap_sum)
            adjusted_values = shap_values_raw[0][keep_indices]
            adjusted_data = input_df_for_analysis.iloc[0].values[keep_indices]
            adjusted_features_display_plot = [display_names_plot[features[i]] for i in keep_indices]
            adjusted_features_display_force = [display_names_force[features[i]] for i in keep_indices]

            shap_exp = shap.Explanation(
                values=adjusted_values,
                base_values=adjusted_base_value,
                data=adjusted_data,
                feature_names=adjusted_features_display_plot
            )

            try:
                from shap.plots import colors as shap_colors
                shap_colors.red.rgb = mcolors.hex2color(COLOR_RED)
                shap_colors.blue.rgb = mcolors.hex2color(COLOR_BLUE)
            except Exception:
                pass

            st.subheader("风险累积归因分析")
            fig_waterfall, ax_wf = plt.subplots(figsize=(10, 6), dpi=220)
            shap.plots.waterfall(shap_exp, show=False, max_display=10)

            for patch in ax_wf.patches:
                try:
                    fc = mcolors.to_rgb(patch.get_facecolor())
                    if fc[0] > fc[2] + 0.1:
                        patch.set_facecolor(COLOR_RED)
                        patch.set_edgecolor(COLOR_RED)
                    elif fc[2] > fc[0] + 0.1:
                        patch.set_facecolor(COLOR_BLUE)
                        patch.set_edgecolor(COLOR_BLUE)
                except Exception:
                    pass

            for text in ax_wf.texts:
                try:
                    text.set_fontweight('bold')
                    c = mcolors.to_rgb(text.get_color())
                    if sum(c) > 2.8:
                        continue
                    if c[0] > c[2] + 0.1:
                        text.set_color(COLOR_RED)
                    elif c[2] > c[0] + 0.1:
                        text.set_color(COLOR_BLUE)
                except Exception:
                    pass

            for line in ax_wf.lines:
                if line.get_linestyle() == '--':
                    line.set_color('#cccccc')

            plt.tight_layout()
            st.pyplot(fig_waterfall, use_container_width=True)
            plt.close(fig_waterfall)

            st.markdown("---")
            st.subheader("风险驱动力分析")
            try:
                rounded_features = np.round(adjusted_data.astype(float), 2)
                force_plot_js = shap.force_plot(
                    base_value=float(adjusted_base_value),
                    shap_values=adjusted_values,
                    features=rounded_features,
                    feature_names=adjusted_features_display_force,
                    plot_cmap=[COLOR_BLUE, COLOR_RED],
                    contribution_threshold=0.0
                )
                min_width = max(1800, 260 * len(adjusted_features_display_force))
                st_shap(force_plot_js, height=320, min_width=min_width)
            except Exception as e:
                st.warning(f"力图生成失败：{e}")

        st.markdown("---")
        st.subheader("归因整改建议")
        feature_shap_dict = {feat: val for feat, val in zip(adjusted_features_display_force, adjusted_values)}
        sorted_features = sorted(feature_shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_risk_features = [item for item in sorted_features if item[1] > 0][:3]

        if top_risk_features:
            st.write("根据模型归因结果，以下因素是当前风险升高的主要来源，建议优先落实对应整改措施：")
            measures_dict = {
                '峰值粉尘浓度': "**降低峰值粉尘浓度**：在主要产尘点安装喷雾降尘设施，优化通风，推行湿式作业。",
                '时间加权平均浓度': "**控制时间加权平均浓度**：提升通风和除尘效率，对重点区域进行封闭降尘，保持水幕常态化开启。",
                '游离二氧化硅': "**管控高游离二氧化硅暴露**：采用抽压结合的除尘风机方案，并提高该类岗位呼吸防护等级。",
                '工龄': "**加强高工龄人员健康管理**：累积暴露风险较高，建议增加体检频次并优先安排离尘轮岗。",
                '每周工时': "**优化每周工作时长**：严格控制接尘作业时间，落实离尘休息制度。",
                '防护有效性': "**提升防护措施有效性**：检查防护用品配置，加强口罩密合性检查和佩戴培训。",
                '爆破作业': "**规范爆破作业**：采用水炮泥和水封爆破技术，爆破前后喷雾洒水，保证通风排尘时间。",
                '采掘作业': "**规范采掘作业**：确保采煤机内外喷雾达标，联动支架喷雾，并落实煤层注水。",
                '支护作业': "**规范支护作业**：严禁干式打眼，使用湿式凿岩，喷浆作业配套湿式除尘。",
                '运输作业': "**规范运输作业**：转载点封闭喷雾，及时清理积尘，防止二次扬尘。",
                '检修作业': "**规范检修作业**：进入高浓度区域检修前先洒水降尘，并配备便携式防尘呼吸器。",
                '其他岗位': "**加强综合岗位防护**：结合现场实际配置防尘措施，强化个人防护佩戴监督。"
            }
            for idx, (feat, val) in enumerate(top_risk_features):
                st.info(f"**风险因素 {idx + 1}（风险增量：+{val:.2f}）**\n\n{measures_dict.get(feat, '')}")
        else:
            st.info("当前样本没有明显的正向风险驱动因素。")
