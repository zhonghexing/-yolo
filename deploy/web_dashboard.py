"""
远程监控 Web 面板
基于 Flask + SocketIO 实时推送检测数据

功能：
    - 实时检测状态监控
    - 缺陷趋势图（24小时/7天/30天）
    - 检测历史查询
    - 严重程度统计
    - 报警配置管理
    - 移动端自适应

启动后访问 http://localhost:5000
"""

import json
import os
import sys
import base64
from datetime import datetime
from pathlib import Path
from threading import Lock

from flask import Flask, render_template_string, jsonify, request, send_file, make_response
from flask_socketio import SocketIO

import db

app = Flask(__name__)
app.config['SECRET_KEY'] = 'steel-defect-inspector'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 实时数据锁
data_lock = Lock()


# ══════════════════════════════════════════════
# HTML 模板
# ══════════════════════════════════════════════

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>钢材缺陷检测 - 远程监控</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0f0f0f; color: #e0e0e0; font-family: -apple-system, 'Segoe UI', 'Microsoft YaHei', sans-serif; }

.header { background: #1a1a1a; padding: 16px 24px; border-bottom: 1px solid #2a2a2a; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 18px; color: #4a9eff; }
.header .status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #888; }
.header .dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

.container { max-width: 1400px; margin: 0 auto; padding: 16px; }

/* 导航标签 */
.nav-tabs { display: flex; gap: 4px; margin-bottom: 16px; background: #1a1a1a; padding: 4px; border-radius: 8px; }
.nav-tab { padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; color: #888; transition: all 0.2s; }
.nav-tab:hover { color: #e0e0e0; background: #222; }
.nav-tab.active { background: #4a9eff; color: #fff; }

/* 统计卡片 */
.stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 16px; }
.stat-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; transition: transform 0.2s; }
.stat-card:hover { transform: translateY(-2px); }
.stat-card .label { font-size: 12px; color: #888; margin-bottom: 4px; }
.stat-card .value { font-size: 28px; font-weight: 700; }
.stat-card .sub { font-size: 11px; color: #555; margin-top: 4px; }
.stat-card.pass { border-left: 3px solid #4ade80; }
.stat-card.pass .value { color: #4ade80; }
.stat-card.fail { border-left: 3px solid #f87171; }
.stat-card.fail .value { color: #f87171; }
.stat-card.total { border-left: 3px solid #4a9eff; }
.stat-card.total .value { color: #4a9eff; }
.stat-card.rate { border-left: 3px solid #fbbf24; }
.stat-card.rate .value { color: #fbbf24; }
.stat-card.speed { border-left: 3px solid #a78bfa; }
.stat-card.speed .value { color: #a78bfa; }

/* 图表区 */
.charts { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 16px; }
.chart-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; }
.chart-card h3 { font-size: 13px; color: #888; margin-bottom: 12px; font-weight: 500; }
.chart-container { position: relative; height: 250px; }

/* 缺陷分布 */
.defect-bars { display: flex; flex-direction: column; gap: 8px; }
.defect-bar { display: flex; align-items: center; gap: 8px; }
.defect-bar .name { width: 50px; font-size: 12px; color: #aaa; text-align: right; }
.defect-bar .bar-bg { flex: 1; height: 20px; background: #222; border-radius: 4px; overflow: hidden; }
.defect-bar .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; background: linear-gradient(90deg, #4a9eff, #60a5fa); }
.defect-bar .count { width: 30px; font-size: 12px; color: #aaa; text-align: left; }

/* 最近检测表格 */
.table-card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.table-card h3 { font-size: 13px; color: #888; margin-bottom: 12px; font-weight: 500; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: #222; color: #888; padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2a2a; font-weight: 500; }
td { padding: 8px 12px; border-bottom: 1px solid #1f1f1f; }
tr:hover { background: #1f1f1f; }
.severity-严重 { color: #f87171; font-weight: 600; }
.severity-中等 { color: #fbbf24; }
.severity-轻微 { color: #4ade80; }
.severity-合格 { color: #4ade80; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-defect { background: rgba(248,113,113,0.15); color: #f87171; }
.badge-pass { background: rgba(74,222,128,0.15); color: #4ade80; }

/* 图片预览 */
.img-preview { cursor: pointer; transition: transform 0.2s; }
.img-preview:hover { transform: scale(1.05); }
.img-thumb { width: 60px; height: 40px; object-fit: cover; border-radius: 4px; border: 1px solid #2a2a2a; }

/* 图片模态框 */
.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.97); overflow: hidden; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: #1a1a1a; height: 44px; }
.modal-header h3 { color: #e0e0e0; font-size: 14px; margin: 0; }
.close { color: #888; font-size: 32px; font-weight: bold; cursor: pointer; padding: 0 8px; line-height: 1; }
.close:hover { color: #fff; }
.modal-tabs { display: flex; gap: 8px; padding: 8px 20px; background: #1a1a1a; }
.modal-tab { padding: 8px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; color: #888; background: #222; border: none; }
.modal-tab.active { background: #4a9eff; color: #fff; }
.modal-tab:hover { background: #333; }
.img-container { display: flex; align-items: center; justify-content: center; width: 100%; height: calc(100vh - 100px); padding: 20px; overflow: hidden; }
.img-container img { display: block; width: 90vw; height: 85vh; object-fit: contain; border-radius: 4px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
.no-image { color: #555; font-size: 18px; padding: 60px; }

/* 实时告警 */
.alerts { max-height: 150px; overflow-y: auto; }
.alert { background: rgba(248,113,113,0.08); border-left: 3px solid #f87171; padding: 8px 12px; margin-bottom: 4px; border-radius: 0 4px 4px 0; font-size: 12px; animation: slideIn 0.3s ease; }
.alert .time { color: #555; margin-right: 8px; }

/* 反馈活动面板 */
.feedback-panel { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.feedback-panel h3 { font-size: 13px; color: #888; margin-bottom: 12px; font-weight: 500; }
.feedback-feed { max-height: 200px; overflow-y: auto; }
.feedback-item { display: flex; align-items: flex-start; gap: 10px; padding: 8px 12px; margin-bottom: 4px; border-radius: 6px; animation: slideIn 0.3s ease; }
.feedback-item.pass { background: rgba(74,222,128,0.06); border-left: 3px solid #4ade80; }
.feedback-item.defect { background: rgba(248,113,113,0.06); border-left: 3px solid #f87171; }
.feedback-item .fb-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.feedback-item .fb-body { flex: 1; min-width: 0; }
.feedback-item .fb-text { font-size: 12px; color: #e0e0e0; }
.feedback-item .fb-meta { font-size: 11px; color: #555; margin-top: 2px; display: flex; gap: 10px; }
.feedback-item .fb-meta .fb-tag { display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; }
.feedback-item .fb-meta .fb-tag-voice { background: rgba(74,158,255,0.15); color: #4a9eff; }
.feedback-item .fb-meta .fb-tag-log { background: rgba(251,191,36,0.15); color: #fbbf24; }
.feedback-item .fb-meta .fb-tag-visual { background: rgba(168,85,247,0.15); color: #a855f7; }
@keyframes slideIn { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: translateX(0); } }

/* 严重程度统计 */
.severity-pie { display: flex; justify-content: center; gap: 24px; margin-top: 12px; }
.severity-item { text-align: center; }
.severity-item .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 4px; }
.severity-item .label { font-size: 12px; color: #888; }
.severity-item .val { font-size: 18px; font-weight: 700; margin-top: 4px; }

/* 报警配置面板 */
.config-panel { display: none; }
.config-panel.active { display: block; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.config-item { background: #222; border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px; }
.config-item .name { font-size: 14px; font-weight: 600; margin-bottom: 8px; }
.config-item label { font-size: 12px; color: #888; display: block; margin-bottom: 4px; }
.config-item input[type=range] { width: 100%; margin: 4px 0; }
.config-item .threshold-val { font-size: 14px; color: #4a9eff; font-weight: 600; }

/* 分页 */
.pagination { display: flex; justify-content: center; gap: 8px; margin-top: 12px; }
.pagination button { background: #222; color: #888; border: 1px solid #2a2a2a; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.pagination button:hover { background: #333; color: #e0e0e0; }
.pagination button.active { background: #4a9eff; color: #fff; border-color: #4a9eff; }

/* 响应式 */
@media (max-width: 900px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .charts { grid-template-columns: 1fr; }
    .config-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="header">
    <h1>🔧 钢材缺陷检测系统 - 远程监控</h1>
    <div class="status">
        <div class="dot" id="statusDot"></div>
        <span id="statusText">连接中...</span>
        <span style="margin-left: 16px;" id="lastUpdate"></span>
    </div>
</div>

<div class="container">
    <!-- 导航标签 -->
    <div class="nav-tabs">
        <div class="nav-tab active" onclick="switchTab('dashboard')">📊 监控面板</div>
        <div class="nav-tab" onclick="switchTab('history')">📋 检测历史</div>
        <div class="nav-tab" onclick="switchTab('visualization')">📈 可视化分析</div>
        <div class="nav-tab" onclick="switchTab('config')">⚙️ 报警配置</div>
    </div>

    <!-- 监控面板 -->
    <div id="tab-dashboard">
        <!-- 统计卡片 -->
        <div class="stats">
            <div class="stat-card total">
                <div class="label">检测总数</div>
                <div class="value" id="statTotal">0</div>
                <div class="sub">过去 24 小时</div>
            </div>
            <div class="stat-card pass">
                <div class="label">合格数</div>
                <div class="value" id="statPass">0</div>
            </div>
            <div class="stat-card fail">
                <div class="label">缺陷数</div>
                <div class="value" id="statDefect">0</div>
            </div>
            <div class="stat-card rate">
                <div class="label">合格率</div>
                <div class="value" id="statRate">0%</div>
            </div>
            <div class="stat-card speed">
                <div class="label">平均推理</div>
                <div class="value" id="statSpeed">0ms</div>
            </div>
        </div>

        <!-- 实时告警 -->
        <div class="table-card" id="alertCard" style="display:none;">
            <h3>⚠️ 最近告警</h3>
            <div class="alerts" id="alerts"></div>
        </div>

        <!-- 反馈活动面板 -->
        <div class="feedback-panel">
            <h3>🔔 反馈活动 <span style="color:#555;font-size:11px;margin-left:8px;">语音 / 日志 / 视觉</span></h3>
            <div class="feedback-feed" id="feedbackFeed">
                <div style="color:#555;font-size:12px;padding:12px;text-align:center;">等待检测...</div>
            </div>
        </div>

        <!-- 图表区 -->
        <div class="charts">
            <div class="chart-card">
                <h3>缺陷趋势
                    <select id="trendRange" onchange="loadTrend()" style="background:#222;color:#888;border:1px solid #2a2a2a;padding:2px 6px;border-radius:4px;font-size:11px;margin-left:8px;">
                        <option value="24">24小时</option>
                        <option value="168">7天</option>
                        <option value="720">30天</option>
                    </select>
                </h3>
                <div class="chart-container">
                    <canvas id="trendChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h3>缺陷分布</h3>
                <div class="defect-bars" id="defectBars"></div>
                <h3 style="margin-top:16px;">严重程度</h3>
                <div class="severity-pie" id="severityPie"></div>
            </div>
        </div>

        <!-- 最近检测记录 -->
        <div class="table-card">
            <h3>最近检测记录 <span style="color:#555;font-size:11px;">（实时更新）</span></h3>
            <table>
                <thead>
                    <tr><th>时间</th><th>图片</th><th>缺陷类型</th><th>置信度</th><th>严重程度</th><th>判定</th></tr>
                </thead>
                <tbody id="recentTable"></tbody>
            </table>
        </div>
    </div>

    <!-- 检测历史 -->
    <div id="tab-visualization" class="config-panel">
        <div class="table-card">
            <h3>可视化分析报告</h3>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
                <select id="vizTimeRange" onchange="refreshCharts()" style="background:#222;color:#e0e0e0;border:1px solid #2a2a2a;padding:6px 10px;border-radius:4px;font-size:12px;">
                    <option value="24">最近 24 小时</option>
                    <option value="168">最近 7 天</option>
                    <option value="720">最近 30 天</option>
                    <option value="all" selected>全部数据</option>
                </select>
                <button onclick="refreshCharts()" style="background:#4a9eff;color:#fff;border:none;padding:6px 12px;border-radius:4px;font-size:12px;cursor:pointer;">🔄 刷新</button>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px;" id="chartsGrid">
                <div class="chart-card">
                    <h3>缺陷趋势</h3>
                    <div style="position:relative;height:250px;"><canvas id="viz-trend"></canvas></div>
                </div>
                <div class="chart-card">
                    <h3>缺陷类型分布</h3>
                    <div style="position:relative;height:250px;"><canvas id="viz-class-dist"></canvas></div>
                </div>
                <div class="chart-card">
                    <h3>合格 / 不合格</h3>
                    <div style="position:relative;height:250px;"><canvas id="viz-passfail"></canvas></div>
                </div>
                <div class="chart-card">
                    <h3>每日检测量</h3>
                    <div style="position:relative;height:250px;"><canvas id="viz-daily"></canvas></div>
                </div>
            </div>
        </div>
    </div>

    <div id="tab-history" class="config-panel">
        <div class="table-card">
            <h3>检测历史查询</h3>
            <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
                <select id="filterSeverity" onchange="loadHistory(1)" style="background:#222;color:#e0e0e0;border:1px solid #2a2a2a;padding:6px 10px;border-radius:4px;font-size:12px;">
                    <option value="">全部严重程度</option>
                    <option value="严重">严重</option>
                    <option value="中等">中等</option>
                    <option value="轻微">轻微</option>
                    <option value="合格">合格</option>
                </select>
                <select id="filterDefect" onchange="loadHistory(1)" style="background:#222;color:#e0e0e0;border:1px solid #2a2a2a;padding:6px 10px;border-radius:4px;font-size:12px;">
                    <option value="">全部状态</option>
                    <option value="1">有缺陷</option>
                    <option value="0">合格</option>
                </select>
            </div>
            <table>
                <thead>
                    <tr><th>ID</th><th>时间</th><th>原图</th><th>标注图</th><th>缺陷类型</th><th>置信度</th><th>严重程度</th><th>推理时间</th><th>判定</th><th>操作</th></tr>
                </thead>
                <tbody id="historyTable"></tbody>
            </table>
            <div class="pagination" id="historyPagination"></div>
        </div>
    </div>

    <!-- 图片查看模态框 -->
    <div id="imageModal" class="modal">
        <div class="modal-header">
            <h3 id="modalTitle">图片预览</h3>
            <span class="close" onclick="closeModal()">&times;</span>
        </div>
        <div class="modal-tabs">
            <div class="modal-tab active" onclick="switchModalTab('original')">📷 原图</div>
            <div class="modal-tab" onclick="switchModalTab('annotated')">✏️ 标注图</div>
        </div>
        <div class="img-container" id="imgContainer">
            <img id="modalImage" src="" alt="图片预览" style="display:none;">
            <div class="no-image" style="display:none;">暂无图片</div>
        </div>
    </div>

    <!-- 报警配置 -->
    <div id="tab-config" class="config-panel">
        <div class="table-card">
            <h3>报警阈值配置（按类别）</h3>
            <p style="font-size:12px;color:#555;margin-bottom:12px;">调整各类别的严重程度判定阈值和报警开关</p>
            <div class="config-grid" id="configGrid"></div>
        </div>
    </div>
</div>

<script>
const socket = io();
let trendChart = null;
let historyPage = 1;
const PAGE_SIZE = 20;

// 连接状态
socket.on('connect', () => {
    document.getElementById('statusDot').style.background = '#4ade80';
    document.getElementById('statusText').textContent = '已连接';
});
socket.on('disconnect', () => {
    document.getElementById('statusDot').style.background = '#f87171';
    document.getElementById('statusText').textContent = '已断开';
});

// 接收实时检测结果
socket.on('new_detection', (data) => {
    updateStats(data.stats);
    addAlert(data.detection);
    addRecentRow(data.detection);
    updateDefectBars(data.stats.class_stats);
    updateSeverityPie(data.stats.severity_stats);
    document.getElementById('lastUpdate').textContent = '更新于 ' + new Date().toLocaleTimeString();
});

// 接收反馈事件
socket.on('feedback_event', (data) => {
    addFeedbackItem(data);
});

// 图片模态框
let currentImagePath = '';
let currentAnnotatedPath = '';

function openModal(imagePath, annotatedPath, title) {
    currentImagePath = imagePath || '';
    currentAnnotatedPath = annotatedPath || '';
    document.getElementById('modalTitle').textContent = title || '图片预览';
    document.getElementById('imageModal').style.display = 'block';
    document.body.style.overflow = 'hidden'; // 防止背景滚动

    // 默认显示原图，如果没有原图则显示标注图
    if (currentImagePath) {
        switchModalTab('original');
    } else if (currentAnnotatedPath) {
        switchModalTab('annotated');
    }
}

function closeModal() {
    document.getElementById('imageModal').style.display = 'none';
    document.body.style.overflow = ''; // 恢复背景滚动
}

function switchModalTab(tab) {
    // 更新标签样式
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
    if (event && event.target) {
        event.target.classList.add('active');
    } else {
        // 默认选中第一个标签
        document.querySelector('.modal-tab').classList.add('active');
    }

    const img = document.getElementById('modalImage');
    const noImg = document.querySelector('.no-image');

    if (tab === 'original' && currentImagePath) {
        img.src = '/api/image?path=' + encodeURIComponent(currentImagePath);
        img.style.cssText = 'display:block !important; width:90vw !important; height:85vh !important; object-fit:contain; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5);';
        if (noImg) noImg.style.display = 'none';
    } else if (tab === 'annotated' && currentAnnotatedPath) {
        img.src = '/api/image?path=' + encodeURIComponent(currentAnnotatedPath);
        img.style.cssText = 'display:block !important; width:90vw !important; height:85vh !important; object-fit:contain; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5);';
        if (noImg) noImg.style.display = 'none';
    } else {
        img.src = '';
        img.style.display = 'none';
        if (noImg) noImg.style.display = 'block';
    }
}

// 点击模态框背景关闭
document.getElementById('imageModal').addEventListener('click', function(e) {
    if (e.target === this) closeModal();
});

// ESC 键关闭模态框
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
});

// 切换标签
function switchTab(tab) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.config-panel').forEach(p => p.classList.remove('active'));
    event.target.classList.add('active');

    if (tab === 'dashboard') {
        document.getElementById('tab-dashboard').style.display = 'block';
    } else {
        document.getElementById('tab-dashboard').style.display = 'none';
        document.getElementById('tab-' + tab).classList.add('active');
        if (tab === 'history') loadHistory(1);
        if (tab === 'config') loadConfig();
    }
}

// 加载趋势
function loadTrend() {
    const hours = document.getElementById('trendRange').value;
    fetch('/api/trend?hours=' + hours).then(r => r.json()).then(data => {
        initTrendChart(data, hours);
    });
}

// 初始加载
fetch('/api/stats').then(r => r.json()).then(data => {
    updateStats(data);
    updateDefectBars(data.class_stats);
    updateSeverityPie(data.severity_stats);
});

fetch('/api/recent?limit=20').then(r => r.json()).then(data => {
    data.forEach(d => addRecentRow(d));
});

loadTrend();

function updateStats(s) {
    document.getElementById('statTotal').textContent = s.total;
    document.getElementById('statPass').textContent = s.pass_count;
    document.getElementById('statDefect').textContent = s.defects;
    document.getElementById('statRate').textContent = s.pass_rate.toFixed(1) + '%';
    document.getElementById('statSpeed').textContent = s.avg_inference_ms + 'ms';
}

function addAlert(d) {
    if (!d.has_defect) return;
    const card = document.getElementById('alertCard');
    card.style.display = 'block';
    const el = document.createElement('div');
    el.className = 'alert';
    el.innerHTML = `<span class="time">${d.timestamp || new Date().toLocaleTimeString()}</span>
        <span class="severity-${d.severity}">[${d.severity}]</span>
        ${d.class_name_cn} 置信度 ${(d.confidence*100).toFixed(0)}%`;
    const container = document.getElementById('alerts');
    container.insertBefore(el, container.firstChild);
    if (container.children.length > 20) container.removeChild(container.lastChild);
}

function addRecentRow(d) {
    const tbody = document.getElementById('recentTable');
    const row = document.createElement('tr');
    const badge = d.has_defect ? '<span class="badge badge-defect">缺陷</span>' : '<span class="badge badge-pass">合格</span>';
    const fileName = d.image_path ? d.image_path.split('/').pop().split('\\\\').pop() : '-';

    // 图片预览按钮 - 使用 data 属性存储路径，避免转义问题
    const imgBtn = d.image_path ?
        `<button class="img-preview" onclick="handleImageClick(this)" data-original="${encodeURIComponent(d.image_path)}" data-annotated="${encodeURIComponent(d.annotated_path || '')}" data-title="${encodeURIComponent(fileName)}" style="background:#222;color:#4a9eff;border:1px solid #2a2a2a;padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer;">查看</button>` : '-';

    row.innerHTML = `
        <td>${d.timestamp || '-'}</td>
        <td>${fileName} ${imgBtn}</td>
        <td>${d.class_name_cn || '-'}</td>
        <td>${d.confidence ? (d.confidence*100).toFixed(1)+'%' : '-'}</td>
        <td class="severity-${d.severity || ''}">${d.severity || '-'}</td>
        <td>${badge}</td>`;
    tbody.insertBefore(row, tbody.firstChild);
    if (tbody.children.length > 50) tbody.removeChild(tbody.lastChild);
}

function handleImageClick(btn) {
    const original = decodeURIComponent(btn.dataset.original || '');
    const annotated = decodeURIComponent(btn.dataset.annotated || '');
    const title = decodeURIComponent(btn.dataset.title || '图片预览');
    openModal(original, annotated, title);
}

function addFeedbackItem(d) {
    const feed = document.getElementById('feedbackFeed');
    // 首次清除占位文字
    if (feed.children.length === 1 && feed.children[0].style && feed.children[0].style.textAlign === 'center') {
        feed.innerHTML = '';
    }

    const isDefect = d.has_defect;
    const icon = isDefect ? '🔴' : '🟢';
    const cls = isDefect ? 'defect' : 'pass';
    const fileName = d.image_path ? d.image_path.split('/').pop().split('\\\\').pop() : '-';
    const voiceText = d.voice_text || '-';
    const confText = d.detections && d.detections.length > 0
        ? d.detections.map(det => det.class_name_cn + ' ' + (det.confidence*100).toFixed(0) + '%').join(', ')
        : '无缺陷';

    // 反馈标签
    let tags = '';
    if (d.voice_enabled) tags += '<span class="fb-tag fb-tag-voice">🔊 语音</span>';
    if (d.log_enabled) tags += '<span class="fb-tag fb-tag-log">📝 日志</span>';
    tags += '<span class="fb-tag fb-tag-visual">👁 视觉</span>';

    const el = document.createElement('div');
    el.className = 'feedback-item ' + cls;
    el.innerHTML = `
        <span class="fb-icon">${icon}</span>
        <div class="fb-body">
            <div class="fb-text">${voiceText} — ${confText}</div>
            <div class="fb-meta">
                <span>${d.timestamp || new Date().toLocaleTimeString()}</span>
                <span>${fileName}</span>
                <span>${d.inference_time_ms ? d.inference_time_ms.toFixed(1) + 'ms' : ''}</span>
                ${tags}
            </div>
        </div>`;
    feed.insertBefore(el, feed.firstChild);
    // 最多保留 30 条
    while (feed.children.length > 30) feed.removeChild(feed.lastChild);
}

function updateDefectBars(classStats) {
    if (!classStats || !classStats.length) return;
    const maxCnt = Math.max(...classStats.map(c => c.cnt), 1);
    const container = document.getElementById('defectBars');
    container.innerHTML = classStats.map(c => {
        const pct = (c.cnt / maxCnt * 100).toFixed(0);
        return `<div class="defect-bar">
            <span class="name">${c.class_name_cn}</span>
            <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
            <span class="count">${c.cnt}</span>
        </div>`;
    }).join('');
}

function updateSeverityPie(sevStats) {
    if (!sevStats) return;
    const colors = {'严重':'#f87171','中等':'#fbbf24','轻微':'#4ade80','合格':'#4ade80'};
    const container = document.getElementById('severityPie');
    container.innerHTML = sevStats.map(s =>
        `<div class="severity-item">
            <div><span class="dot" style="background:${colors[s.severity]||'#888'}"></span><span class="label">${s.severity}</span></div>
            <div class="val" style="color:${colors[s.severity]||'#888'}">${s.cnt}</div>
        </div>`
    ).join('');
}

function initTrendChart(data, hours) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (trendChart) trendChart.destroy();

    // 根据时间范围选标签格式
    const labelFormat = hours <= 24
        ? (t => t ? t.split(' ')[1].slice(0,5) : '')   // "HH:MM"
        : (t => t ? t.slice(5) : '');                    // "MM-DD"

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => labelFormat(d.time_bucket)),
            datasets: [{
                label: '缺陷数',
                data: data.map(d => d.defects || 0),
                borderColor: '#f87171',
                backgroundColor: 'rgba(248,113,113,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 3,
            }, {
                label: '检测总数',
                data: data.map(d => d.total || 0),
                borderColor: '#4a9eff',
                backgroundColor: 'rgba(74,158,255,0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#888', font: { size: 11 } } } },
            scales: {
                x: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1f1f1f' } },
                y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1f1f1f' }, beginAtZero: true }
            }
        }
    });
}

// 历史查询
function loadHistory(page) {
    historyPage = page;
    const severity = document.getElementById('filterSeverity').value;
    const defect = document.getElementById('filterDefect').value;

    let url = `/api/history?page=${page}&limit=${PAGE_SIZE}`;
    if (severity) url += `&severity=${severity}`;
    if (defect !== '') url += `&has_defect=${defect}`;

    fetch(url).then(r => r.json()).then(data => {
        const tbody = document.getElementById('historyTable');
        tbody.innerHTML = data.records.map(d => {
            const badge = d.has_defect ? '<span class="badge badge-defect">缺陷</span>' : '<span class="badge badge-pass">合格</span>';
            const fileName = d.image_name || (d.image_path ? d.image_path.split('/').pop() : '-');

            // 原图预览按钮 - 点击直接显示原图
            const origBtn = d.image_path ?
                `<button class="img-preview" onclick="openModalDirect('${encodeURIComponent(d.image_path)}', '${encodeURIComponent(d.annotated_path || '')}', '${encodeURIComponent(fileName)}', 'original')" style="background:#222;color:#4a9eff;border:1px solid #2a2a2a;padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer;">原图</button>` : '-';

            // 标注图预览按钮 - 点击直接显示标注图
            const annotBtn = d.annotated_path ?
                `<button class="img-preview" onclick="openModalDirect('${encodeURIComponent(d.image_path)}', '${encodeURIComponent(d.annotated_path)}', '${encodeURIComponent(fileName + ' - 标注')}', 'annotated')" style="background:#222;color:#4ade80;border:1px solid #2a2a2a;padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer;">标注</button>` : '-';
            const annotBtnDisabled = !d.annotated_path ?
                `<button style="background:#222;color:#555;border:1px solid #2a2a2a;padding:2px 8px;border-radius:4px;font-size:11px;cursor:not-allowed;" disabled>标注</button>` : '';

            const delBtn = `<button onclick="deleteRecord(${d.id}, this)" style="background:#2a2a2a;color:#f87171;border:1px solid #3a3a3a;padding:2px 8px;border-radius:4px;font-size:11px;cursor:pointer;" title="删除">🗑</button>`;

            return `<tr>
                <td>${d.id}</td>
                <td>${d.timestamp || '-'}</td>
                <td>${origBtn}</td>
                <td>${annotBtn || annotBtnDisabled}</td>
                <td>${d.class_name_cn || '-'}</td>
                <td>${d.confidence ? (d.confidence*100).toFixed(1)+'%' : '-'}</td>
                <td class="severity-${d.severity || ''}">${d.severity || '-'}</td>
                <td>${d.inference_time_ms ? d.inference_time_ms.toFixed(1) + 'ms' : '-'}</td>
                <td>${badge}</td>
                <td>${delBtn}</td>
            </tr>`;
        }).join('');

        // 分页
        const totalPages = Math.ceil(data.total / PAGE_SIZE);
        const pag = document.getElementById('historyPagination');
        let html = '';
        if (page > 1) html += `<button onclick="loadHistory(${page-1})">上一页</button>`;
        for (let i = Math.max(1, page-2); i <= Math.min(totalPages, page+2); i++) {
            html += `<button class="${i===page?'active':''}" onclick="loadHistory(${i})">${i}</button>`;
        }
        if (page < totalPages) html += `<button onclick="loadHistory(${page+1})">下一页</button>`;
        pag.innerHTML = html;
    });
}

function deleteRecord(id, btn) {
    if (!confirm('确定删除这条记录？')) return;
    fetch('/api/history/delete', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: id})
    }).then(r => r.json()).then(data => {
        if (data.ok) {
            // 删除行
            const row = btn.closest('tr');
            row.style.transition = 'opacity 0.3s';
            row.style.opacity = '0';
            setTimeout(() => row.remove(), 300);
        } else {
            alert('删除失败');
        }
    }).catch(() => alert('删除失败'));
}

// 直接打开模态框并显示指定类型的图片
function openModalDirect(encodedOriginal, encodedAnnotated, encodedTitle, showTab) {
    const original = decodeURIComponent(encodedOriginal || '');
    const annotated = decodeURIComponent(encodedAnnotated || '');
    const title = decodeURIComponent(encodedTitle || '图片预览');

    currentImagePath = original;
    currentAnnotatedPath = annotated;
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('imageModal').style.display = 'block';
    document.body.style.overflow = 'hidden';

    // 直接显示指定类型的图片
    const img = document.getElementById('modalImage');
    const noImg = document.querySelector('.no-image');

    // 更新标签样式
    document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));

    if (showTab === 'annotated' && annotated) {
        // 显示标注图
        img.src = '/api/image?path=' + encodeURIComponent(annotated);
        img.style.cssText = 'display:block !important; width:90vw !important; height:85vh !important; object-fit:contain; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5);';
        if (noImg) noImg.style.display = 'none';
        // 选中标注图标签
        document.querySelectorAll('.modal-tab')[1].classList.add('active');
    } else {
        // 显示原图
        img.src = '/api/image?path=' + encodeURIComponent(original);
        img.style.cssText = 'display:block !important; width:90vw !important; height:85vh !important; object-fit:contain; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5);';
        if (noImg) noImg.style.display = 'none';
        // 选中原图标签
        document.querySelectorAll('.modal-tab')[0].classList.add('active');
    }
}

// 报警配置
function loadConfig() {
    fetch('/api/alert_config').then(r => r.json()).then(data => {
        const grid = document.getElementById('configGrid');
        const cnNames = {
            'crazing':'龟裂','inclusion':'夹杂','patches':'斑块',
            'pitted_surface':'麻点','rolled-in_scale':'氧化皮','scratches':'划痕'
        };
        grid.innerHTML = Object.entries(data).map(([name, cfg]) => {
            const cn = cnNames[name] || name;
            const levels = cfg.severity_levels || {};
            return `<div class="config-item">
                <div class="name">${cn} (${name})</div>
                <label>
                    <input type="checkbox" ${cfg.enabled ? 'checked' : ''}
                        onchange="updateConfig('${name}', 'enabled', this.checked)"> 启用报警
                </label>
                <label>报警阈值: <span class="threshold-val" id="thresh-val-${name}">${cfg.threshold.toFixed(2)}</span></label>
                <input type="range" min="0.05" max="0.95" step="0.05" value="${cfg.threshold}"
                    oninput="document.getElementById('thresh-val-${name}').textContent=parseFloat(this.value).toFixed(2)"
                    onchange="updateConfig('${name}', 'threshold', parseFloat(this.value))">
                <label style="margin-top:8px;">严重程度阈值:</label>
                <div style="display:flex;gap:8px;align-items:center;margin-top:4px;">
                    <span style="color:#4ade80;font-size:11px;">轻微≥</span>
                    <input type="number" min="0.1" max="0.9" step="0.05" value="${levels['轻微']||0.25}"
                        style="width:50px;background:#333;color:#e0e0e0;border:1px solid #444;padding:2px 4px;border-radius:3px;font-size:11px;"
                        onchange="updateSeverityLevel('${name}', '轻微', parseFloat(this.value))">
                    <span style="color:#fbbf24;font-size:11px;">中等≥</span>
                    <input type="number" min="0.1" max="0.9" step="0.05" value="${levels['中等']||0.50}"
                        style="width:50px;background:#333;color:#e0e0e0;border:1px solid #444;padding:2px 4px;border-radius:3px;font-size:11px;"
                        onchange="updateSeverityLevel('${name}', '中等', parseFloat(this.value))">
                    <span style="color:#f87171;font-size:11px;">严重≥</span>
                    <input type="number" min="0.1" max="0.9" step="0.05" value="${levels['严重']||0.70}"
                        style="width:50px;background:#333;color:#e0e0e0;border:1px solid #444;padding:2px 4px;border-radius:3px;font-size:11px;"
                        onchange="updateSeverityLevel('${name}', '严重', parseFloat(this.value))">
                </div>
            </div>`;
        }).join('');
    });
}

function updateConfig(class_name, key, value) {
    fetch('/api/alert_config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({class_name, [key]: value})
    });
}

function updateSeverityLevel(class_name, level, value) {
    fetch('/api/alert_config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({class_name, severity_level: level, severity_value: value})
    });
}

// Chart image modal
function openChartModal(src) {
    if (!src || src === '') return;
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('modalImage');
    const title = document.getElementById('modalTitle');
    title.textContent = '可视化图表';
    img.src = src;
    img.style.cssText = 'display:block !important; width:90vw !important; height:85vh !important; object-fit:contain; border-radius:4px; box-shadow:0 4px 20px rgba(0,0,0,0.5);';
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden';
    document.querySelector('.no-image').style.display = 'none';
}

// 可视化图表实例
let vizCharts = {};

function destroyVizCharts() {
    Object.values(vizCharts).forEach(c => { if (c) c.destroy(); });
    vizCharts = {};
}

function refreshCharts() {
    const hours = document.getElementById('vizTimeRange').value;
    const params = hours === 'all' ? '' : '?hours=' + hours;

    // 并行加载所有数据
    Promise.all([
        fetch('/api/trend' + params).then(r => r.json()),
        fetch('/api/stats' + (hours === 'all' ? '?hours=99999' : params)).then(r => r.json()),
        fetch('/api/class_trend' + (hours === 'all' ? '?hours=99999' : params)).then(r => r.json()),
    ]).then(([trendData, statsData, classData]) => {
        destroyVizCharts();
        const chartOpts = {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#888', font: { size: 11 } } } },
            scales: {
                x: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1f1f1f' } },
                y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1f1f1f' }, beginAtZero: true }
            }
        };

        // 1. 缺陷趋势
        const labelFmt = (hours === 'all' || hours > 168) ? (t => t ? t.slice(5) : '') : (t => t ? t.split(' ')[1].slice(0,5) : '');
        vizCharts.trend = new Chart(document.getElementById('viz-trend').getContext('2d'), {
            type: 'line', data: {
                labels: trendData.map(d => labelFmt(d.time_bucket)),
                datasets: [{ label: '缺陷数', data: trendData.map(d => d.defects || 0), borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.1)', fill: true, tension: 0.4, pointRadius: 2 },
                           { label: '检测总数', data: trendData.map(d => d.total || 0), borderColor: '#4a9eff', backgroundColor: 'rgba(74,158,255,0.1)', fill: true, tension: 0.4, pointRadius: 2 }]
            }, options: chartOpts
        });

        // 2. 缺陷类型分布（饼图）
        const classStats = statsData.class_stats || [];
        const classColors = ['#f87171','#fbbf24','#4ade80','#a78bfa','#fb923c','#38bdf8'];
        vizCharts.classDist = new Chart(document.getElementById('viz-class-dist').getContext('2d'), {
            type: 'doughnut', data: {
                labels: classStats.map(c => c.class_name_cn),
                datasets: [{ data: classStats.map(c => c.cnt), backgroundColor: classColors, borderColor: '#1a1a1a', borderWidth: 2 }]
            }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#888', font: { size: 11 } } } } }
        });

        // 3. 合格/不合格（饼图）
        const passCount = statsData.pass_count || 0;
        const defectCount = statsData.defects || 0;
        vizCharts.passfail = new Chart(document.getElementById('viz-passfail').getContext('2d'), {
            type: 'doughnut', data: {
                labels: ['合格', '不合格'],
                datasets: [{ data: [passCount, defectCount], backgroundColor: ['#4ade80', '#f87171'], borderColor: '#1a1a1a', borderWidth: 2 }]
            }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#888', font: { size: 11 } } } } }
        });

        // 4. 每日检测量（堆叠柱状图 + 数字标签）
        fetch('/api/daily_stats' + (hours === 'all' ? '' : params)).then(r => r.json()).then(dailyData => {
            const dailyOpts = JSON.parse(JSON.stringify(chartOpts));
            dailyOpts.scales.x.stacked = true;
            dailyOpts.scales.y.stacked = true;
            dailyOpts.plugins.datalabels = {
                anchor: 'end', align: 'top', color: '#ccc', font: { size: 10, weight: 'bold' },
                formatter: (v) => v > 0 ? v : ''
            };
            // 堆叠：底部合格（绿），顶部缺陷（红）
            vizCharts.daily = new Chart(document.getElementById('viz-daily').getContext('2d'), {
                type: 'bar', data: {
                    labels: dailyData.map(d => d.date ? d.date.slice(5) : ''),
                    datasets: [
                        { label: '合格数', data: dailyData.map(d => d.pass_count || 0), backgroundColor: 'rgba(74,222,128,0.7)', borderColor: '#4ade80', borderWidth: 1 },
                        { label: '缺陷数', data: dailyData.map(d => d.defect_count || 0), backgroundColor: 'rgba(248,113,113,0.7)', borderColor: '#f87171', borderWidth: 1 }
                    ]
                }, options: dailyOpts,
                plugins: [ChartDataLabels]
            });
        }).catch(() => {});
    }).catch(err => console.error('图表加载失败:', err));
}

// Load charts on tab switch
const _origSwitchTab = switchTab;
switchTab = function(tab) {
    _origSwitchTab.call(null, tab);
    if (tab === 'visualization') refreshCharts();
};
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════
# 路由
# ══════════════════════════════════════════════

@app.route('/')
def index():
    resp = make_response(render_template_string(HTML_TEMPLATE))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/stats')
def api_stats():
    hours = request.args.get('hours', 24, type=int)
    return jsonify(db.get_stats(hours))


@app.route('/api/recent')
def api_recent():
    limit = request.args.get('limit', 50, type=int)
    return jsonify(db.get_recent(limit))


@app.route('/api/history')
def api_history():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    severity = request.args.get('severity', None)
    has_defect = request.args.get('has_defect', None)

    offset = (page - 1) * limit
    has_defect_bool = None
    if has_defect is not None:
        has_defect_bool = has_defect == '1'

    records = db.get_history(limit=limit, offset=offset, has_defect=has_defect_bool, severity=severity)

    # 获取总数
    conn = db.get_connection()
    conditions = []
    params = []
    if has_defect_bool is not None:
        conditions.append("has_defect = ?")
        params.append(int(has_defect_bool))
    if severity:
        conditions.append("severity = ?")
        params.append(severity)
    where = " AND ".join(conditions) if conditions else "1=1"
    total = conn.execute(f"SELECT COUNT(*) FROM detections WHERE {where}", params).fetchone()[0]
    conn.close()

    return jsonify({'records': records, 'total': total})


@app.route('/api/history/delete', methods=['POST'])
def api_history_delete():
    """删除单条检测记录"""
    data = request.json
    if not data or 'id' not in data:
        return jsonify({'error': 'missing id'}), 400
    ok = db.delete_detection(data['id'])
    return jsonify({'ok': ok})


@app.route('/api/trend')
def api_trend():
    hours = request.args.get('hours', 24, type=int)
    return jsonify(db.get_trend(hours))


@app.route('/api/daily_stats')
def api_daily_stats():
    hours = request.args.get('hours', 720, type=int)
    days = max(1, hours // 24)
    return jsonify(db.get_daily_stats(days))


@app.route('/api/class_trend')
def api_class_trend():
    hours = request.args.get('hours', 24, type=int)
    return jsonify(db.get_class_trend(hours))


@app.route('/api/alert_config', methods=['GET'])
def api_get_alert_config():
    return jsonify(db.get_all_alert_configs())


@app.route('/api/alert_config', methods=['POST'])
def api_set_alert_config():
    data = request.json
    if not data:
        return jsonify({'error': 'no data'}), 400

    class_name = data.get('class_name')
    if not class_name:
        return jsonify({'error': 'no class_name'}), 400

    enabled = data.get('enabled')
    threshold = data.get('threshold')

    # 处理严重程度级别更新
    severity_level = data.get('severity_level')
    severity_value = data.get('severity_value')
    severity_levels = None

    if severity_level and severity_value is not None:
        # 获取现有配置
        existing = db.get_all_alert_configs()
        levels = existing.get(class_name, {}).get('severity_levels', {})
        levels[severity_level] = severity_value
        severity_levels = levels

    db.update_alert_config(class_name, enabled=enabled, threshold=threshold, severity_levels=severity_levels)
    return jsonify({'ok': True})


@app.route('/api/push', methods=['POST'])
def api_push():
    """接收检测结果推送（从桌面应用调用）"""
    data = request.json
    if not data:
        return jsonify({'error': 'no data'}), 400

    # 保存到数据库
    detections_data = data.get('detections', [])
    detections = []
    for d in detections_data:
        det = type('Det', (), {
            'class_name': d.get('class_name', ''),
            'class_name_cn': d.get('class_name_cn', ''),
            'confidence': d.get('confidence', 0),
            'bbox': d.get('bbox', [0, 0, 0, 0]),
            'is_defect': d.get('is_defect', True),
        })()
        detections.append(det)

    db.save_detection(
        image_path=data.get('image_path', ''),
        inference_time_ms=data.get('inference_time_ms', 0),
        source_type=data.get('source_type', 'image'),
        annotated_path=data.get('annotated_path'),
        detections=detections,
    )

    # 推送给所有 WebSocket 客户端（补充时间、判定、严重程度）
    stats = db.get_stats(24)
    has_defect = any(d.get('is_defect', True) for d in detections_data)
    overall_severity, _ = db.compute_overall_severity(detections)
    enriched = dict(data)
    enriched['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    enriched['has_defect'] = has_defect
    enriched['severity'] = overall_severity
    socketio.emit('new_detection', {
        'detection': enriched,
        'stats': stats,
    })

    return jsonify({'ok': True})


@app.route('/api/review_stats')
def api_review_stats():
    """获取人工复核统计"""
    return jsonify(db.get_review_stats())


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """接收反馈事件推送（语音/日志/视觉）"""
    data = request.json
    if not data:
        return jsonify({'error': 'no data'}), 400

    # 通过 WebSocket 推送给所有客户端
    socketio.emit('feedback_event', data)

    return jsonify({'ok': True})


@app.route('/api/image')
def api_image():
    """提供图片文件访问"""
    image_path = request.args.get('path', '')
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': '图片不存在'}), 404

    try:
        return send_file(image_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════


@app.route('/api/chart')
def api_chart():
    """提供可视化图表文件访问"""
    name = request.args.get('name', '')
    allowed = ['confusion_matrix', 'defect_distribution', 'confidence_distribution',
               'inference_time', 'pass_fail_summary', 'defect_bar_chart']
    if name not in allowed:
        return jsonify({'error': 'not allowed'}), 400
    chart_path = os.path.join('runs', 'demo', 'visualization', f'{name}.png')
    if not os.path.exists(chart_path):
        return jsonify({'error': '图表不存在，请先运行 demo.py 生成图表'}), 404
    return send_file(chart_path)


def start_web(port=5000, debug=False):
    """启动 Web 监控服务"""
    print(f"[Web 监控] 启动中... http://0.0.0.0:{port}")
    print(f"[Web 监控] 局域网内其他设备可通过 http://<本机IP>:{port} 访问")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    start_web()
