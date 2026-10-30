import os
import sys
import io
import csv
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

# Ensure parent directory is on sys.path so rag_core can be imported
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from rag_core import parse_upload, chunk_text, retrieve, answer_from_chunks
except ImportError as e:
    raise ImportError(f"Failed to import rag_core from {BASE_DIR}: {e}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

DATASET_PATH = BASE_DIR / 'data' / 'dataset.csv'

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuRAG | Intelligent Document Q&A & Analysis</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .answer-gradient {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(59, 130, 246, 0.03) 100%);
            border-left: 4px solid #3b82f6;
        }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-4 py-3.5 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-2xl">📚</span>
                <div>
                    <h1 class="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                        DocuRAG <span class="text-xs px-2 py-0.5 rounded-full bg-blue-900/60 text-blue-300 border border-blue-700/50">Vercel Serverless Ready</span>
                    </h1>
                    <p class="text-xs text-slate-400">Intelligent Document Question-Answering & Analysis</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span id="doc-count-badge" class="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">0 Documents</span>
                <span id="chunk-count-badge" class="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">0 Chunks</span>
                <a href="https://github.com/taslimzafar/RAG_Based_DocumentAnalyser" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-medium border border-slate-700 transition">
                    GitHub Repo ↗
                </a>
            </div>
        </div>
    </header>

    <!-- Main Container -->
    <main class="max-w-7xl mx-auto px-4 py-6 flex-1 w-full flex flex-col lg:flex-row gap-6">

        <!-- Sidebar / Controls (Left) -->
        <aside class="w-full lg:w-80 flex-shrink-0 space-y-5">
            <!-- Upload Box -->
            <div class="glass-card rounded-xl p-5 shadow-lg">
                <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">📁 Upload Documents</h2>
                <div class="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-lg p-5 text-center cursor-pointer transition bg-slate-900/50" onclick="document.getElementById('file-input').click()">
                    <span class="text-3xl block mb-2">📄</span>
                    <p class="text-sm font-medium text-slate-200">Click to upload files</p>
                    <p class="text-xs text-slate-400 mt-1">PDF, DOCX, TXT, or MD</p>
                    <input type="file" id="file-input" multiple accept=".pdf,.docx,.txt,.md" class="hidden" onchange="handleFileUpload(event)">
                </div>
                <div id="upload-status" class="mt-3 text-xs text-slate-400 hidden"></div>
            </div>

            <!-- RAG Settings -->
            <div class="glass-card rounded-xl p-5 shadow-lg space-y-4">
                <h2 class="text-sm font-semibold uppercase tracking-wider text-slate-400">⚙️ Engine Parameters</h2>
                <div>
                    <div class="flex justify-between text-xs text-slate-300 mb-1">
                        <span>Top-K Sources</span>
                        <span id="top-k-val" class="font-bold text-blue-400">4</span>
                    </div>
                    <input type="range" id="top-k" min="1" max="8" value="4" class="w-full accent-blue-500 cursor-pointer" oninput="document.getElementById('top-k-val').innerText = this.value">
                </div>
                <div>
                    <div class="flex justify-between text-xs text-slate-300 mb-1">
                        <span>Chunk Size (words)</span>
                        <span id="chunk-size-val" class="font-bold text-blue-400">180</span>
                    </div>
                    <input type="range" id="chunk-size" min="80" max="300" step="20" value="180" class="w-full accent-blue-500 cursor-pointer" oninput="document.getElementById('chunk-size-val').innerText = this.value">
                </div>
                <div>
                    <div class="flex justify-between text-xs text-slate-300 mb-1">
                        <span>Chunk Overlap (words)</span>
                        <span id="overlap-val" class="font-bold text-blue-400">35</span>
                    </div>
                    <input type="range" id="chunk-overlap" min="10" max="60" step="5" value="35" class="w-full accent-blue-500 cursor-pointer" oninput="document.getElementById('overlap-val').innerText = this.value">
                </div>
            </div>

            <div class="p-4 rounded-xl bg-blue-950/40 border border-blue-800/40 text-xs text-blue-200 leading-relaxed">
                🔒 <strong>100% Stateless & Private:</strong> Uploads and retrieval operate directly in memory without writing to permanent storage.
            </div>
        </aside>

        <!-- Right Main Viewport -->
        <section class="flex-1 flex flex-col space-y-5">
            <!-- Tabs Header -->
            <div class="flex border-b border-slate-800 space-x-1 sm:space-x-3 text-sm font-medium">
                <button onclick="switchTab('qa')" id="tab-btn-qa" class="px-4 py-2.5 text-blue-400 border-b-2 border-blue-500 transition">💬 Document Q&A</button>
                <button onclick="switchTab('inspector')" id="tab-btn-inspector" class="px-4 py-2.5 text-slate-400 hover:text-slate-200 transition">🔍 Chunk Inspector</button>
                <button onclick="switchTab('dataset')" id="tab-btn-dataset" class="px-4 py-2.5 text-slate-400 hover:text-slate-200 transition">📊 Dataset Evaluation</button>
                <button onclick="switchTab('about')" id="tab-btn-about" class="px-4 py-2.5 text-slate-400 hover:text-slate-200 transition">ℹ️ Architecture</button>
            </div>

            <!-- TAB 1: Document Q&A -->
            <div id="tab-qa" class="space-y-5">
                <div id="no-docs-banner" class="glass-card rounded-xl p-6 text-center py-10">
                    <span class="text-4xl block mb-3">👈</span>
                    <h3 class="text-base font-semibold text-slate-200">No Documents Uploaded Yet</h3>
                    <p class="text-sm text-slate-400 max-w-md mx-auto mt-1">
                        Upload one or more PDF, DOCX, TXT, or Markdown documents in the left sidebar to begin asking questions.
                    </p>
                </div>

                <div id="qa-interface" class="hidden space-y-5">
                    <!-- Quick Prompt Pills -->
                    <div>
                        <span class="text-xs text-slate-400 block mb-2 font-medium">Quick suggestions:</span>
                        <div class="flex flex-wrap gap-2">
                            <button onclick="setPrompt('What is the primary purpose and scope of this document?')" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-full transition">📌 Summarize purpose</button>
                            <button onclick="setPrompt('What are the key findings, results, or conclusions?')" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-full transition">🔑 Key findings</button>
                            <button onclick="setPrompt('What requirements, conditions, or constraints are specified?')" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-full transition">📋 Requirements</button>
                            <button onclick="setPrompt('What are the recommended action items, next steps, or policies?')" class="text-xs bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-full transition">⚡ Action items</button>
                        </div>
                    </div>

                    <!-- Search Input Bar -->
                    <div class="flex gap-2">
                        <input type="text" id="query-input" placeholder="Ask a question about your documents..." class="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-sm text-white focus:outline-none focus:border-blue-500 placeholder-slate-500 shadow-inner" onkeydown="if(event.key==='Enter') executeQuery()">
                        <button onclick="executeQuery()" id="btn-search" class="bg-blue-600 hover:bg-blue-500 text-white font-medium px-5 py-3 rounded-lg text-sm transition shadow-lg flex items-center gap-2">
                            <span>🚀</span> <span>Get Answer</span>
                        </button>
                        <button onclick="document.getElementById('query-input').value=''" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-4 py-3 rounded-lg text-sm transition">
                            🧹
                        </button>
                    </div>

                    <!-- Loading Spinner -->
                    <div id="query-loader" class="hidden text-center py-6 text-slate-400 text-sm">
                        <div class="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mb-2"></div>
                        <p>Scanning index and synthesizing extractive answer...</p>
                    </div>

                    <!-- Answer Section -->
                    <div id="answer-section" class="hidden space-y-4">
                        <div class="answer-gradient rounded-xl p-5 shadow-lg">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-xs font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                                    💡 Synthesized Extractive Answer
                                </span>
                                <span id="confidence-badge" class="text-xs px-2.5 py-0.5 rounded-full font-semibold"></span>
                            </div>
                            <p id="answer-text" class="text-base text-slate-100 leading-relaxed font-normal"></p>
                        </div>

                        <!-- Retrieved Passages Accordion -->
                        <div>
                            <h4 class="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                                📖 Retrieved Grounding Passages
                            </h4>
                            <div id="sources-list" class="space-y-3"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- TAB 2: Chunk Inspector -->
            <div id="tab-inspector" class="hidden space-y-5">
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Total Files</span>
                        <div id="insp-files" class="text-xl font-bold text-white mt-1">0</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Total Chunks</span>
                        <div id="insp-chunks" class="text-xl font-bold text-blue-400 mt-1">0</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Total Words</span>
                        <div id="insp-words" class="text-xl font-bold text-emerald-400 mt-1">0</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Avg Words/Chunk</span>
                        <div id="insp-avg" class="text-xl font-bold text-purple-400 mt-1">0</div>
                    </div>
                </div>

                <div class="glass-card rounded-xl p-5">
                    <div class="flex items-center justify-between mb-3">
                        <h4 class="text-sm font-semibold text-slate-200">Chunk Explorer</h4>
                        <span id="chunk-index-indicator" class="text-xs text-slate-400">Chunk 0 of 0</span>
                    </div>
                    <input type="range" id="chunk-slider" min="1" max="1" value="1" class="w-full accent-blue-500 mb-4" oninput="showChunk(this.value)">
                    <div id="chunk-content-box" class="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono text-xs text-slate-300 whitespace-pre-wrap max-h-80 overflow-y-auto">
                        No chunks available yet. Upload files to preview text.
                    </div>
                </div>
            </div>

            <!-- TAB 3: Dataset Evaluation -->
            <div id="tab-dataset" class="hidden space-y-6">
                <!-- KPI Cards -->
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Evaluation Records</span>
                        <div id="kpi-rows" class="text-xl font-bold text-white mt-1">500</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Unique Documents</span>
                        <div id="kpi-docs" class="text-xl font-bold text-blue-400 mt-1">-</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Mean Retrieval Score</span>
                        <div id="kpi-retrieval" class="text-xl font-bold text-emerald-400 mt-1">-</div>
                    </div>
                    <div class="glass-card rounded-xl p-4 text-center">
                        <span class="text-xs text-slate-400">Mean Quality Score</span>
                        <div id="kpi-quality" class="text-xl font-bold text-amber-400 mt-1">-</div>
                    </div>
                </div>

                <!-- Charts Container -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div class="glass-card rounded-xl p-5">
                        <h4 class="text-sm font-semibold text-slate-200 mb-3">Rows by Document Type</h4>
                        <div class="h-64"><canvas id="chart-doc-type"></canvas></div>
                    </div>
                    <div class="glass-card rounded-xl p-5">
                        <h4 class="text-sm font-semibold text-slate-200 mb-3">Mean Quality by Query Type</h4>
                        <div class="h-64"><canvas id="chart-query-type"></canvas></div>
                    </div>
                </div>
            </div>

            <!-- TAB 4: Architecture -->
            <div id="tab-about" class="hidden space-y-5">
                <div class="glass-card rounded-xl p-6 space-y-4">
                    <h3 class="text-base font-bold text-white">🏗️ How DocuRAG Works on Vercel & Locally</h3>
                    <p class="text-sm text-slate-300 leading-relaxed">
                        DocuRAG executes a 100% deterministic, dependency-light lexical Retrieval-Augmented Generation pipeline. 
                        On Vercel, it runs as an ultra-fast serverless application that operates completely in-memory without persistent state bottlenecks.
                    </p>
                    <div class="bg-slate-900 border border-slate-800 rounded-lg p-4 font-mono text-xs text-blue-300 leading-relaxed">
Documents (PDF, Word, TXT, MD) ──► In-Memory Parser ──► Sliding-Window Chunker ──► Inverted Index
                                                                                         │
Question ──► Stopword Filter & Token Normalizer ──► Dual-Factor Overlap Scorer ──────────┘
                                                          │
                                             Top-K Ranked Passages
                                                          │
                                        Extractive Sentence Grounding
                    </div>
                    <h4 class="text-sm font-semibold text-slate-200 mt-4">Scoring Heuristic:</h4>
                    <p class="text-xs text-slate-400">
                        Passages are ranked by combining <strong>85% Token Coverage</strong> (percentage of unique query terms contained in the chunk) 
                        and <strong>15% Token Density</strong> (concentration of terms relative to chunk length).
                    </p>
                </div>
            </div>
        </section>
    </main>

    <!-- Client-Side State & Logic -->
    <script>
        let currentChunks = [];
        let currentStats = {};
        let datasetLoaded = false;
        let chart1 = null, chart2 = null;

        function switchTab(tabId) {
            ['qa', 'inspector', 'dataset', 'about'].forEach(t => {
                document.getElementById('tab-' + t).classList.add('hidden');
                document.getElementById('tab-btn-' + t).className = 'px-4 py-2.5 text-slate-400 hover:text-slate-200 transition';
            });
            document.getElementById('tab-' + tabId).classList.remove('hidden');
            document.getElementById('tab-btn-' + tabId).className = 'px-4 py-2.5 text-blue-400 border-b-2 border-blue-500 font-semibold transition';

            if (tabId === 'dataset' && !datasetLoaded) {
                loadDatasetAnalytics();
            }
        }

        async function handleFileUpload(event) {
            const files = event.target.files;
            if (!files || files.length === 0) return;

            const status = document.getElementById('upload-status');
            status.innerText = "Indexing documents...";
            status.classList.remove('hidden');

            const formData = new FormData();
            for (let f of files) formData.append('files', f);
            formData.append('chunk_size', document.getElementById('chunk-size').value);
            formData.append('chunk_overlap', document.getElementById('chunk-overlap').value);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.success) {
                    currentChunks = data.chunks;
                    currentStats = data.file_stats;

                    document.getElementById('doc-count-badge').innerText = Object.keys(currentStats).length + " Documents";
                    document.getElementById('chunk-count-badge').innerText = currentChunks.length + " Chunks";

                    document.getElementById('no-docs-banner').classList.add('hidden');
                    document.getElementById('qa-interface').classList.remove('hidden');

                    status.innerText = `✓ Processed ${Object.keys(currentStats).length} file(s) into ${currentChunks.length} chunks.`;
                    status.className = "mt-3 text-xs text-emerald-400";

                    updateInspector();
                } else {
                    status.innerText = "Error: " + data.error;
                    status.className = "mt-3 text-xs text-rose-400";
                }
            } catch (err) {
                status.innerText = "Upload failed: " + err.message;
                status.className = "mt-3 text-xs text-rose-400";
            }
        }

        function setPrompt(txt) {
            document.getElementById('query-input').value = txt;
            executeQuery();
        }

        async function executeQuery() {
            const query = document.getElementById('query-input').value.trim();
            if (!query || currentChunks.length === 0) return;

            const loader = document.getElementById('query-loader');
            const answerSection = document.getElementById('answer-section');
            loader.classList.remove('hidden');
            answerSection.classList.add('hidden');

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        chunks: currentChunks,
                        top_k: parseInt(document.getElementById('top-k').value),
                        max_sentences: 5
                    })
                });
                const data = await res.json();
                loader.classList.add('hidden');

                if (data.success) {
                    answerSection.classList.remove('hidden');
                    document.getElementById('answer-text').innerText = data.answer;

                    // Confidence badge
                    const badge = document.getElementById('confidence-badge');
                    const score = data.top_score;
                    if (score >= 0.5) {
                        badge.className = "text-xs px-2.5 py-0.5 rounded-full font-semibold bg-emerald-950 text-emerald-300 border border-emerald-700/60";
                        badge.innerText = `High Confidence (${score.toFixed(3)})`;
                    } else if (score >= 0.25) {
                        badge.className = "text-xs px-2.5 py-0.5 rounded-full font-semibold bg-amber-950 text-amber-300 border border-amber-700/60";
                        badge.innerText = `Moderate Match (${score.toFixed(3)})`;
                    } else {
                        badge.className = "text-xs px-2.5 py-0.5 rounded-full font-semibold bg-rose-950 text-rose-300 border border-rose-700/60";
                        badge.innerText = `Weak Match (${score.toFixed(3)})`;
                    }

                    // Render sources
                    const sourcesContainer = document.getElementById('sources-list');
                    sourcesContainer.innerHTML = '';
                    data.hits.forEach((hit, idx) => {
                        const div = document.createElement('div');
                        div.className = "glass-card rounded-lg p-4 text-xs space-y-2";
                        div.innerHTML = `
                            <div class="flex justify-between items-center">
                                <span class="font-semibold text-slate-200">Excerpt ${idx + 1} · ${hit.source} (Chunk #${hit.chunk_id})</span>
                                <span class="text-blue-400 font-mono">Relevance: ${hit.score.toFixed(3)}</span>
                            </div>
                            <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div class="bg-blue-500 h-1.5" style="width: ${Math.min(100, hit.score * 100)}%"></div>
                            </div>
                            <p class="text-slate-300 italic">"${hit.text}"</p>
                        `;
                        sourcesContainer.appendChild(div);
                    });
                }
            } catch (err) {
                loader.classList.add('hidden');
                alert("Query failed: " + err.message);
            }
        }

        function updateInspector() {
            const files = Object.keys(currentStats);
            let totWords = 0;
            files.forEach(f => totWords += currentStats[f].word_count || 0);

            document.getElementById('insp-files').innerText = files.length;
            document.getElementById('insp-chunks').innerText = currentChunks.length;
            document.getElementById('insp-words').innerText = totWords.toLocaleString();
            document.getElementById('insp-avg').innerText = currentChunks.length ? Math.round(totWords / currentChunks.length) : 0;

            const slider = document.getElementById('chunk-slider');
            if (currentChunks.length > 0) {
                slider.max = currentChunks.length;
                slider.value = 1;
                showChunk(1);
            }
        }

        function showChunk(val) {
            const idx = parseInt(val) - 1;
            if (idx >= 0 && idx < currentChunks.length) {
                const chunk = currentChunks[idx];
                document.getElementById('chunk-index-indicator').innerText = `Chunk ${idx + 1} of ${currentChunks.length}`;
                document.getElementById('chunk-content-box').innerText = `[Source: ${chunk.source} | Chunk #${chunk.chunk_id}]\n\n${chunk.text}`;
            }
        }

        async function loadDatasetAnalytics() {
            try {
                const res = await fetch('/api/dataset');
                const data = await res.json();
                if (data.success) {
                    datasetLoaded = true;
                    const m = data.metrics;
                    document.getElementById('kpi-rows').innerText = m.total_rows.toLocaleString();
                    document.getElementById('kpi-docs').innerText = m.unique_docs.toLocaleString();
                    document.getElementById('kpi-retrieval').innerText = m.mean_retrieval.toFixed(3);
                    document.getElementById('kpi-quality').innerText = m.mean_quality.toFixed(3);

                    // Render Chart 1 (Doc Type counts)
                    const ctx1 = document.getElementById('chart-doc-type').getContext('2d');
                    if (chart1) chart1.destroy();
                    chart1 = new Chart(ctx1, {
                        type: 'bar',
                        data: {
                            labels: Object.keys(m.doc_type_counts),
                            datasets: [{
                                label: 'Rows',
                                data: Object.values(m.doc_type_counts),
                                backgroundColor: '#3b82f6',
                                borderRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                            }
                        }
                    });

                    // Render Chart 2 (Query Type quality)
                    const ctx2 = document.getElementById('chart-query-type').getContext('2d');
                    if (chart2) chart2.destroy();
                    chart2 = new Chart(ctx2, {
                        type: 'bar',
                        data: {
                            labels: Object.keys(m.query_type_quality),
                            datasets: [{
                                label: 'Mean Quality',
                                data: Object.values(m.query_type_quality),
                                backgroundColor: '#10b981',
                                borderRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                            }
                        }
                    });
                }
            } catch (err) {
                console.error("Failed to load dataset", err);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        files = request.files.getlist('files')
        chunk_size = int(request.form.get('chunk_size', 180))
        chunk_overlap = int(request.form.get('chunk_overlap', 35))

        all_chunks = []
        file_stats = {}

        for f in files:
            name = f.filename
            data = f.read()
            text = parse_upload(name, data)
            doc_chunks = chunk_text(text, name, size=chunk_size, overlap=chunk_overlap)
            all_chunks.extend(doc_chunks)
            file_stats[name] = {
                "word_count": len(text.split()),
                "chunk_count": len(doc_chunks),
                "char_count": len(text)
            }

        return jsonify({
            "success": True,
            "chunks": all_chunks,
            "file_stats": file_stats
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        payload = request.get_json(force=True) or {}
        query = payload.get('query', '').strip()
        chunks = payload.get('chunks', [])
        top_k = int(payload.get('top_k', 4))
        max_sentences = int(payload.get('max_sentences', 5))

        if not query or not chunks:
            return jsonify({
                "success": True,
                "answer": "Please enter a question and make sure documents are uploaded.",
                "hits": [],
                "top_score": 0.0
            })

        hits = retrieve(query, chunks, top_k=top_k)
        if not hits or hits[0]["score"] <= 0:
            return jsonify({
                "success": True,
                "answer": "No relevant passages were found in the uploaded documents.",
                "hits": [],
                "top_score": 0.0
            })

        answer = answer_from_chunks(query, hits, max_sentences=max_sentences)
        return jsonify({
            "success": True,
            "answer": answer,
            "hits": hits,
            "top_score": hits[0]["score"]
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    try:
        if not DATASET_PATH.exists():
            return jsonify({"success": False, "error": "dataset.csv not found"}), 404

        rows = []
        with open(DATASET_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        total_rows = len(rows)
        doc_ids = set()
        doc_type_counts = {}
        query_type_scores = {}
        query_type_counts = {}
        total_retrieval = 0.0
        total_quality = 0.0

        for r in rows:
            doc_ids.add(r.get('doc_id', ''))
            dtype = r.get('doc_type', 'unknown')
            doc_type_counts[dtype] = doc_type_counts.get(dtype, 0) + 1

            qtype = r.get('query_type', 'unknown')
            ret_score = float(r.get('retrieval_score', 0.0))
            qual_score = float(r.get('answer_quality_score', 0.0))

            total_retrieval += ret_score
            total_quality += qual_score

            query_type_scores[qtype] = query_type_scores.get(qtype, 0.0) + qual_score
            query_type_counts[qtype] = query_type_counts.get(qtype, 0) + 1

        query_type_quality = {}
        for q, total in query_type_scores.items():
            query_type_quality[q] = total / max(1, query_type_counts.get(q, 1))

        return jsonify({
            "success": True,
            "metrics": {
                "total_rows": total_rows,
                "unique_docs": len(doc_ids),
                "mean_retrieval": total_retrieval / max(1, total_rows),
                "mean_quality": total_quality / max(1, total_rows),
                "doc_type_counts": doc_type_counts,
                "query_type_quality": query_type_quality
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
