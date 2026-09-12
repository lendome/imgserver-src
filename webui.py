"""Admin WebUI module - standalone Flask app for server administration."""

import logging
from flask import Flask, jsonify, render_template_string, request

logger = logging.getLogger(__name__)

_webui_app = None


def create_webui_app() -> Flask:
    """Create the admin webui Flask application."""
    global _webui_app

    if _webui_app is not None:
        return _webui_app

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/api/status")
    def api_status():
        from .server_state import server_state
        from .vram import vram_manager
        from .gpu_lock import gpu_lock
        from .pipelines import get_loaded_pipelines, get_active_checkpoint
        from .classifiers import classifier_registry
        from .queue import get_queue
        from .config import get_config

        config = get_config()
        queue = get_queue()
        queue_status = queue.get_queue_status()
        loaded = get_loaded_pipelines()

        vram_total = vram_manager.get_total()
        vram_used = vram_manager.get_used()

        return jsonify({
            "server": {
                "uptime_seconds": round(server_state.uptime, 2),
                "generation_count": server_state.generation_count,
                "error_count": server_state.error_count,
                "idle_seconds": round(server_state.idle_time, 2),
                "current_task": server_state.current_task or None,
            },
            "gpu": {
                "busy": gpu_lock.is_busy(),
                "holder": gpu_lock.current_holder,
            },
            "vram": {
                "free_gb": round(vram_manager.get_free() / (1024**3), 2),
                "used_gb": round(vram_used / (1024**3), 2),
                "total_gb": round(vram_total / (1024**3), 2),
                "used_percent": round(vram_used / vram_total * 100, 1) if vram_total > 0 else 0,
            },
            "pipelines": {
                "loaded": list(loaded.keys()),
                "details": {
                    name: {
                        "is_loaded": p.is_loaded,
                        "checkpoint": get_active_checkpoint(name.split(":")[0]),
                    }
                    for name, p in loaded.items()
                },
            },
            "classifiers": {
                "current": classifier_registry.current_name,
                "available": classifier_registry.available,
            },
            "queue": queue_status,
            "config": {
                "default_checkpoint": config.default_checkpoint,
                "available_checkpoints": config.get_available_checkpoints(),
                "default_width": config.default_width,
                "default_height": config.default_height,
                "default_steps": config.default_steps,
                "default_cfg_scale": config.default_cfg_scale,
            },
        })

    @app.route("/api/unload", methods=["POST"])
    def api_unload():
        from .pipelines import unload_all, unload_pipeline
        data = request.get_json() or {}
        pipeline_name = data.get("pipeline")

        if pipeline_name:
            result = unload_pipeline(pipeline_name)
            return jsonify({"unloaded": result, "pipeline": pipeline_name})
        else:
            count = unload_all()
            return jsonify({"unloaded": count})

    @app.route("/api/queue/cancel", methods=["POST"])
    def api_queue_cancel():
        from .queue import get_queue
        from .abort import abort_controller
        data = request.get_json() or {}
        job_id = data.get("job_id")

        if not job_id:
            return jsonify({"error": "job_id required"}), 400

        queue = get_queue()
        success = queue.cancel_job(job_id, abort_controller)
        return jsonify({"cancelled": success, "job_id": job_id})

    @app.route("/api/queue/clear", methods=["POST"])
    def api_queue_clear():
        from .queue import get_queue
        import time
        queue = get_queue()

        with queue._queue_lock:
            cancelled = 0
            for job_id in list(queue._queue):
                job = queue._jobs.get(job_id)
                if job and job.status == "queued":
                    job.status = "cancelled"
                    job.completed_at = time.time()
                    cancelled += 1
            queue._queue.clear()
            queue._update_positions()

        return jsonify({"cleared": cancelled})

    @app.route("/api/server/reset-stats", methods=["POST"])
    def api_reset_stats():
        from .server_state import server_state
        server_state.generation_count = 0
        server_state.error_count = 0
        return jsonify({"reset": True})

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.exception("WebUI error")
        return jsonify({"error": str(e)}), 500

    _webui_app = app
    logger.info("Admin WebUI created")
    return app


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Server Admin</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { font-size: 1.8rem; margin-bottom: 20px; color: #00d4ff; display: flex; align-items: center; gap: 10px; }
        h2 { font-size: 1.1rem; margin: 24px 0 10px; color: #00d4ff; border-bottom: 1px solid #0f3460; padding-bottom: 6px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
        .card { background: #16213e; border-radius: 8px; padding: 16px; border: 1px solid #0f3460; }
        .card h3 { font-size: 0.78rem; color: #666; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .stat { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid #0f3460; font-size: 0.9rem; }
        .stat:last-of-type { border-bottom: none; }
        .stat-label { color: #aaa; }
        .stat-value { font-weight: 600; color: #00d4ff; text-align: right; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .stat-value.warn  { color: #ff9800; }
        .stat-value.bad   { color: #f44336; }
        .stat-value.good  { color: #4caf50; }

        .progress-bar { width: 100%; height: 14px; background: #0f3460; border-radius: 7px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00d4ff, #00ff88); transition: width 0.4s; border-radius: 7px; }
        .progress-fill.high     { background: linear-gradient(90deg, #ff9800, #ff5722); }
        .progress-fill.critical { background: linear-gradient(90deg, #f44336, #d32f2f); }

        .btn { padding: 7px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.82rem; font-weight: 600; transition: opacity 0.15s; }
        .btn:hover { opacity: 0.85; }
        .btn-danger  { background: #c62828; color: #fff; }
        .btn-neutral { background: #0f3460; color: #eee; border: 1px solid #1a4a7a; }
        .actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }

        .badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.72rem; font-weight: 700; }
        .badge-queued     { background: #e65100; color: #fff; }
        .badge-processing { background: #0277bd; color: #fff; }
        .badge-done       { background: #2e7d32; color: #fff; }
        .badge-failed     { background: #b71c1c; color: #fff; }
        .badge-cancelled  { background: #424242; color: #ccc; }

        .job-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
        .job { background: #0d1b2e; border: 1px solid #0f3460; border-radius: 5px; padding: 10px; font-size: 0.83rem; }
        .job-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .job-id { font-family: monospace; color: #7ec8e3; }
        .job-prompt { color: #aaa; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .job-meta { color: #666; font-size: 0.75rem; display: flex; gap: 10px; flex-wrap: wrap; }
        .job-error { color: #f44336; margin-top: 4px; font-size: 0.78rem; }
        .empty { color: #555; font-style: italic; font-size: 0.85rem; padding: 8px 0; }

        .chip { display: inline-block; background: #0f3460; border-radius: 4px; padding: 3px 9px; font-size: 0.8rem; margin: 2px; }

        .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #4caf50; flex-shrink: 0; }
        .dot.off { background: #555; }

        .refresh-row { display: flex; align-items: center; gap: 10px; font-size: 0.85rem; color: #888; margin-bottom: 4px; }
        .last-update { font-size: 0.8rem; color: #555; }
    </style>
</head>
<body>
<div class="container">
    <h1>
        <span>Image Server Admin</span>
        <span id="live-dot" class="dot off" title="Live updates"></span>
    </h1>

    <div class="refresh-row">
        <label><input type="checkbox" id="autoRefresh" checked> Auto-refresh (3s)</label>
        <button class="btn btn-neutral" onclick="fetchStatus()">Refresh now</button>
        <span class="last-update" id="lastUpdate"></span>
    </div>

    <h2>Server</h2>
    <div class="grid">
        <div class="card">
            <h3>Statistics</h3>
            <div class="stat"><span class="stat-label">Uptime</span><span class="stat-value" id="uptime">-</span></div>
            <div class="stat"><span class="stat-label">Generations</span><span class="stat-value" id="generations">-</span></div>
            <div class="stat"><span class="stat-label">Errors</span><span class="stat-value" id="errors">-</span></div>
            <div class="stat"><span class="stat-label">Idle Time</span><span class="stat-value" id="idleTime">-</span></div>
            <div class="stat"><span class="stat-label">Current Task</span><span class="stat-value" id="currentTask">-</span></div>
            <div class="stat"><span class="stat-label">GPU</span><span class="stat-value" id="gpuStatus">-</span></div>
            <div class="actions">
                <button class="btn btn-danger" onclick="resetStats()">Reset Stats</button>
            </div>
        </div>

        <div class="card">
            <h3>VRAM</h3>
            <div class="stat"><span class="stat-label">Used</span><span class="stat-value" id="vramUsed">-</span></div>
            <div class="stat"><span class="stat-label">Free</span><span class="stat-value" id="vramFree">-</span></div>
            <div class="stat"><span class="stat-label">Total</span><span class="stat-value" id="vramTotal">-</span></div>
            <div class="progress-bar"><div class="progress-fill" id="vramBar" style="width:0%"></div></div>
        </div>

        <div class="card">
            <h3>Loaded Pipelines</h3>
            <div id="pipelines"><span class="empty">None loaded</span></div>
            <div class="actions">
                <button class="btn btn-danger" onclick="unloadAll()">Unload All</button>
            </div>
        </div>

        <div class="card">
            <h3>Classifiers</h3>
            <div class="stat"><span class="stat-label">Active</span><span class="stat-value" id="currentClassifier">-</span></div>
            <div style="margin-top:10px;"><span style="color:#666;font-size:0.8rem;">Available</span><div id="availableClassifiers" style="margin-top:4px;"></div></div>
        </div>
    </div>

    <h2>Queue</h2>
    <div style="display:flex;gap:14px;margin-bottom:10px;">
        <div class="card" style="padding:12px;min-width:120px;">
            <h3>Queued</h3>
            <div style="font-size:1.8rem;font-weight:700;color:#ff9800;" id="queueDepth">-</div>
        </div>
        <div class="card" style="padding:12px;min-width:120px;">
            <h3>Processing</h3>
            <div style="font-size:1.8rem;font-weight:700;color:#00d4ff;" id="processingCount">-</div>
        </div>
    </div>
    <div class="actions" style="margin-bottom:12px;">
        <button class="btn btn-danger" onclick="clearQueue()">Clear Queue</button>
    </div>
    <div class="grid">
        <div class="card">
            <h3>Queued Jobs</h3>
            <div class="job-list" id="queuedJobs"><span class="empty">Empty</span></div>
        </div>
        <div class="card">
            <h3>Processing</h3>
            <div class="job-list" id="processingJobs"><span class="empty">Empty</span></div>
        </div>
        <div class="card">
            <h3>Recent Completed</h3>
            <div class="job-list" id="completedJobs"><span class="empty">Empty</span></div>
        </div>
    </div>

    <h2>Configuration</h2>
    <div class="grid">
        <div class="card">
            <h3>Defaults</h3>
            <div class="stat"><span class="stat-label">Checkpoint</span><span class="stat-value" id="defaultCheckpoint">-</span></div>
            <div class="stat"><span class="stat-label">Size</span><span class="stat-value" id="defaultSize">-</span></div>
            <div class="stat"><span class="stat-label">Steps</span><span class="stat-value" id="defaultSteps">-</span></div>
            <div class="stat"><span class="stat-label">CFG Scale</span><span class="stat-value" id="defaultCfg">-</span></div>
        </div>
        <div class="card">
            <h3>Available Checkpoints</h3>
            <div id="checkpoints" style="max-height:200px;overflow-y:auto;"></div>
        </div>
    </div>
</div>

<script>
function fmtTime(s) {
    if (s < 60) return s.toFixed(1) + 's';
    if (s < 3600) return (s / 60).toFixed(1) + 'm';
    return (s / 3600).toFixed(1) + 'h';
}

function renderJob(job) {
    var prompt = (job.request && job.request.prompt) ? job.request.prompt.substring(0, 60) : '(no prompt)';
    var waitTime = job.wait_time != null ? job.wait_time.toFixed(1) + 's' : null;
    var procTime = job.processing_time != null ? job.processing_time.toFixed(1) + 's' : null;
    var cancelBtn = job.status === 'queued'
        ? '<button class="btn btn-danger" style="padding:3px 8px;font-size:0.75rem;" onclick="cancelJob(\'' + job.id + '\')">Cancel</button>'
        : '';
    var errLine = (job.status === 'failed' && job.error)
        ? '<div class="job-error">' + job.error.substring(0, 80) + '</div>'
        : '';
    var metaParts = [];
    if (job.position > 0) metaParts.push('pos #' + job.position);
    if (waitTime) metaParts.push('wait ' + waitTime);
    if (procTime) metaParts.push('proc ' + procTime);
    var meta = metaParts.join(' \xb7 ');

    return '<div class="job">'
        + '<div class="job-header">'
        + '<span class="job-id">' + job.id.substring(0, 8) + '</span>'
        + '<div style="display:flex;gap:6px;align-items:center;">'
        + '<span class="badge badge-' + job.status + '">' + job.status + '</span>'
        + cancelBtn
        + '</div></div>'
        + '<div class="job-prompt">' + prompt + '</div>'
        + (meta ? '<div class="job-meta">' + meta + '</div>' : '')
        + errLine
        + '</div>';
}

async function fetchStatus() {
    try {
        var res = await fetch('/api/status');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        var d = await res.json();

        document.getElementById('live-dot').classList.remove('off');

        document.getElementById('uptime').textContent = fmtTime(d.server.uptime_seconds);
        document.getElementById('generations').textContent = d.server.generation_count;
        var errEl = document.getElementById('errors');
        errEl.textContent = d.server.error_count;
        errEl.className = 'stat-value' + (d.server.error_count > 0 ? ' bad' : '');
        document.getElementById('idleTime').textContent = fmtTime(d.server.idle_seconds);
        document.getElementById('currentTask').textContent = d.server.current_task || 'Idle';

        var gpuEl = document.getElementById('gpuStatus');
        gpuEl.textContent = d.gpu.busy ? ('Busy' + (d.gpu.holder ? ' (' + d.gpu.holder + ')' : '')) : 'Idle';
        gpuEl.className = 'stat-value' + (d.gpu.busy ? ' warn' : ' good');

        document.getElementById('vramUsed').textContent = d.vram.used_gb + ' GB';
        document.getElementById('vramFree').textContent = d.vram.free_gb + ' GB';
        document.getElementById('vramTotal').textContent = d.vram.total_gb + ' GB';
        var bar = document.getElementById('vramBar');
        bar.style.width = d.vram.used_percent + '%';
        bar.className = 'progress-fill' + (d.vram.used_percent > 90 ? ' critical' : d.vram.used_percent > 70 ? ' high' : '');

        var pDiv = document.getElementById('pipelines');
        if (!d.pipelines.loaded.length) {
            pDiv.innerHTML = '<span class="empty">None loaded</span>';
        } else {
            pDiv.innerHTML = d.pipelines.loaded.map(function(p) {
                var ckpt = d.pipelines.details[p] && d.pipelines.details[p].checkpoint;
                return '<div class="stat"><span class="stat-label">' + p + '</span><span class="stat-value">' + (ckpt || '\u2014') + '</span></div>';
            }).join('');
        }

        document.getElementById('currentClassifier').textContent = d.classifiers.current || 'None';
        document.getElementById('availableClassifiers').innerHTML = d.classifiers.available.map(function(c) {
            return '<span class="chip">' + c + '</span>';
        }).join('');

        document.getElementById('queueDepth').textContent = d.queue.queue_depth;
        document.getElementById('processingCount').textContent = d.queue.processing_count;

        var qEl = document.getElementById('queuedJobs');
        qEl.innerHTML = d.queue.queued.length ? d.queue.queued.map(renderJob).join('') : '<span class="empty">Empty</span>';
        var pEl = document.getElementById('processingJobs');
        pEl.innerHTML = d.queue.processing.length ? d.queue.processing.map(renderJob).join('') : '<span class="empty">Empty</span>';
        var cEl = document.getElementById('completedJobs');
        cEl.innerHTML = d.queue.recent_completed.length ? d.queue.recent_completed.map(renderJob).join('') : '<span class="empty">Empty</span>';

        document.getElementById('defaultCheckpoint').textContent = d.config.default_checkpoint;
        document.getElementById('defaultSize').textContent = d.config.default_width + ' \xd7 ' + d.config.default_height;
        document.getElementById('defaultSteps').textContent = d.config.default_steps;
        document.getElementById('defaultCfg').textContent = d.config.default_cfg_scale;
        document.getElementById('checkpoints').innerHTML = d.config.available_checkpoints.length
            ? d.config.available_checkpoints.map(function(c) { return '<div class="chip">' + c + '</div>'; }).join('')
            : '<span class="empty">None found</span>';

        document.getElementById('lastUpdate').textContent = 'Updated ' + new Date().toLocaleTimeString();

    } catch(e) {
        document.getElementById('live-dot').classList.add('off');
        console.error('fetchStatus failed:', e);
    }
}

function post(url, body) {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : null,
    }).then(function(r) { return r.json(); });
}

function unloadAll() {
    if (!confirm('Unload all pipelines?')) return;
    post('/api/unload', {}).then(fetchStatus);
}

function cancelJob(id) {
    post('/api/queue/cancel', { job_id: id }).then(fetchStatus);
}

function clearQueue() {
    if (!confirm('Clear all queued jobs?')) return;
    post('/api/queue/clear').then(fetchStatus);
}

function resetStats() {
    if (!confirm('Reset server statistics?')) return;
    post('/api/server/reset-stats').then(fetchStatus);
}

fetchStatus();
setInterval(function() {
    if (document.getElementById('autoRefresh').checked) fetchStatus();
}, 3000);
</script>
</body>
</html>
"""


def run_webui(host: str = "0.0.0.0", port: int = 5001):
    """Run the admin webui server."""
    app = create_webui_app()
    logger.info(f"Starting admin webui on {host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
