"""
FastAPI Dashboard for GenAI-Traces.

Provides a web UI to view traces, results, and analytics.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""
    traces_dir: str = "./outputs/traces"
    results_dir: str = "./outputs"
    csv_dir: str = "./outputs/csv"
    json_dir: str = "./outputs/json"
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "GenAI-Traces Dashboard"
    
    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Create config from environment variables."""
        return cls(
            traces_dir=os.getenv("GENAI_TRACES_DIR", "./outputs/traces"),
            results_dir=os.getenv("GENAI_RESULTS_DIR", "./outputs"),
            csv_dir=os.getenv("GENAI_CSV_DIR", "./outputs/csv"),
            json_dir=os.getenv("GENAI_JSON_DIR", "./outputs/json"),
            host=os.getenv("GENAI_DASHBOARD_HOST", "0.0.0.0"),
            port=int(os.getenv("GENAI_DASHBOARD_PORT", "8000")),
            title=os.getenv("GENAI_DASHBOARD_TITLE", "GenAI-Traces Dashboard"),
        )


def create_app(config: Optional[DashboardConfig] = None) -> FastAPI:
    """
    Create the FastAPI dashboard application.
    
    Args:
        config: DashboardConfig instance or None to use defaults/env vars
        
    Returns:
        FastAPI application instance
    """
    config = config or DashboardConfig.from_env()
    
    app = FastAPI(
        title=config.title,
        description="Web dashboard for viewing GenAI-Traces data",
        version="1.0.0",
    )
    
    # Store config in app state
    app.state.config = config
    
    # ========================================
    # HTML Templates (inline for simplicity)
    # ========================================
    
    def get_base_html(title: str, content: str) -> str:
        """Generate base HTML template."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {config.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .navbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .navbar h1 {{ font-size: 1.5rem; }}
        .navbar nav {{ margin-top: 0.5rem; }}
        .navbar a {{ 
            color: rgba(255,255,255,0.9); 
            text-decoration: none; 
            margin-right: 1.5rem;
            font-weight: 500;
        }}
        .navbar a:hover {{ color: white; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .card h2 {{ 
            color: #667eea; 
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .stat-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-card .label {{ color: #666; margin-top: 0.5rem; }}
        .stat-card.success .value {{ color: #10b981; }}
        .stat-card.error .value {{ color: #ef4444; }}
        .stat-card.warning .value {{ color: #f59e0b; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ 
            background: #f8f9fa; 
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .badge.pass {{ background: #d1fae5; color: #065f46; }}
        .badge.fail {{ background: #fee2e2; color: #991b1b; }}
        .badge.error {{ background: #fef3c7; color: #92400e; }}
        .badge.ok {{ background: #d1fae5; color: #065f46; }}
        .badge.llm {{ background: #dbeafe; color: #1e40af; }}
        .badge.agent {{ background: #e0e7ff; color: #3730a3; }}
        .badge.tool {{ background: #fce7f3; color: #9d174d; }}
        .btn {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #667eea;
            color: white;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 500;
            margin-right: 0.5rem;
        }}
        .btn:hover {{ background: #5a67d8; }}
        .btn.secondary {{ background: #6b7280; }}
        .file-list {{ list-style: none; }}
        .file-list li {{
            padding: 0.75rem;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .file-list li:hover {{ background: #f8f9fa; }}
        pre {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85rem;
        }}
        .search-box {{
            padding: 0.75rem 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            width: 100%;
            max-width: 400px;
            margin-bottom: 1rem;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <h1>{config.title}</h1>
        <nav>
            <a href="/">Dashboard</a>
            <a href="/traces">Traces</a>
            <a href="/results">Results</a>
            <a href="/files">Files</a>
            <a href="/api/docs">API Docs</a>
        </nav>
    </div>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""
    
    # ========================================
    # Helper Functions
    # ========================================
    
    def load_traces(limit: int = 100) -> List[Dict]:
        """Load traces from JSONL files."""
        traces = []
        traces_path = Path(config.traces_dir)
        
        if not traces_path.exists():
            return traces
        
        for jsonl_file in sorted(traces_path.glob("*.jsonl"), reverse=True):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            traces.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    if len(traces) >= limit:
                        break
            if len(traces) >= limit:
                break
        
        return traces
    
    def load_results() -> Dict[str, Any]:
        """Load latest results from JSON."""
        json_path = Path(config.json_dir)
        
        if not json_path.exists():
            return {}
        
        json_files = sorted(json_path.glob("results_*.json"), reverse=True)
        if not json_files:
            return {}
        
        with open(json_files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_file_list() -> Dict[str, List[Dict]]:
        """Get list of all output files."""
        files = {
            "traces": [],
            "json": [],
            "csv": [],
        }
        
        # Traces
        traces_path = Path(config.traces_dir)
        if traces_path.exists():
            for f in traces_path.glob("*.jsonl"):
                files["traces"].append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        
        # JSON results
        json_path = Path(config.json_dir)
        if json_path.exists():
            for f in json_path.glob("*.json"):
                files["json"].append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
        
        # CSV files
        csv_path = Path(config.csv_dir)
        if csv_path.exists():
            for f in csv_path.rglob("*.csv"):
                files["csv"].append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    "category": f.parent.name if f.parent != csv_path else "root",
                })
        
        return files
    
    # ========================================
    # HTML Routes
    # ========================================
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Main dashboard page."""
        traces = load_traces(limit=1000)
        results = load_results()
        
        # Calculate stats
        total_traces = len(traces)
        ok_traces = sum(1 for t in traces if t.get("status") == "ok")
        error_traces = sum(1 for t in traces if t.get("status") == "error")
        
        # Trace types breakdown
        type_counts = {}
        for t in traces:
            span_type = t.get("span_type", "unknown")
            type_counts[span_type] = type_counts.get(span_type, 0) + 1
        
        # Results summary
        summary = results.get("summary", {})
        
        content = f"""
        <h2 style="margin-bottom: 1.5rem;">Overview</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{total_traces}</div>
                <div class="label">Total Traces</div>
            </div>
            <div class="stat-card success">
                <div class="value">{ok_traces}</div>
                <div class="label">Successful</div>
            </div>
            <div class="stat-card error">
                <div class="value">{error_traces}</div>
                <div class="label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="value">{summary.get('total_tests', 0)}</div>
                <div class="label">Tests Run</div>
            </div>
            <div class="stat-card success">
                <div class="value">{summary.get('passed', 0)}</div>
                <div class="label">Tests Passed</div>
            </div>
            <div class="stat-card warning">
                <div class="value">{summary.get('pass_rate', 0):.1%}</div>
                <div class="label">Pass Rate</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Traces by Type</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td><span class="badge {t}">{t}</span></td>
                    <td>{c}</td>
                    <td>{c/total_traces*100:.1f}%</td>
                </tr>
                ''' for t, c in sorted(type_counts.items(), key=lambda x: -x[1]))}
            </table>
        </div>
        
        <div class="card">
            <h2>Recent Traces</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Time</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td>{t.get("name", "N/A")}</td>
                    <td><span class="badge {t.get("span_type", "")}">{t.get("span_type", "N/A")}</span></td>
                    <td><span class="badge {t.get("status", "")}">{t.get("status", "N/A")}</span></td>
                    <td>{t.get("duration_ms", "N/A")} ms</td>
                    <td>{t.get("start_time", "N/A")[:19]}</td>
                </tr>
                ''' for t in traces[:10])}
            </table>
            <p style="margin-top: 1rem;"><a href="/traces" class="btn">View All Traces</a></p>
        </div>
        """
        
        return HTMLResponse(get_base_html("Dashboard", content))
    
    @app.get("/traces", response_class=HTMLResponse)
    async def traces_page(
        limit: int = Query(100, ge=1, le=1000),
        span_type: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """Traces listing page."""
        traces = load_traces(limit=limit)
        
        # Apply filters
        if span_type:
            traces = [t for t in traces if t.get("span_type") == span_type]
        if status:
            traces = [t for t in traces if t.get("status") == status]
        
        content = f"""
        <h2 style="margin-bottom: 1.5rem;">Traces ({len(traces)})</h2>
        
        <div class="card">
            <form method="get" style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                <select name="span_type" style="padding: 0.5rem; border-radius: 6px; border: 1px solid #ddd;">
                    <option value="">All Types</option>
                    <option value="llm" {"selected" if span_type == "llm" else ""}>LLM</option>
                    <option value="agent" {"selected" if span_type == "agent" else ""}>Agent</option>
                    <option value="tool" {"selected" if span_type == "tool" else ""}>Tool</option>
                    <option value="retrieval" {"selected" if span_type == "retrieval" else ""}>Retrieval</option>
                </select>
                <select name="status" style="padding: 0.5rem; border-radius: 6px; border: 1px solid #ddd;">
                    <option value="">All Status</option>
                    <option value="ok" {"selected" if status == "ok" else ""}>OK</option>
                    <option value="error" {"selected" if status == "error" else ""}>Error</option>
                </select>
                <button type="submit" class="btn">Filter</button>
            </form>
            
            <table>
                <tr>
                    <th>Trace ID</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Model</th>
                    <th>Time</th>
                    <th>Actions</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td><code>{t.get("trace_id", "")[:12]}...</code></td>
                    <td>{t.get("name", "N/A")}</td>
                    <td><span class="badge {t.get("span_type", "")}">{t.get("span_type", "N/A")}</span></td>
                    <td><span class="badge {t.get("status", "")}">{t.get("status", "N/A")}</span></td>
                    <td>{t.get("duration_ms", "N/A")} ms</td>
                    <td>{t.get("attributes", {}).get("llm.model.name", "-")}</td>
                    <td>{t.get("start_time", "N/A")[:19]}</td>
                    <td><a href="/api/traces/{t.get("trace_id", "")}" class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">View</a></td>
                </tr>
                ''' for t in traces)}
            </table>
        </div>
        """
        
        return HTMLResponse(get_base_html("Traces", content))
    
    @app.get("/results", response_class=HTMLResponse)
    async def results_page():
        """Results page."""
        results = load_results()
        summary = results.get("summary", {})
        functionalities = results.get("functionalities", {})
        
        content = f"""
        <h2 style="margin-bottom: 1.5rem;">Test Results</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{summary.get('total_tests', 0)}</div>
                <div class="label">Total Tests</div>
            </div>
            <div class="stat-card success">
                <div class="value">{summary.get('passed', 0)}</div>
                <div class="label">Passed</div>
            </div>
            <div class="stat-card error">
                <div class="value">{summary.get('failed', 0)}</div>
                <div class="label">Failed</div>
            </div>
            <div class="stat-card warning">
                <div class="value">{summary.get('pass_rate', 0):.1%}</div>
                <div class="label">Pass Rate</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Results by Functionality</h2>
            <table>
                <tr>
                    <th>Functionality</th>
                    <th>Description</th>
                    <th>Total</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Pass Rate</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{data.get("description", "")}</td>
                    <td>{data.get("total_tests", 0)}</td>
                    <td style="color: #10b981;">{data.get("passed", 0)}</td>
                    <td style="color: #ef4444;">{data.get("failed", 0)}</td>
                    <td><span class="badge {"pass" if data.get("pass_rate", 0) >= 0.8 else "fail"}">{data.get("pass_rate", 0):.1%}</span></td>
                </tr>
                ''' for name, data in functionalities.items())}
            </table>
        </div>
        
        <div class="card">
            <h2>Download Results</h2>
            <p style="margin-bottom: 1rem;">Download test results in various formats:</p>
            <a href="/api/results/json" class="btn">Download JSON</a>
            <a href="/api/results/csv" class="btn secondary">Download CSV</a>
        </div>
        """
        
        return HTMLResponse(get_base_html("Results", content))
    
    @app.get("/files", response_class=HTMLResponse)
    async def files_page():
        """Files listing page."""
        files = get_file_list()
        
        def format_size(size: int) -> str:
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size/1024:.1f} KB"
            else:
                return f"{size/(1024*1024):.1f} MB"
        
        content = f"""
        <h2 style="margin-bottom: 1.5rem;">Output Files</h2>
        
        <div class="card">
            <h2>Trace Files ({len(files['traces'])})</h2>
            <ul class="file-list">
                {"".join(f'''
                <li>
                    <span>{f["name"]} <small style="color: #666;">({format_size(f["size"])})</small></span>
                    <a href="/api/files/traces/{f["name"]}" class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">Download</a>
                </li>
                ''' for f in files['traces'])}
            </ul>
        </div>
        
        <div class="card">
            <h2>JSON Results ({len(files['json'])})</h2>
            <ul class="file-list">
                {"".join(f'''
                <li>
                    <span>{f["name"]} <small style="color: #666;">({format_size(f["size"])})</small></span>
                    <a href="/api/files/json/{f["name"]}" class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">Download</a>
                </li>
                ''' for f in files['json'])}
            </ul>
        </div>
        
        <div class="card">
            <h2>CSV Files ({len(files['csv'])})</h2>
            <ul class="file-list">
                {"".join(f'''
                <li>
                    <span>{f["name"]} <small style="color: #666;">({f["category"]} - {format_size(f["size"])})</small></span>
                    <a href="/api/files/csv/{f["name"]}" class="btn" style="padding: 0.25rem 0.5rem; font-size: 0.8rem;">Download</a>
                </li>
                ''' for f in files['csv'])}
            </ul>
        </div>
        """
        
        return HTMLResponse(get_base_html("Files", content))
    
    # ========================================
    # API Routes
    # ========================================
    
    @app.get("/api/traces")
    async def api_get_traces(
        limit: int = Query(100, ge=1, le=1000),
        span_type: Optional[str] = None,
        status: Optional[str] = None,
    ):
        """Get traces as JSON."""
        traces = load_traces(limit=limit)
        
        if span_type:
            traces = [t for t in traces if t.get("span_type") == span_type]
        if status:
            traces = [t for t in traces if t.get("status") == status]
        
        return {"count": len(traces), "traces": traces}
    
    @app.get("/api/traces/{trace_id}")
    async def api_get_trace(trace_id: str):
        """Get a specific trace by ID."""
        traces = load_traces(limit=10000)
        
        for trace in traces:
            if trace.get("trace_id") == trace_id:
                return trace
        
        raise HTTPException(status_code=404, detail="Trace not found")
    
    @app.get("/api/results")
    async def api_get_results():
        """Get test results as JSON."""
        return load_results()
    
    @app.get("/api/results/json")
    async def api_download_results_json():
        """Download results as JSON file."""
        json_path = Path(config.json_dir)
        json_files = sorted(json_path.glob("results_*.json"), reverse=True)
        
        if not json_files:
            raise HTTPException(status_code=404, detail="No results found")
        
        return FileResponse(
            json_files[0],
            media_type="application/json",
            filename=json_files[0].name,
        )
    
    @app.get("/api/results/csv")
    async def api_download_results_csv():
        """Download results as CSV file."""
        csv_path = Path(config.csv_dir)
        csv_files = sorted(csv_path.glob("all_results_*.csv"), reverse=True)
        
        if not csv_files:
            raise HTTPException(status_code=404, detail="No CSV results found")
        
        return FileResponse(
            csv_files[0],
            media_type="text/csv",
            filename=csv_files[0].name,
        )
    
    @app.get("/api/files/{category}/{filename}")
    async def api_download_file(category: str, filename: str):
        """Download a specific file."""
        if category == "traces":
            filepath = Path(config.traces_dir) / filename
        elif category == "json":
            filepath = Path(config.json_dir) / filename
        elif category == "csv":
            # Search in csv directory and subdirectories
            csv_path = Path(config.csv_dir)
            filepath = csv_path / filename
            if not filepath.exists():
                for subdir in csv_path.iterdir():
                    if subdir.is_dir():
                        potential = subdir / filename
                        if potential.exists():
                            filepath = potential
                            break
        else:
            raise HTTPException(status_code=400, detail="Invalid category")
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        media_type = "application/json" if filename.endswith(".json") else \
                     "text/csv" if filename.endswith(".csv") else \
                     "application/octet-stream"
        
        return FileResponse(filepath, media_type=media_type, filename=filename)
    
    @app.get("/api/stats")
    async def api_get_stats():
        """Get dashboard statistics."""
        traces = load_traces(limit=10000)
        results = load_results()
        files = get_file_list()
        
        return {
            "traces": {
                "total": len(traces),
                "ok": sum(1 for t in traces if t.get("status") == "ok"),
                "error": sum(1 for t in traces if t.get("status") == "error"),
                "by_type": {
                    t: sum(1 for tr in traces if tr.get("span_type") == t)
                    for t in set(tr.get("span_type") for tr in traces)
                },
            },
            "results": results.get("summary", {}),
            "files": {
                "traces": len(files["traces"]),
                "json": len(files["json"]),
                "csv": len(files["csv"]),
            },
        }
    
    return app


def run_dashboard(config: Optional[DashboardConfig] = None):
    """Run the dashboard server."""
    import uvicorn
    
    config = config or DashboardConfig.from_env()
    app = create_app(config)
    
    print(f"Starting GenAI-Traces Dashboard at http://{config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    run_dashboard()
