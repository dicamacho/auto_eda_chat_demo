
# app.py — Auto-EDA Agent Demo (Hard-coded ggplot2 theme, with px.defaults shim)

import os, re, json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import duckdb
import plotly.express as px
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------
# Page (wide layout)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Auto-EDA Agent Demo", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------
# Hard-code ggplot2 theme/colors + compatibility shim
# ---------------------------------------------------------------------
TEMPLATE = "ggplot2"
pio.templates.default = TEMPLATE
px.defaults.template = TEMPLATE

# Some environments don't define px.defaults.text; Plotly tries to read it
for _attr in ["text"]:
    if not hasattr(px.defaults, _attr):
        setattr(px.defaults, _attr, None)

# Minimal helper styling (kept super light to let ggplot2 shine)
GRID_COLOR = "rgba(0,0,0,0.08)"

# ---------------------------------------------------------------------
# Optional OpenAI client
# ---------------------------------------------------------------------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL") or (st.secrets.get("OPENAI_BASE_URL") if hasattr(st, "secrets") else None)
    try:
        return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    except Exception:
        return None

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            conv = pd.to_datetime(df[c], errors="raise", infer_datetime_format=True)
            if conv.notna().mean() > 0.8:
                df[c] = conv
        except Exception:
            pass
    return df

def fmt_money(x):
    return f"${x:,.0f}" if pd.notna(x) else "-"

def fmt_pct(x):
    try:
        return f"{x*100:,.1f}%" if pd.notna(x) else "-"
    except Exception:
        return "-"

def beautify_fig(fig, x_title=None, y_title=None):
    # Respect ggplot2 styling; only set titles and template.
    fig.update_layout(template=TEMPLATE, legend_title=None)
    if x_title: fig.update_xaxes(title=x_title)
    if y_title: fig.update_yaxes(title=y_title)
    return fig

def summarize_dataframe(df: pd.DataFrame, max_categories=12, sample_rows=20):
    info = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": []}
    for col in df.columns:
        s = df[col]
        c = {"name": str(col), "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
             "null_count": int(s.isna().sum()), "null_pct": float(s.isna().mean()*100.0 if len(s) else 0.0),
             "unique": int(s.nunique(dropna=True))}
        if np.issubdtype(s.dtype, np.number):
            desc = s.describe(include="all")
            c["stats"] = {
                "mean": float(desc.get("mean", np.nan)) if len(desc) else None,
                "std": float(desc.get("std", np.nan)) if len(desc) else None,
                "min": float(desc.get("min", np.nan)) if len(desc) else None,
                "q25": float(s.quantile(0.25)) if s.notna().any() else None,
                "median": float(desc.get("50%", np.nan)) if "50%" in desc else None,
                "q75": float(s.quantile(0.75)) if s.notna().any() else None,
                "max": float(desc.get("max", np.nan)) if len(desc) else None,
            }
        elif np.issubdtype(s.dtype, np.datetime64):
            if s.notna().any():
                c["stats"] = {"min": str(pd.to_datetime(s.min())), "max": str(pd.to_datetime(s.max()))}
        else:
            vc = s.astype(str).value_counts(dropna=True).head(max_categories)
            c["top_values"] = vc.to_dict()
        info["columns"].append(c)
    sample = df.head(sample_rows).to_dict(orient="records")
    return info, sample

# ---------------------------------------------------------------------
# Heuristic auto-charts fallback
# ---------------------------------------------------------------------
def auto_charts(df: pd.DataFrame):
    charts = []
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dt  = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dt]

    if cat:
        c = cat[0]
        vc = df[c].astype(str).value_counts().reset_index()
        vc.columns = [c, "count"]
        vc["cum_pct"] = vc["count"].cumsum() / vc["count"].sum()
        fig = px.bar(vc.head(20), x=c, y="count", title=f"Top {c} (Pareto, head 20)")
        fig.add_scatter(x=vc[c].head(20), y=vc["cum_pct"].head(20), mode="lines+markers", name="Cumulative %", yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        charts.append(("Pareto categories", beautify_fig(fig, c, "count")))

    if len(num) >= 2:
        corr = df[num].corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (numeric)")
        charts.append(("Correlation", beautify_fig(fig)))

    if dt and num:
        d, y = dt[0], num[0]
        g = df.dropna(subset=[d, y]).copy()
        g["month"] = pd.to_datetime(g[d]).dt.to_period("M").dt.to_timestamp()
        s = g.groupby("month", as_index=False)[y].sum()
        fig = px.line(s, x="month", y=y, markers=True, title=f"{y} over time (monthly sum)")
        charts.append(("Monthly", beautify_fig(fig, "month", y)))

    if cat and num:
        c, y = cat[0], num[0]
        topk = df[c].astype(str).value_counts().head(6).index
        dfx = df[df[c].astype(str).isin(topk)]
        fig = px.box(dfx, x=c, y=y, points="suspectedoutliers", title=f"{y} by {c} (box)")
        charts.append(("Box", beautify_fig(fig, c, y)))

    if dt and cat:
        d, c = dt[0], cat[0]
        k = df[c].astype(str).value_counts().head(5).index
        g = df[df[c].astype(str).isin(k)].copy()
        g["month"] = pd.to_datetime(g[d]).dt.to_period("M").dt.to_timestamp()
        tbl = g.groupby(["month", c]).size().unstack(fill_value=0)
        long = ((tbl.div(tbl.sum(axis=1), axis=0)).reset_index().melt(id_vars="month", var_name=c, value_name="pct"))
        fig = px.area(long, x="month", y="pct", color=c, groupnorm="fraction", title=f"Distribution of {c} over time")
        fig.update_yaxes(tickformat=".0%")
        charts.append(("% over time", beautify_fig(fig, "month", "%")))

    if len(cat) >= 2:
        a, b = cat[0], cat[1]
        g = df.groupby([a,b], as_index=False).size().sort_values("size", ascending=False)
        fig = px.treemap(g, path=[a,b], values="size", title=f"Treemap: {a} / {b}")
        charts.append(("Treemap", beautify_fig(fig)))

    if num:
        col = num[0]
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        charts.append(("Histogram", beautify_fig(fig, col, "count")))

    return charts

# ---------------------------------------------------------------------
# SQL guard
# ---------------------------------------------------------------------
def safe_sql(sql: str) -> bool:
    SAFE = re.compile(r"^\s*select\s", re.IGNORECASE | re.DOTALL)
    blocked = ["insert","update","delete","drop","alter","create","attach","pragma","copy","call","load",";"]
    if not SAFE.match(sql or ""):
        return False
    s = (sql or "").lower()
    return not any(b in s for b in blocked)

# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------
def llm_auto_insights(client, profile_json):
    if client is None: return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content":"You are a concise senior data analyst."},
                {"role":"user","content":(
                    "Write 3–6 bullet executive insights and 3–5 follow-up business questions "
                    "based on this dataset profile.\n\nPROFILE JSON:\n"
                    + json.dumps(profile_json, default=str)[:8000]
                )}
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"*LLM insights unavailable:* {e}"

def llm_to_sql(client, question: str, profile_json: dict) -> dict:
    if client is None:
        return {"sql": None, "explanation": None, "natural_answer": "Chat SQL disabled (no OPENAI_API_KEY set)."}
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":(
                    "You write **DuckDB** SQL against a single table named `data`. "
                    "Return JSON with keys 'sql' and 'explanation'. SQL must be SELECT-only."
                )},
                {"role":"user","content":(
                    f"Question: {question}\n\n"
                    f"Schema & stats:\n{json.dumps(profile_json, default=str)[:6000]}"
                )},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content)
        sql = parsed.get("sql")
        exp = parsed.get("explanation")
        if not sql or not safe_sql(sql):
            return {"sql": None, "explanation": None, "natural_answer": "I couldn't generate a safe query. Try rephrasing."}
        return {"sql": sql, "explanation": exp, "natural_answer": None}
    except Exception as e:
        return {"sql": None, "explanation": None, "natural_answer": f"LLM error: {e}"}

# ---------------------------------------------------------------------
# Agent (Planner → Executor → Critic)
# ---------------------------------------------------------------------
def call_planner(client, profile_json, goal):
    if client is None:
        return {"goal": goal, "steps": [{"action":"insight","text":"LLM is disabled. Add OPENAI_API_KEY to enable planning."}]}
    sys = (
        "You are a planning agent for data analysis. "
        "Return JSON with keys: goal (string), steps (array). "
        "Each step is one of: "
        "{action:'sql', query:'SELECT ...'}, "
        "{action:'chart', type:'bar|line|box|area|scatter', x:'', y:'', color?:'' , title?}, "
        "{action:'insight', text:''}. "
        "SQL must be SELECT-only and target a table named `data`. Prefer 3-6 steps. "
        "For chart steps, ensure the preceding SQL returns the columns referenced."
    )
    usr = f"Goal: {goal}\n\nDataset profile:\n{json.dumps(profile_json, default=str)[:6000]}"
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {"goal": goal, "steps": [{"action":"insight","text":f"Planner failed: {e}"}]}

def run_plan_with_critic(client, plan, df, step_budget=6):
    con = duckdb.connect(database=":memory:")
    con.register("data", df)
    report = {"goal": plan.get("goal"), "steps": [], "charts": [], "insights": []}
    last_df = None

    def observe_df(dfr: pd.DataFrame, max_rows=5):
        if dfr is None: return {"rows": 0, "columns": [], "sample": []}
        head = dfr.head(max_rows).to_dict(orient="records")
        return {"rows": int(len(dfr)), "columns": list(map(str, dfr.columns.tolist())), "sample": head}

    for _ in range(min(step_budget, len(plan.get("steps", [])))):
        step = plan["steps"][0] if isinstance(plan["steps"], list) and plan["steps"] else None
        if not step: break
        plan["steps"] = plan["steps"][1:]
        action = (step.get("action") or "").lower()

        obs = {"status":"ok"}
        if action == "sql":
            sql = step.get("query","")
            if not safe_sql(sql):
                obs = {"status":"error","reason":"Unsafe or invalid SQL"}
            else:
                try:
                    last_df = con.execute(sql).fetchdf()
                    obs = {"status":"ok","result":observe_df(last_df)}
                except Exception as e:
                    obs = {"status":"error","reason":str(e)}
        elif action == "chart":
            if last_df is None or last_df.empty:
                obs = {"status":"error","reason":"No data from previous step to chart."}
            else:
                t = step.get("type","bar")
                x = step.get("x"); y = step.get("y"); color = step.get("color"); title = step.get("title")
                try:
                    if t == "bar":
                        fig = px.bar(last_df, x=x, y=y, color=color, title=title, text=y)
                    elif t == "line":
                        fig = px.line(last_df, x=x, y=y, color=color, title=title)
                    elif t == "box":
                        fig = px.box(last_df, x=x, y=y, color=color, title=title)
                    elif t == "area":
                        fig = px.area(last_df, x=x, y=y, color=color, title=title)
                    elif t == "scatter":
                        fig = px.scatter(last_df, x=x, y=y, color=color, title=title)
                    else:
                        fig = px.bar(last_df, x=x, y=y, color=color, title=title, text=y)
                    beautify_fig(fig)
                    report["charts"].append(fig)
                    obs = {"status":"ok","chart":"rendered"}
                except Exception as e:
                    obs = {"status":"error","reason":str(e)}
        elif action == "insight":
            txt = step.get("text","")
            report["insights"].append(txt)
            obs = {"status":"ok","insight":"added"}
        else:
            obs = {"status":"error","reason":"Unknown action"}

        report["steps"].append({"step": step, "observation": obs})

        if client is not None:
            summary = {"step": step, "observation": obs}
            try:
                crit = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    response_format={"type":"json_object"},
                    messages=[
                        {"role":"system","content":(
                            "You are a strict critic. Return JSON: "
                            "{'status':'ok'|'revise', 'reason':str, 'fix'?: step_json}. "
                            "Propose 'revise' only if the step clearly fails or a tiny fix will improve it."
                        )},
                        {"role":"user","content": json.dumps(summary, default=str)}
                    ],
                )
                verdict = json.loads(crit.choices[0].message.content)
                if verdict.get("status") == "revise" and verdict.get("fix"):
                    plan["steps"].insert(0, verdict["fix"])
            except Exception:
                pass

    return report

def render_agent_report(report):
    goal = report.get("goal", "Analysis")
    st.subheader(f"Agent report — {goal}")
    for fig in report.get("charts", []):
        st.plotly_chart(fig, use_container_width=True)
    if report.get("insights"):
        st.markdown("**Findings**")
        for bullet in report["insights"]:
            st.markdown(f"- {bullet}")

# ---------------------------------------------------------------------
# Column classification + viz expert planner
# ---------------------------------------------------------------------
def detect_id_series(s: pd.Series, name: str) -> bool:
    n = len(s)
    if n == 0: return False
    name_flag = bool(re.search(r'\b(id|uuid|guid|code|hash|number|no|invoice|order|ticket)\b', str(name).lower()))
    uniq_ratio = s.astype(str).nunique(dropna=True) / max(n, 1)
    sample_str = s.dropna().astype(str).head(100).str.lower()
    uuid_flag = sample_str.str.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$').mean() > 0.2
    long_token = (sample_str.str.len().mean() if len(sample_str) else 0) > 18
    mono_like = False
    if np.issubdtype(s.dtype, np.number):
        try:
            diffs = pd.Series(np.diff(s.dropna().values))
            mono_like = (diffs.abs() <= 1).mean() > 0.8
        except Exception:
            pass
    return (uniq_ratio > 0.9 and (name_flag or uuid_flag or long_token or mono_like))

def classify_columns(df: pd.DataFrame):
    ids, times, nums, cats, texts = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        if detect_id_series(s, c):
            ids.append(c); continue
        if np.issubdtype(s.dtype, np.datetime64):
            times.append(c); continue
        if np.issubdtype(s.dtype, np.number):
            nums.append(c); continue
        uniq = s.astype(str).nunique(dropna=True)
        if uniq <= 50:
            cats.append(c)
        else:
            avglen = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
            (cats if (uniq <= 200 and avglen < 20) else texts).append(c)
    return {"id": ids, "time": times, "num": nums, "cat": cats, "text": texts}

ALLOWED_CHARTS = {"bar","line","area","box","scatter","treemap"}

def llm_chart_planner(client, profile, sample, buckets, k=6):
    if client is None: return None
    prompt = {
      "role": "user",
      "content": (
        "You are a **senior data visualization expert**. Return JSON with key 'charts' (array). "
        "Use only these df columns:\n"
        f"{json.dumps(buckets, default=str)}\n\n"
        "Rules:\n"
        "- Avoid id columns completely.\n"
        "- Prefer time on x when present; use monthly buckets if daily is noisy.\n"
        "- For bar charts with categorical x and numeric y, aggregate y (sum) and sort descending; keep top-N (<=12).\n"
        "- Keep category cardinality <= 20 overall.\n"
        "- Titles must be executive-ready (e.g., 'Total Profit by Category').\n"
        "- Choose elegant, uncluttered visuals (avoid overplotting).\n"
        "- Types allowed: bar, line, area, box, scatter, treemap.\n"
        "- Return 4–8 charts using fields present in the dataset.\n\n"
        "Dataset profile:\n" + json.dumps({"profile": profile, "sample_head": sample}, default=str)[:6000]
      )
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.15,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":"Respond ONLY with JSON: {\"charts\":[...]}"},
                      prompt],
        )
        data = json.loads(resp.choices[0].message.content)
        specs = data.get("charts", [])[:k]
        valid = []
        for s in specs:
            if s.get("type") in ALLOWED_CHARTS and s.get("x"):
                valid.append({k:v for k,v in s.items() if k in {"type","x","y","color","title","note"}})
        return valid
    except Exception:
        return None

# ---------------------------------------------------------------------
# Pretty bars + lollipop helpers
# ---------------------------------------------------------------------
def _fmt_short(num, is_pct=False, is_money=False):
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return ""
    if is_pct:
        return f"{num*100:.1f}%"
    if is_money:
        if abs(num) >= 1_000_000:
            return f"${num/1_000_000:.1f}M"
        if abs(num) >= 1_000:
            return f"${num/1_000:.1f}k"
        return f"${num:,.0f}"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if abs(num) >= 1_000:
        return f"{num/1_000:.1f}k"
    return f"{num:,.0f}"

def _is_money_col(colname: str):
    n = (colname or "").lower()
    return any(k in n for k in ["sales","revenue","amount","cost","price","charge","spend","profit"])

def _is_pct_col(colname: str):
    n = (colname or "").lower()
    return any(k in n for k in ["rate","pct","percent","ratio","margin"])

def _make_lollipop(df, x, y, title=""):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[y], y=df[x], mode="lines",
        line=dict(width=2),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df[y], y=df[x], mode="markers",
        marker=dict(size=10),
        text=[_fmt_short(v, _is_pct_col(y), _is_money_col(y)) for v in df[y]],
        hovertemplate=f"<b>%{{y}}</b><br>{y}: %{{x:,.2f}}<extra></extra>",
        name=y
    ))
    fig.update_yaxes(title=x, zeroline=False, categoryorder="array", categoryarray=df[x].tolist())
    if _is_pct_col(y):
        fig.update_xaxes(title=y, tickformat=".0%")
    elif _is_money_col(y):
        fig.update_xaxes(title=y, tickprefix="$", tickformat="~s")
    else:
        fig.update_xaxes(title=y, tickformat="~s")
    fig.update_layout(title=title, template=TEMPLATE)
    return beautify_fig(fig)

def render_spec(df, spec, topn=12, as_percent=False):
    typ, x, y, color = spec["type"], spec["x"], spec.get("y"), spec.get("color")
    ttl = spec.get("title") or (f"{y} by {x}" if y else f"{x}")

    if typ == "bar" and x in df.columns and y and y in df.columns:
        g = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False)
        if as_percent:
            total = g[y].sum() or 1
            g[y] = g[y] / total
        df = g.head(topn)

    if typ in {"line","area"} and x in df.columns and y and y in df.columns and np.issubdtype(df[x].dtype, np.datetime64):
        g = df.dropna(subset=[x, y]).copy()
        g["__month__"] = pd.to_datetime(g[x]).dt.to_period("M").dt.to_timestamp()
        df = g.groupby("__month__", as_index=False)[y].sum().rename(columns={"__month__": x})

    if typ == "bar" and x in df.columns and not np.issubdtype(df[x].dtype, np.number):
        if df[x].astype(str).nunique(dropna=True) > 10:
            return _make_lollipop(df, x, y, title=ttl)

    if typ == "bar":
        fig = px.bar(df, x=x, y=y, color=color, title=ttl, text=y)
        fig.update_traces(
            texttemplate=[
                _fmt_short(v, _is_pct_col(y) or as_percent, _is_money_col(y)) if v is not None else ""
                for v in df[y]
            ],
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0
        )
        if _is_pct_col(y) or as_percent:
            fig.update_yaxes(tickformat=".0%")
        elif _is_money_col(y):
            fig.update_yaxes(tickprefix="$", tickformat="~s")
        else:
            fig.update_yaxes(tickformat="~s")
        fig.update_xaxes(categoryorder="total descending")
        fig.update_layout(bargap=0.35, uniformtext_minsize=10, uniformtext_mode="hide")
        try:
            avg_val = df[y].mean()
            fig.add_hline(y=avg_val, line_width=1, line_dash="dot",
                          annotation_text=f"avg {y}: {_fmt_short(avg_val, _is_pct_col(y) or as_percent, _is_money_col(y))}",
                          annotation_position="top left")
        except Exception:
            pass

    elif typ == "line":
        fig = px.line(df, x=x, y=y, color=color, markers=True, title=ttl)

    elif typ == "area":
        fig = px.area(df, x=x, y=y, color=color, title=ttl)

    elif typ == "box":
        fig = px.box(df, x=x, y=y, color=color, title=ttl, points="suspectedoutliers")

    elif typ == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=ttl)

    elif typ == "treemap":
        if y and y in df.columns and np.issubdtype(df[y].dtype, np.number):
            g = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False)
            fig = px.treemap(g, path=[x], values=y, title=ttl)
        else:
            g = df.groupby(x, as_index=False).size().sort_values("size", ascending=False)
            fig = px.treemap(g, path=[x], values="size", title=ttl)
    else:
        return None

    fig.update_traces(hovertemplate=None)
    return beautify_fig(fig, x, y or "value")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown("""
<div style="padding: 8px 14px; border: 1px solid rgba(2,6,23,0.06); border-radius: 10px; background: #ffffff">
  <h1 style="margin:0;font-size:1.8rem;">📊 Auto-EDA Agent Demo</h1>
  <p style="margin:0.25rem 0 0 0;opacity:0.75;">
    Upload a CSV and let the agent pick elegant, insightful charts, summarize findings, chat with SQL, or run a multi-step plan.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar: data only
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    use_demo = st.toggle("Use demo dataset", value=True, help="Turn off to upload your own CSV.")
    uploaded = None if use_demo else st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.caption("💡 Tip: Add your `OPENAI_API_KEY` in **Settings → Secrets** to enable LLM-powered features.")

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df(file_or_demo: str):
    if file_or_demo == "demo":
        return ensure_datetime(pd.read_csv("demo_retail_sales.csv"))
    return ensure_datetime(pd.read_csv(file_or_demo))

df = load_df("demo") if use_demo else (load_df(uploaded) if uploaded is not None else None)
if df is None:
    st.info("Upload a CSV from the sidebar or switch on the demo dataset.")
    st.stop()

# Profile + client
profile, sample = summarize_dataframe(df)
profile_json = {"profile": profile, "sample_head": sample}
client = get_openai_client()

# KPIs
kpi = {}
try:
    if "sales" in df.columns: kpi["Total Sales"] = fmt_money(df["sales"].sum())
    if "profit" in df.columns:
        kpi["Profit"] = fmt_money(df["profit"].sum())
        if "sales" in df.columns and df["sales"].sum() > 0:
            kpi["Margin"] = fmt_pct(df["profit"].sum()/df["sales"].sum())
    dt_cols = df.select_dtypes(include=[np.datetime64]).columns
    if len(dt_cols):
        dcol = dt_cols[0]
        rng = f"{pd.to_datetime(df[dcol]).min().date()} → {pd.to_datetime(df[dcol]).max().date()}"
        kpi["Date Range"] = rng
except Exception:
    pass

cols = st.columns(4)
for (label, val), col in zip(kpi.items(), cols):
    col.metric(label, val)

tabs = st.tabs(["📈 Charts", "🧠 Insights", "💬 Chat", "🤖 Agent", "🧾 Schema"])

# ----------------- Charts Tab -----------------
with tabs[0]:
    st.subheader("Agent-picked visuals")
    c1, c2 = st.columns([1,1])
    topn = c1.slider("Top N (bars)", 5, 25, 12)
    as_percent = c2.toggle("Show bars as % of total", value=False, help="For aggregated bar charts")

    buckets = classify_columns(df)
    specs = None
    if client:
        with st.spinner("Letting the viz expert choose charts..."):
            specs = llm_chart_planner(client, profile, sample, buckets, k=6)

    rendered = 0
    if specs:
        for i, s in enumerate(specs):
            fig = render_spec(df, s, topn=topn, as_percent=as_percent)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                try:
                    png = fig.to_image(format="png")
                    st.download_button("Download chart (PNG)", data=png, file_name=f"chart_{i+1}.png", mime="image/png",
                                       use_container_width=True, key=f"dl_spec_{i}")
                except Exception:
                    st.caption("⬇️ Install `kaleido` to enable chart downloads.")
                if s.get("note"):
                    st.caption(f"🧠 {s['note']}")
                rendered += 1

    if rendered == 0:
        st.caption("No planner charts passed quality checks — showing heuristic charts instead.")
        for i, (title, fig) in enumerate(auto_charts(df)):
            st.plotly_chart(fig, use_container_width=True)

# ----------------- Insights Tab -----------------
with tabs[1]:
    st.subheader("Executive insights")
    if client is None:
        st.info("Set an `OPENAI_API_KEY` to enable LLM-generated insights.")
    else:
        with st.spinner("Summarizing dataset..."):
            insights = llm_auto_insights(client, profile_json)
        st.markdown(insights or "_No insights available._")
        if insights:
            st.download_button("⬇️ Download executive summary", data=insights, file_name="executive_summary.md",
                               mime="text/markdown", use_container_width=True)

# ----------------- Chat Tab -----------------
with tabs[2]:
    st.subheader("Ask questions in natural language")
    st.caption("Try: *total sales by state*, *profit by month*, *discount vs unit_price*, *top subcategories by quantity*.")
    con = duckdb.connect(database=":memory:")
    con.register("data", df)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about this dataset…")
    if user_q:
        st.session_state.chat_history.append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = llm_to_sql(client, user_q, profile_json)
            if res.get("natural_answer"):
                st.markdown(res["natural_answer"])
            else:
                sql = res.get("sql")
                exp = res.get("explanation") or ""
                if not sql:
                    st.warning("No safe SQL produced. Try rephrasing or be more specific.")
                else:
                    st.markdown(f"**Proposed SQL**\n```sql\n{sql}\n```")
                    if exp: st.caption(exp)
                    try:
                        ans = con.execute(sql).fetchdf()
                        st.dataframe(ans, use_container_width=True)
                        if 1 <= ans.shape[1] <= 3 and ans.shape[0] > 0:
                            cols_ = ans.columns.tolist()
                            if ans.shape[1] == 2:
                                fig = px.bar(ans, x=cols_[0], y=cols_[1], title=f"{cols_[1]} by {cols_[0]}", text=cols_[1])
                                st.plotly_chart(fig, use_container_width=True)
                            elif ans.shape[1] == 3:
                                fig = px.bar(ans, x=cols_[0], y=cols_[1], color=cols_[2], barmode="group",
                                             title=f"{cols_[1]} by {cols_[0]} colored by {cols_[2]}", text=cols_[1])
                                st.plotly_chart(fig, use_container_width=True)
                        st.session_state.chat_history.append({"role":"assistant","content":f"Returned {len(ans)} rows."})
                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        st.session_state.chat_history.append({"role":"assistant","content":f"Query failed: {e}"})

# ----------------- Agent Tab -----------------
with tabs[3]:
    st.subheader("Agentic analysis")
    if client is None:
        st.info("Set an `OPENAI_API_KEY` to enable the Agent.")
    else:
        goal = st.text_input("Goal", value="Overview KPIs, trends, top segments, and key drivers.",
                             help="Describe what you want the agent to analyze.")
        if st.button("Analyze my data", use_container_width=True):
            with st.spinner("Planning and executing..."):
                plan = call_planner(client, profile_json, goal)
                report = run_plan_with_critic(client, plan, df, step_budget=6)
            render_agent_report(report)

            # Export concise agent report (JSON-safe)
            def compile_agent_markdown(report, kpi=None):
                def to_json_safe(obj):
                    try:
                        return json.dumps(obj, ensure_ascii=False, default=str)
                    except Exception:
                        return str(obj)
                kpi = kpi or {}
                lines = [f"# Agent Report — {report.get('goal','Analysis')}"]
                if kpi:
                    lines += ["## KPIs", *[f"- **{k}**: {v}" for k, v in kpi.items()], ""]
                if report.get("insights"):
                    lines += ["## Findings", *[f"- {s}" for s in report["insights"]], ""]
                lines += ["## Steps & Observations"]
                for i, s in enumerate(report.get("steps", []), 1):
                    lines.append(f"**{i}. {s['step'].get('action','')}** — {to_json_safe(s['step'])}")
                    lines.append(f"   - Obs: {to_json_safe(s['observation'])}")
                return "\n".join(lines)
            md = compile_agent_markdown(report, kpi=kpi)
            st.download_button("⬇️ Download agent report", md, "agent_report.md", "text/markdown", use_container_width=True)
            with st.expander("Show plan & observations"):
                st.json(report["steps"], expanded=False)

# ----------------- Schema Tab -----------------
with tabs[4]:
    st.subheader("Column summary")
    st.json(profile, expanded=False)

st.markdown("---")
st.caption("Built with Streamlit • DuckDB • Plotly (ggplot2) • OpenAI (optional) • PNG export: kaleido • Source: dicamacho/auto_eda_chat_demo")
