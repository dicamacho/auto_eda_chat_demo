# app.py
import os, re, json
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Auto-EDA Chat Demo", page_icon="📊", layout="wide")

# ---------- Optional LLM ----------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    key = os.environ.get("OPENAI_API_KEY", None)
    if not key:
        key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def safe_sql(sql: str) -> bool:
    SAFE = re.compile(r"^\s*select\s", re.IGNORECASE | re.DOTALL)
    blocked = ["insert","update","delete","drop","alter","create","attach","pragma","copy","call","load",";"]
    if not SAFE.match(sql or ""):
        return False
    s = (sql or "").lower()
    return not any(b in s for b in blocked)

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
                )}
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        sql = parsed.get("sql")
        exp = parsed.get("explanation")
        if not sql or not safe_sql(sql):
            return {"sql": None, "explanation": None, "natural_answer": "I couldn't generate a safe query. Try rephrasing."}
        return {"sql": sql, "explanation": exp, "natural_answer": None}
    except Exception as e:
        return {"sql": None, "explanation": None, "natural_answer": f"LLM error: {e}"}

# ---------- Utility ----------
def ensure_datetime(df: pd.DataFrame):
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            conv = pd.to_datetime(df[c], errors="raise", infer_datetime_format=True)
            if conv.notna().mean() > 0.8:
                df[c] = conv
        except Exception:
            pass
    return df

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

def auto_charts(df: pd.DataFrame):
    charts = []
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dt = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dt]

    # Numeric histograms
    for col in num[:4]:
        try:
            charts.append(("Distribution of " + col, px.histogram(df, x=col, nbins=30)))
        except Exception: pass

    # Categorical bars
    for col in cat[:3]:
        try:
            vc = df[col].astype(str).value_counts().head(12).reset_index()
            vc.columns = [col, "count"]
            charts.append((f"Top {col} values", px.bar(vc, x=col, y="count")))
        except Exception: pass

    # First datetime vs first numeric
    if dt and num:
        d, y = dt[0], num[0]
        try:
            tmp = df[[d, y]].dropna()
            tmp = tmp.groupby(pd.to_datetime(tmp[d]).dt.date, as_index=False)[y].mean()
            tmp[d] = pd.to_datetime(tmp[d])
            charts.append((f"{y} over time", px.line(tmp, x=d, y=y)))
        except Exception: pass

    # Scatter first two numerics
    if len(num) >= 2:
        x, y = num[0], num[1]
        try:
            charts.append((f"{y} vs {x}", px.scatter(df, x=x, y=y, trendline="ols")))
        except Exception: pass

    return charts

# ---------- UI ----------
st.markdown("""
<div style="padding: 8px 14px; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; background: rgba(255,255,255,0.03)">
  <h1 style="margin:0;font-size:1.8rem;">📊 Auto‑EDA Chat Demo</h1>
  <p style="margin:0.25rem 0 0 0;opacity:0.8;">
    Upload a CSV or try the demo dataset. Get instant charts, concise insights, and chat with an assistant that proposes safe, SELECT‑only SQL.
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Demo Controls")
    use_demo = st.toggle("Use demo dataset", value=True, help="Turn off to upload your own CSV.")
    if not use_demo:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    else:
        uploaded = None

    st.divider()
    st.caption("💡 Tip: Add your `OPENAI_API_KEY` in **Settings → Secrets** to enable Insights & Chat.")

# Load data
@st.cache_data(show_spinner=False)
def load_df(file_or_demo: str):
    if file_or_demo == "demo":
        df = pd.read_csv("demo_retail_sales.csv")
    else:
        df = pd.read_csv(file_or_demo)
    return ensure_datetime(df)

if use_demo:
    df = load_df("demo")
else:
    if uploaded is None:
        st.info("Upload a CSV from the sidebar or switch on the demo dataset.")
        st.stop()
    df = load_df(uploaded)

# Profile
profile, sample = summarize_dataframe(df)
profile_json = {"profile": profile, "sample_head": sample}
client = get_openai_client()

# Top KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", f"{profile['n_rows']:,}")
c2.metric("Columns", str(profile['n_cols']))
missing_pct = np.mean([c.get("null_pct",0.0) for c in profile["columns"]]) if profile["columns"] else 0.0
c3.metric("Avg Missing", f"{missing_pct:0.2f}%")
c4.metric("Numeric Cols", str(sum(1 for c in df.columns if np.issubdtype(df[c].dtype, np.number))))

tabs = st.tabs(["📈 Charts", "🧠 Insights", "💬 Chat", "🧾 Schema"])

with tabs[0]:
    st.subheader("Auto‑generated visuals")
    for title, fig in auto_charts(df):
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Executive insights")
    if client is None:
        st.info("Set an `OPENAI_API_KEY` to enable LLM‑generated insights.")
    else:
        with st.spinner("Summarizing dataset..."):
            insights = llm_auto_insights(client, profile_json)
        st.markdown(insights or "_No insights available._")

with tabs[2]:
    st.subheader("Ask questions in natural language")
    st.write("The assistant will propose **SELECT‑only** DuckDB SQL; we validate and execute it.")
    con = duckdb.connect(database=":memory:")
    con.register("data", df)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("e.g., total sales by state, trend of profit by month, top subcategories by quantity...")
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
                        # Quick auto-bar if small
                        if 1 <= ans.shape[1] <= 3 and ans.shape[0] > 0:
                            cols = ans.columns.tolist()
                            if ans.shape[1] == 2:
                                fig = px.bar(ans, x=cols[0], y=cols[1], title=f"{cols[1]} by {cols[0]}")
                                st.plotly_chart(fig, use_container_width=True)
                            elif ans.shape[1] == 3:
                                fig = px.bar(ans, x=cols[0], y=cols[1], color=cols[2], barmode="group",
                                             title=f"{cols[1]} by {cols[0]} colored by {cols[2]}")
                                st.plotly_chart(fig, use_container_width=True)
                        st.session_state.chat_history.append({"role":"assistant","content":f"Returned {len(ans)} rows."})
                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        st.session_state.chat_history.append({"role":"assistant","content":f"Query failed: {e}"})

with tabs[3]:
    st.subheader("Column summary")
    st.json(profile, expanded=False)

st.markdown("---")
st.caption("Built with Streamlit • DuckDB • Plotly • OpenAI (optional)")
