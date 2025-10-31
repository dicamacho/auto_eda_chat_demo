# app.py (upgraded + downloads + theme toggle)
import os, re, json
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="Auto-EDA Chat Demo", page_icon="📊", layout="wide")

# ---------- Theme Toggle ----------
with st.sidebar:
    theme = st.selectbox("Theme", ["Dark","Light"], index=0)

# ---------- Plotly Theme (elegant dark) ----------
pio.templates["elegant_dark"] = pio.templates["plotly_dark"]
pio.templates["elegant_dark"].layout.update(
    font=dict(family="Inter, Segoe UI, system-ui", size=14),
    paper_bgcolor="#0B0F19", plot_bgcolor="#111827",
    colorway=["#60A5FA","#22D3EE","#A78BFA","#F472B6","#FCD34D","#34D399"],
    hoverlabel=dict(bgcolor="#111827", font_size=13),
    margin=dict(l=50,r=20,t=40,b=40),
)
pio.templates.default = "elegant_dark" if theme == "Dark" else "plotly_white"

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
    key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL") or (st.secrets.get("OPENAI_BASE_URL") if hasattr(st, "secrets") else None)
    try:
        return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
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
        parsed = json.loads(resp.choices[0].message.content)
        sql = parsed.get("sql")
        exp = parsed.get("explanation")
        if not sql or not safe_sql(sql):
            return {"sql": None, "explanation": None, "natural_answer": "I couldn't generate a safe query. Try rephrasing."}
        return {"sql": sql, "explanation": exp, "natural_answer": None}
    except Exception as e:
        return {"sql": None, "explanation": None, "natural_answer": f"LLM error: {e}"}

# ---------- Utils ----------
def ensure_datetime(df: pd.DataFrame):
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
    if x_title: fig.update_xaxes(title=x_title, showgrid=False)
    if y_title: fig.update_yaxes(title=y_title, showgrid=True, gridcolor="rgba(0,0,0,0.06)" if pio.templates.default=="plotly_white" else "rgba(255,255,255,0.06)")
    fig.update_layout(legend_title=None)
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

def auto_charts(df: pd.DataFrame):
    charts = []
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    dt  = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    cat = [c for c in df.columns if c not in num and c not in dt]

    # A) Pareto top category
    if cat:
        c = cat[0]
        vc = df[c].astype(str).value_counts().reset_index()
        vc.columns = [c, "count"]
        vc["cum_pct"] = vc["count"].cumsum() / vc["count"].sum()
        fig = px.bar(vc.head(20), x=c, y="count", title=f"Top {c} (Pareto, head 20)")
        fig.add_scatter(x=vc[c].head(20), y=vc["cum_pct"].head(20), mode="lines+markers", name="Cumulative %", yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        charts.append(("Pareto categories", beautify_fig(fig, c, "count")))

    # B) Correlation heatmap
    if len(num) >= 2:
        corr = df[num].corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (numeric)")
        charts.append(("Correlation", beautify_fig(fig)))

    # C) Monthly trend + rolling mean
    if dt and num:
        d, y = dt[0], num[0]
        g = df.dropna(subset=[d, y]).copy()
        g["month"] = pd.to_datetime(g[d]).dt.to_period("M").dt.to_timestamp()
        s = g.groupby("month", as_index=False)[y].mean()
        s["roll"] = s[y].rolling(3, min_periods=1).mean()
        fig = px.line(s, x="month", y=[y,"roll"], markers=True, title=f"{y} over time (monthly avg + 3-mo rolling)")
        charts.append(("Monthly", beautify_fig(fig, "month", y)))

    # D) Box plot by top category
    if cat and num:
        c, y = cat[0], num[0]
        topk = df[c].astype(str).value_counts().head(6).index
        dfx = df[df[c].astype(str).isin(topk)]
        fig = px.box(dfx, x=c, y=y, points="suspectedoutliers", title=f"{y} by {c} (box)")
        charts.append(("Box", beautify_fig(fig, c, y)))

    # E) % distribution over time (safe)
    if dt and cat:
        d, c = dt[0], cat[0]
        k = df[c].astype(str).value_counts().head(5).index
        g = df[df[c].astype(str).isin(k)].copy()
        g["month"] = pd.to_datetime(g[d]).dt.to_period("M").dt.to_timestamp()
        tbl = g.groupby(["month", c]).size().unstack(fill_value=0)
        long = (
            (tbl.div(tbl.sum(axis=1), axis=0))
            .reset_index()
            .melt(id_vars="month", var_name=c, value_name="pct")
        )
        fig = px.area(long, x="month", y="pct", color=c, groupnorm="fraction", title=f"Distribution of {c} over time")
        fig.update_yaxes(tickformat=".0%")
        charts.append(("% over time", beautify_fig(fig, "month", "%")))

    # F) Treemap categories
    if len(cat) >= 2:
        a, b = cat[0], cat[1]
        g = df.groupby([a,b], as_index=False).size().sort_values("size", ascending=False)
        fig = px.treemap(g, path=[a,b], values="size", title=f"Treemap: {a} / {b}")
        charts.append(("Treemap", beautify_fig(fig)))

    # G) One numeric histogram
    if num:
        col = num[0]
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        charts.append(("Histogram", beautify_fig(fig, col, "count")))

    # H) Optional: Top states by sales if those fields exist
    if "sales" in df.columns and "state" in df.columns:
        g = df.groupby("state", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(12)
        fig = px.bar(g, x="state", y="sales", title="Top states by sales")
        fig.update_traces(hovertemplate="State: %{x}<br>Sales: %{y:$,.0f}<extra></extra>")
        charts.append(("Top states by sales", beautify_fig(fig, "state", "sales")))

    return charts

# ---------- Header ----------
st.markdown("""
<div style="padding: 8px 14px; border: 1px solid rgba(127,127,127,0.12); border-radius: 10px; background: rgba(127,127,127,0.05)">
  <h1 style="margin:0;font-size:1.8rem;">📊 Auto-EDA Chat Demo</h1>
  <p style="margin:0.25rem 0 0 0;opacity:0.8;">
    Upload a CSV or try the demo dataset. Get instant charts, concise insights, and chat with an assistant that proposes safe, SELECT-only SQL.
  </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    use_demo = st.toggle("Use demo dataset", value=True, help="Turn off to upload your own CSV.")
    if not use_demo:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    else:
        uploaded = None
    st.divider()
    st.caption("💡 Tip: Add your `OPENAI_API_KEY` in **Settings → Secrets** to enable Insights & Chat.")

@st.cache_data(show_spinner=False)
def load_df(file_or_demo: str):
    if file_or_demo == "demo":
        return ensure_datetime(pd.read_csv("demo_retail_sales.csv"))
    return ensure_datetime(pd.read_csv(file_or_demo))

if use_demo:
    df = load_df("demo")
else:
    if uploaded is None:
        st.info("Upload a CSV from the sidebar or switch on the demo dataset.")
        st.stop()
    df = load_df(uploaded)

# Profile + LLM client
profile, sample = summarize_dataframe(df)
profile_json = {"profile": profile, "sample_head": sample}
client = get_openai_client()

# KPI highlights (best-effort)
kpi = {}
try:
    if "sales" in df.columns:
        kpi["Total Sales"] = fmt_money(df["sales"].sum())
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

tabs = st.tabs(["📈 Charts", "🧠 Insights", "💬 Chat", "🧾 Schema"])

# ---------- Charts Tab ----------
with tabs[0]:
    st.subheader("Auto-generated visuals")
    for i, (title, fig) in enumerate(auto_charts(df)):
        st.plotly_chart(fig, use_container_width=True)
        # PNG download (requires kaleido)
        try:
            png = fig.to_image(format="png")
            safe_title = "".join(ch if ch.isalnum() or ch in (" ","_","-") else "_" for ch in title).strip().replace(" ","_")
            st.download_button(
                "Download chart (PNG)",
                data=png,
                file_name=f"{safe_title or 'chart'}_{i+1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_{i}"
            )
        except Exception:
            st.caption("⬇️ Install `kaleido` to enable chart downloads (add to requirements.txt).")

# ---------- Insights Tab ----------
with tabs[1]:
    st.subheader("Executive insights")
    if client is None:
        st.info("Set an `OPENAI_API_KEY` to enable LLM-generated insights.")
    else:
        with st.spinner("Summarizing dataset..."):
            insights = llm_auto_insights(client, profile_json)
        st.markdown(insights or "_No insights available._")
        if insights:
            st.download_button(
                "⬇️ Download executive summary",
                data=insights,
                file_name="executive_summary.md",
                mime="text/markdown",
                use_container_width=True
            )

# ---------- Chat Tab ----------
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
                        # Auto bar if small + PNG download
                        if 1 <= ans.shape[1] <= 3 and ans.shape[0] > 0:
                            cols_ = ans.columns.tolist()
                            if ans.shape[1] == 2:
                                fig = px.bar(ans, x=cols_[0], y=cols_[1], title=f"{cols_[1]} by {cols_[0]}")
                                fig.update_traces(hovertemplate=f"{cols_[0]}: %{{x}}<br>{cols_[1]}: %{{y:,.2f}}<extra></extra>")
                                st.plotly_chart(fig, use_container_width=True)
                                try:
                                    png = fig.to_image(format="png")
                                    st.download_button("Download chart (PNG)", data=png, file_name="chat_chart.png", mime="image/png", use_container_width=True, key=f"dl_chat")
                                except Exception:
                                    st.caption("⬇️ Install `kaleido` to enable chart downloads (add to requirements.txt).")
                            elif ans.shape[1] == 3:
                                fig = px.bar(ans, x=cols_[0], y=cols_[1], color=cols_[2], barmode="group",
                                             title=f"{cols_[1]} by {cols_[0]} colored by {cols_[2]}")
                                st.plotly_chart(fig, use_container_width=True)
                        st.session_state.chat_history.append({"role":"assistant","content":f"Returned {len(ans)} rows."})
                    except Exception as e:
                        st.error(f"Query failed: {e}")
                        st.session_state.chat_history.append({"role":"assistant","content":f"Query failed: {e}"})

# ---------- Schema Tab ----------
with tabs[3]:
    st.subheader("Column summary")
    st.json(profile, expanded=False)

st.markdown("---")
st.caption("Built with Streamlit • DuckDB • Plotly • OpenAI (optional) • Source: dicamacho/auto_eda_chat_demo")
