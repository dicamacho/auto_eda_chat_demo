# Auto‑EDA Chat Demo

A clean, public Streamlit app users can click: upload a CSV (or use the built‑in demo dataset), get instant charts, optional LLM insights, and chat via **SELECT‑only DuckDB SQL**.

## 🚀 Quickstart (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 One‑Click Deploy (Streamlit Community Cloud) To Replicate:
1) Push these files to a new GitHub repo (e.g., `auto-eda-chat-demo`).  
2) Create an app at https://share.streamlit.io/ (or https://streamlit.io/cloud) → point to `app.py`.  
3) In **⚙️ Settings → Secrets**, add:
```
OPENAI_API_KEY = "sk-..."
```
4) Deploy. The app works without a key (charts only), but Insights/Chat will be enabled with the key.

## ✨ Highlights
- Fast EDA on **any CSV** or the bundled demo dataset
- Auto‑generated visuals (histograms, bars, time‑series, scatter)
- LLM executive‑summary insights (optional)
- NL→SQL chat that validates **SELECT‑only** queries and executes them on DuckDB
- Modern, dark theme that looks great in screenshots

## 📝 Summary of Deployment
> Built and deployed a Streamlit app for automatic exploratory data analysis. Users upload a CSV or use a demo dataset to generate interactive Plotly charts, receive LLM‑generated executive insights, and ask questions in natural language. The agent proposes **SELECT‑only** DuckDB SQL which is validated and executed safely, with results visualized instantly. Stack: Streamlit, DuckDB, Plotly, OpenAI API, Python (pandas).

## 🔧 Notes
- For very large CSVs, consider adding sampling on load or column selection.
- All LLM features are optional; without a key the app still showcases the UI/EDA capabilities.
