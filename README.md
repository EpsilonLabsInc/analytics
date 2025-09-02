## Analytics dashboard for Epsilon Health

Setup env (please install `uv` package manager if you don't have it):

```
uv venv
uv source .venv/bin/activate
uv pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run dashboard.py
```

With the dashboard running, you can upload the CSV and inspect the data.