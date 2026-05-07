# Deploying WrinkleFE to Streamlit Community Cloud

This repo includes `app.py` (a Streamlit wrapper around `wrinklefe.analysis`)
and a `requirements.txt` tuned for the cloud environment. Follow the steps
below to publish a public URL your colleagues can hit from a browser.

## 1. Pre-flight checks

Before deploying, confirm everything works locally:

```bash
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

A browser tab should open at <http://localhost:8501>. Tweak parameters in the
sidebar, click **Run analysis**, and verify the **Results** tab populates.

## 2. Push to GitHub

The Streamlit Cloud deployer reads from a GitHub repository, so make sure
your branch is on GitHub and you know the branch name. For this repo:

```
ranipdx-glitch/wrinklefe
```

If you've been working on a feature branch (e.g. `claude/review-open-issues-uX18N`),
either deploy directly from that branch or merge to `main` first.

## 3. Sign in to Streamlit Community Cloud

1. Go to <https://share.streamlit.io>
2. Click **Sign in with GitHub** and authorize the Streamlit app.
3. On first sign-in, grant access to the GitHub org/account that owns the repo
   (`ranipdx-glitch`).

## 4. Create the app

1. Click **Create app** (top right) → **Deploy a public app from GitHub**.
2. Fill in:
   - **Repository:** `ranipdx-glitch/wrinklefe`
   - **Branch:** `main` (or your feature branch)
   - **Main file path:** `app.py`
   - **App URL:** pick a sub-domain like `wrinklefe.streamlit.app`
3. Click **Advanced settings** if you need a non-default Python version
   (Streamlit Cloud defaults to 3.11 currently). Pick **Python 3.11** to match
   `pyproject.toml`'s `requires-python = ">=3.10"` and the test matrix.
4. Click **Deploy**.

Streamlit Cloud will:
- clone the repo
- create a virtualenv
- run `pip install -r requirements.txt`
- run `pip install -e .` if a `pyproject.toml` is present
- start `streamlit run app.py`

The first build typically takes 2–4 minutes (numpy/scipy/matplotlib wheels).
Subsequent redeploys reuse the cache and finish in ~30 s.

## 5. Test the live URL

Once the build finishes the URL becomes live (e.g.
`https://wrinklefe.streamlit.app`). Send it to your engineers — no GitHub or
Streamlit account is needed to use a public app.

## 6. Iterate

Every push to the configured branch triggers an automatic redeploy. Use the
**Manage app** panel (bottom-right gear in the live app) to:
- view live logs
- reboot the app if it gets stuck
- change Python version or secrets
- pause / delete the app

## 7. Resource limits to know about

Streamlit Cloud (Community tier) gives every app:
- **1 GB RAM**
- **1 vCPU**
- **shared filesystem**, ephemeral
- no GPU
- public-only repo by default (private requires a paid plan)

Implications for WrinkleFE:
- The default config keeps `analytical_only=True` — full FE solves can blow
  past 1 GB on dense meshes. If you want to allow full FE, cap `nx*ny*nz_per_ply`
  or reduce the default mesh.
- Don't write user-supplied data anywhere except `tempfile`-scoped paths;
  the filesystem resets on every reboot.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `PyQt6` | It shouldn't be in `requirements.txt` (this repo's already correct). Streamlit Cloud has no display server. |
| `pyvistaqt` install errors | Same — pyvista 3D is desktop-only. The Streamlit app uses matplotlib. |
| App boots but `import wrinklefe` fails | Confirm the package installs from source: `pip install -e .` works locally. The deployer runs the same step. |
| Plots don't render | Make sure `matplotlib.use("Agg")` is called before any `pyplot` import. `app.py` already does this. |
| Slow first run | Expected — numpy/scipy wheels are ~80 MB combined. Subsequent runs are cached. |
| Out of memory on large sweep | Reduce `nx/ny/nz_per_ply` or stick to `analytical_only`. |

## 9. Optional next steps

- Add a `runtime.txt` with `python-3.11` if you want to pin the Python version
  exactly (Streamlit Cloud reads this).
- Add Streamlit secrets via the Manage panel if the app needs API keys later.
- Configure a custom domain in Streamlit Cloud's settings.
- Add CI to run `streamlit run --headless` smoke tests on every PR.
