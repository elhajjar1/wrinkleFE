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

## 10. Acknowledgment gate + usage logging (optional)

The app opens with a one-time **acknowledgment gate** (`usage_tracking.py`):
visitors are asked to cite / star the repo and may optionally leave an email
before the tool unlocks. Acknowledgment is remembered per browser session via
`st.session_state`, so repeat users see it only once.

This is a **soft gate** by design. The app is public and MIT-licensed, so it
can't *enforce* citation (anyone can `git clone` and run locally) — the gate
makes the request visible and captures interested users. It works with **no
configuration**: with no secrets set, the gate still shows and the rest of the
app runs; only the logging is skipped.

To turn the gate **off** (e.g. a self-hosted internal deploy), set the env var
`WRINKLEFE_DISABLE_GATE=1` — on Streamlit Cloud, add it under *Manage app →
Settings*. The test suite sets the same switch so `AppTest` can drive the app.

To also capture signups + run events in a **Google Sheet**:

1. In Google Cloud Console, create a project and enable the **Google Sheets
   API** and **Google Drive API**.
2. Create a **Service Account**, add a **JSON key**, and download it.
3. Create a Google Sheet to collect rows. Copy its id from the URL
   (`.../spreadsheets/d/<SHEET_KEY>/edit`).
4. **Share** that Sheet with the service account's `client_email` as **Editor**.
5. Paste the credentials into Streamlit secrets. Locally, copy
   [`.streamlit/secrets.toml.example`](../../.streamlit/secrets.toml.example) to
   `.streamlit/secrets.toml` (gitignored — never commit it). On Streamlit
   Cloud, paste the same content into **Manage app → Settings → Secrets**.

The sheet gets one row per event with columns
`timestamp_utc, event, email, session_id, props`:

| event    | when                              | useful columns                              |
|----------|-----------------------------------|---------------------------------------------|
| `signup` | visitor clicks **Enter the app**  | `email` (blank if they skipped it)          |
| `run`    | an analysis completes             | `props` → morphology, loading, n_plies, …   |

`session_id` is a random per-session token (no identity, no IP) so you can
tell repeat runs from distinct visits. Logging is **fail-soft**: a missing
dependency, unshared sheet, or network blip is swallowed and never interrupts
an analysis. The required `gspread` / `google-auth` packages are already in
`requirements.txt`.

> **Privacy:** you're storing emails and basic usage. The gate shows a short
> notice, keeps email optional, and collects nothing else — keep it that way.

## App features

Once the app is running, the sidebar offers three features worth calling out
that aren't exposed through the Python API. All three live in `app.py` and
are available in both the hosted Streamlit instance and a local
`streamlit run app.py`.

### Morphology schematics

When **Expert mode** is on, the **Morphology** selectbox is followed by a
live cartoon (rendered by `_morphology_schematic()` in `app.py`) that
matches the active choice — stack / convex / concave / uniform / graded.
A `?` popover next to the selector also lists thumbnails of all five so you
can compare them at a glance without changing the selection. For the
`graded` mode an additional **Decay floor** slider appears (0 = full decay
to zero at the surface, 1 = uniform).

Below the morphology cartoon, a separate **wrinkle preview** plot tracks the
current `amplitude` (`A`), `wavelength` (`λ`), envelope `width` (`w`) and
`decay_floor` in real time, so you can see the through-thickness profile
update as you drag the sidebar sliders — no analysis run needed.

### Layup notation (contracted)

The **Layup** textarea accepts either a contracted laminate string or an
explicit comma/semicolon/newline-separated list of ply angles in degrees.
Quoting the in-app help text (`app.py:417-436`):

> Accepts contracted notation like `[0/45/-45/90]_3s` or an explicit
> comma-separated list of angles in degrees.

Modifiers (also from the in-app help):

- `±θ` expands to `θ, -θ` (two plies)
- `θ_n` repeats a single ply *n* times (e.g. `0_4` → four 0° plies)
- `_n` after the bracket repeats the whole sublaminate *n* times
- Trailing `s` mirrors the stack to enforce symmetry

| Contracted input    | Expanded plies                                    | Count |
|---------------------|---------------------------------------------------|-------|
| `[0/45/-45/90]_3s`  | `0/45/-45/90` repeated 3× then mirrored (default) | 24    |
| `[0/±45/90]s`       | `0, 45, -45, 90` mirrored                         | 8     |
| `[0_2/90]_2`        | `0, 0, 90` repeated twice                         | 6     |
| `[±30]_2`           | `30, -30` repeated twice                          | 4     |

The default layup is `[0/45/-45/90]_3s` — the 24-ply quasi-isotropic stack.
Both the contracted and explicit forms can be edited freely in the textarea
and validated immediately by the plies-count preview below it.

### Custom-material editor

In **Expert mode** the **Material** selectbox gains a `Custom…` entry at
the bottom. Selecting it opens an inline expander seeded from IM7/8552
with editable fields for:

- *Elastic constants*: `E1`, `E2`, `E3`, `G12`, `G13`, `G23`, `ν12`,
  `ν13`, `ν23`
- *Strength allowables*: `Xt`, `Xc`, `Yt`, `Yc`, `Zt`, `Zc`, `S12`,
  `S13`, `S23`

Override only the values you care about — every other field
(hygrothermal coefficients, `γ_Y`, LaRC fracture-toughness parameters) is
inherited from the seed material. The custom material is named via a
free-text **Material name** field and used for the current run only;
**custom materials do not persist across sessions** (the app's filesystem
is ephemeral on Streamlit Cloud and the session state resets on reload).
For permanent additions to the library see [CONTRIBUTING.md](../../CONTRIBUTING.md).
