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

A browser tab should open at <http://localhost:8501>. Pass the acknowledgment
gate (section 10), tweak parameters in the sidebar, click **Run analysis**, and
check that the results appear on the **Analyze** tab.

## 2. Push to GitHub

The Streamlit Cloud deployer reads from a GitHub repository, so make sure
your branch is on GitHub and you know the branch name. For this repo:

```
elhajjar1/wrinkleFE
```

If you've been working on a feature branch, either deploy directly from that
branch or merge to `main` first.

## 3. Sign in to Streamlit Community Cloud

1. Go to <https://share.streamlit.io>
2. Click **Sign in with GitHub** and authorize the Streamlit app.
3. On first sign-in, grant access to the GitHub org/account that owns the repo
   (`elhajjar1`).

## 4. Create the app

1. Click **Create app** (top right) and choose to deploy an existing app from
   GitHub.
2. Fill in:
   - **Repository:** `elhajjar1/wrinkleFE`
   - **Branch:** `main` (or your feature branch)
   - **Main file path:** `app.py`
   - **App URL:** pick a sub-domain like `wrinklefe.streamlit.app`
3. Under **Advanced settings**, pick the Python version. Any version the CI
   test matrix covers (3.10–3.13) works; Streamlit's default is 3.12 at the
   time of writing. The version can only be chosen here: changing it later
   means deleting and redeploying the app.
4. Click **Deploy**.

Streamlit Cloud will:
- clone the repo
- install the dependencies in `requirements.txt` (it uses the first
  dependency file it finds, so `pyproject.toml` is not read)
- start `streamlit run app.py`

It does not install the `wrinklefe` package itself: `app.py` puts the
repository's `src/` directory on `sys.path`, so the import works without it.
`requirements.txt` sets only a floor for Streamlit (`streamlit>=1.57.0`), so a
deploy picks up the newest release. The exact `streamlit==1.65.0` in
`requirements-lint.txt` is for the type checker, not for the app.

The first build typically takes 2–4 minutes (numpy/scipy/matplotlib wheels).
Subsequent redeploys reuse the cache and finish in ~30 s.

## 5. Test the live URL

Once the build finishes the URL becomes live (e.g.
`https://wrinklefe.streamlit.app`). Send it to your engineers — no GitHub or
Streamlit account is needed to use a public app.

## 6. Iterate

Every push to the configured branch triggers an automatic redeploy. Use the
**Manage app** panel (lower-right corner of the live app) to:
- view live logs
- reboot the app if it gets stuck
- change secrets (the Python version needs a redeploy, see step 4)
- delete the app

## 7. Resource limits to know about

Streamlit's documentation gives every Community Cloud app the same
approximate limits (as of February 2024, subject to change):
- **CPU:** 0.078 cores minimum, 2 cores maximum
- **Memory:** 690 MB minimum, 2.7 GB maximum
- **Storage:** up to 50 GB, ephemeral
- no GPU

Apps with no traffic for 12 hours go to sleep until a visitor wakes them.

Implications for WrinkleFE:
- The default sidebar runs analytical-only. Expert mode turns the full FE
  solve on, with the mesh capped by the widget ranges (`nx` ≤ 64, `ny` ≤ 32,
  `nz_per_ply` ≤ 4); dense meshes can still run out of memory, so lower those
  caps in `app.py` if that happens.
- Don't write user-supplied data anywhere except `tempfile`-scoped paths;
  the filesystem resets on every reboot.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on `PyQt6` | It shouldn't be in `requirements.txt` (this repo's already correct). Streamlit Cloud has no display server. |
| `pyvistaqt` install errors | Same — pyvista 3D is desktop-only. The Streamlit app uses matplotlib and Plotly. |
| App boots but `import wrinklefe` fails | The deployer does not install the package; `app.py` imports it from `src/` via `sys.path`. Check that `src/wrinklefe/` is on the deployed branch and that `requirements.txt` lists every runtime dependency. |
| Plots don't render | Make sure `matplotlib.use("Agg")` is called before any `pyplot` import. `app.py` already does this. |
| Slow first run | Expected — numpy/scipy wheels are ~80 MB combined. Subsequent runs are cached. |
| Out of memory on large sweep | Reduce `nx/ny/nz_per_ply` or stick to `analytical_only`. |

## 9. Optional next steps

- Add Streamlit secrets via the Manage panel if the app needs API keys later.
- Change the app's `*.streamlit.app` subdomain in its settings.

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
| `goalseek` | an acceptable-limit search finishes | `props` → parameter, objective, status, … |
| `compare_morphologies` | a morphology comparison finishes | `props` → morphologies, loading, … |

`session_id` is a random per-session token (no identity, no IP) so you can
tell repeat runs from distinct visits. Logging is **fail-soft**: a missing
dependency, unshared sheet, or network blip is swallowed and never interrupts
an analysis. The required `gspread` / `google-auth` packages are already in
`requirements.txt`.

> **Privacy:** you're storing emails and basic usage. The gate shows a short
> notice, keeps email optional, and collects nothing else — keep it that way.

## App features

Once the app is running, the sidebar offers four features worth calling out:
three browser conveniences (the Python API takes the layup and material
directly, through `wrinklefe.core.layup.parse_layup` and
`OrthotropicMaterial`), and the acceptable-limit search, which is the browser
form of the `wrinklefe critical` subcommand. All four live in `app.py` and are
available in both the hosted Streamlit instance and a local
`streamlit run app.py`.

### Morphology schematics

When **Expert mode** is on, the **Morphology** selectbox is followed by a
live cartoon (rendered by `_morphology_schematic()` in `app.py`) that
matches the active choice — stack / convex / concave / uniform / graded /
tool_flat — and tracks the amplitude, wavelength and envelope-width inputs.
The selector's help text describes all six and says which to pick from what
the outer surfaces do. For the `graded` mode an additional **Decay floor**
slider appears (0 = full decay to zero at the surface, 1 = uniform); for
`tool_flat` the pinned-surface and transition-ply controls appear below the
cartoon.

On the **Analyze** tab, the **Wrinkle & laminate preview** expander draws the
deformed ply stack through the thickness from the same inputs, so you can see
the profile update as you change the sidebar — no analysis run needed.

### Layup notation (contracted)

The **Layup** textarea accepts either a contracted laminate string or an
explicit comma/semicolon/newline-separated list of ply angles in degrees.
Quoting the in-app help text:

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
Both the contracted and explicit forms can be edited freely in the textarea;
the cross-section and the **Layup stack visualizer** on the Analyze tab show
the parsed plies and report a layup that does not parse.

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

### Acceptable-limit mode ("Find acceptable limit")

**Run analysis** answers *"given this wrinkle, what is the knockdown?"*.
The button directly beneath it inverts the question — *"what is the largest
wrinkle these inputs can carry?"* — which is the number an inspection
criterion or an NCR disposition is written around. It runs
`wrinklefe.goalseek.find_critical_value` on exactly the configuration a run
would solve; the *Acceptable-limit settings* expander holds the searched
parameter (`amplitude` or `wavelength`), the target (a knockdown factor or
an absolute MPa allowable) and, in Expert mode, the bracket, scan-point
count and root tolerance.

Cost is stated before the click. The search runs on the analytical path —
about 25 forward evaluations, well under a second — unless the
configuration enables an FE-only feature (CZM, surface resin pockets, a
non-uniform transverse mode, `tool_flat`), in which case the expander warns
that the same 25 evaluations become full FE solves.

Three things about the result are worth knowing:

- **The reported value is the conservative one.** `brentq` converges *to*
  the root, not to the acceptable side of it, so the search backs the
  answer off toward safety and confirms it with an extra forward run. The
  achieved knockdown shown next to the limit is that confirming
  measurement, not an interpolation; the raw root is quoted only for
  transparency and should never be the number that leaves the app.
- **A refusal is shown verbatim.** When the scanned curve admits no unique
  answer the app prints the engine's own message, which quotes the
  measurements behind the refusal and names the knob that has to move.
  `no_crossing` renders as a warning rather than an error — it can simply
  mean that no defect size in range fails the criterion. The scan curve and
  the per-evaluation ledger are rendered either way, because the curve *is*
  the diagnosis.
- **The limit only reaches an NCR when it belongs to that NCR.** The
  Export tab attaches an *Acceptance limit (goal-seek)* block to the
  validation summary only when the search converged **and** its
  configuration payload equals the payload the displayed results were
  computed from. A limit derived for a different laminate, loading or
  morphology is left off, and the tab says so rather than omitting it
  silently.

**Apply this amplitude (or wavelength) to the sidebar** seeds the geometry
widget with the limit (clamped to the widget's range) so the full analysis
can be run at it. Doing so changes the inputs, so the search above is then
correctly flagged as computed for the previous configuration.
