# How to publish emcad (step by step)

This guide walks you through putting a new version of emcad on:

- **TestPyPI** (test.pypi.org): a practice copy of PyPI. Nothing you do there
  affects real users. Use it to rehearse.
- **PyPI** (pypi.org): the real thing. This is what `pip install emcad`
  downloads.

**You do not need a Windows or Linux computer.** GitHub builds the package for
every operating system (Windows, macOS Intel, macOS Apple Silicon, Linux on
normal PCs, Linux on ARM) on its own machines, tests it, and uploads it. Your
job is to press a few buttons and check the result.

There are three parts:

- **Part 1 – One-time setup.** Do this once, ever. About 20 minutes.
- **Part 2 – Upload to TestPyPI.** The practice run. Recommended every time.
- **Part 3 – Upload to PyPI.** The real release.

> **Golden rule:** a version number can be uploaded **only once**, ever, on
> each site. If `0.2.0` is on PyPI, you can never upload a different `0.2.0`,
> not even after deleting it. To fix a mistake, you make `0.2.1`. That's
> normal and nothing to worry about.

---

## Part 1 – One-time setup

### 1.1 Get the code onto GitHub's `main` branch

GitHub only shows the "Run workflow" button (used in Parts 2 and 3) for
workflows that exist on the `main` branch. The new build setup currently lives
only on your computer, on the branch `rust-kernel`.

1. Open a terminal in the emcad folder (in VS Code: menu **Terminal → New
   Terminal**).
2. Copy and paste these lines one at a time, pressing Enter after each:

   ```bash
   git checkout rust-kernel
   git add -A
   git commit -m "Rust kernel, regularize, wheel builds"
   git push -u origin rust-kernel
   ```

   If `git commit` says "nothing to commit", that's fine; carry on.
3. Go to <https://github.com/FennisRobert/emcad>. A yellow bar should say
   **"rust-kernel had recent pushes"** with a green button **"Compare & pull
   request"**. Click it. (No bar? Click **Pull requests → New pull request**,
   set "compare" to `rust-kernel`, and continue.)
4. Click the green **"Create pull request"** button.
5. Wait. Near the bottom of the page you'll see checks running (yellow dots).
   This is GitHub building and testing emcad on every operating system. It
   takes about 10–20 minutes. Make a coffee.
6. When everything is a **green tick**, click **"Merge pull request"**, then
   **"Confirm merge"**.
   - If something shows a **red cross**, stop. Don't merge. See
     [When things go wrong](#when-things-go-wrong).

### 1.2 Tell PyPI to trust GitHub (no passwords needed later)

GitHub uploads using "Trusted Publishing", which works without passwords or
API tokens. You just tell PyPI once which GitHub workflow is allowed to upload.

1. Log in at <https://pypi.org>.
2. Open this page directly:
   <https://pypi.org/manage/project/emcad/settings/publishing/>
   (or: **Your projects → emcad → Manage → Publishing** in the left menu).
3. Under **"Add a new publisher"**, make sure the **GitHub** tab is selected
   and fill in **exactly** (capital letters matter):

   | Field | Type this |
   |---|---|
   | Owner | `FennisRobert` |
   | Repository name | `emcad` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` |

4. Click **"Add"**.

### 1.3 Tell TestPyPI to trust GitHub too

TestPyPI is a completely separate website with its own accounts.

1. Go to <https://test.pypi.org> and click **Register** (top right) to make an
   account, if you don't have one yet. Confirm your email and set up
   two-factor authentication when it asks (same as on PyPI).
2. Since emcad doesn't exist on TestPyPI yet, you add a **pending**
   publisher. Open:
   <https://test.pypi.org/manage/account/publishing/>
3. Under **"Add a new pending publisher"**, on the **GitHub** tab, fill in:

   | Field | Type this |
   |---|---|
   | PyPI Project Name | `emcad` |
   | Owner | `FennisRobert` |
   | Repository name | `emcad` |
   | Workflow name | `wheels.yml` |
   | Environment name | `testpypi` |

4. Click **"Add"**.

### 1.4 Create the two "environments" on GitHub (your safety switch)

These make every upload wait for you to click "Approve", so nothing can ever
be uploaded by accident.

1. Go to <https://github.com/FennisRobert/emcad/settings/environments>
   (or: your repo → **Settings** tab → **Environments** in the left menu).
2. Click **"New environment"**, type `pypi`, and click **"Configure
   environment"**.
3. Tick **"Required reviewers"**, type your GitHub username in the box, pick
   yourself from the list, and click **"Save protection rules"**.
4. Go back to Environments and repeat steps 2–3 with the name `testpypi`.

Setup is done. You never have to do Part 1 again.

---

## Before every release: choose a version number

1. Decide the new version. Currently it is `0.2.0`. (`0.1.0` is already on
   PyPI, so `0.2.0` is the first one this setup will upload.)
   - Small fix → bump the last number: `0.2.0` → `0.2.1`
   - New features → bump the middle number: `0.2.1` → `0.3.0`
2. Open `pyproject.toml` and change the line `version = "0.2.0"` to your
   new number.
3. Open `rust/Cargo.toml` and change its `version = "0.2.0"` line (the one
   near the top) to the same number. This one is only for tidiness.
4. Save both files, then in the terminal:

   ```bash
   git checkout main
   git pull
   git add pyproject.toml rust/Cargo.toml
   git commit -m "Version 0.2.1"
   git push
   ```

   (Use your real version number in the message. For the very first release,
   `0.2.0` is already set, so skip this whole section.)
5. Go to <https://github.com/FennisRobert/emcad/actions>. The top run (named
   after your commit) should finish with a **green tick** within about 20
   minutes. **Only continue when it's green.**

---

## Part 2 – Upload to TestPyPI (practice run)

1. Go to <https://github.com/FennisRobert/emcad/actions>.
2. In the left menu, click **"wheels"**.
3. On the right, click the grey **"Run workflow"** button. A small box opens:
   - **"Use workflow from"**: leave it on `Branch: main`.
   - **"Upload the built distributions to"**: choose **`testpypi`**.
   - Click the green **"Run workflow"** button.
4. Refresh the page after a few seconds. A new run appears at the top; click
   it. You'll see the jobs: one `wheel …` job per operating system, plus
   `sdist`, plus `publish to testpypi`.
5. Wait about 15–20 minutes until all `wheel` jobs and `sdist` have green
   ticks.
6. A yellow box appears saying the deployment is **waiting for review**.
   Click **"Review deployments"**, tick **`testpypi`**, and click **"Approve
   and deploy"**.
7. Wait about a minute until `publish to testpypi` has a green tick.
8. Check it worked: open <https://test.pypi.org/project/emcad/>, then click
   **"Download files"** on the left. You should see **7 files**:
   - one ending in `.tar.gz` (the source code)
   - six ending in `.whl`, one per system. Their names contain:
     `manylinux…x86_64`, `manylinux…aarch64`, `musllinux…x86_64`,
     `macosx…x86_64`, `macosx…arm64`, and `win_amd64`.

### Optional: try installing it from TestPyPI

This installs it into a throwaway folder, so your normal setup stays untouched:

```bash
cd ~
python3 -m venv try-emcad
source try-emcad/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ emcad==0.2.0
python -c "import emcad; print('emcad works!')"
deactivate
rm -rf try-emcad
```

(Use your real version number instead of `0.2.0`. On Windows, the
`source` line is `try-emcad\Scripts\activate` and the last line is
`rmdir /s try-emcad`.) If it prints **emcad works!**, everything is fine.

---

## Part 3 – Upload to PyPI (the real release)

Exactly the same as Part 2, with one difference: choose **`pypi`**.

1. Go to <https://github.com/FennisRobert/emcad/actions>.
2. Click **"wheels"** in the left menu.
3. Click **"Run workflow"**. Keep `Branch: main`, set **"Upload the built
   distributions to"** to **`pypi`**, and click the green **"Run workflow"**.
4. Open the new run and wait until all `wheel` jobs and `sdist` are green.
5. Click **"Review deployments"**, tick **`pypi`**, and click **"Approve and
   deploy"**. This is the point of no return; until you click it, nothing has
   been uploaded.
6. When `publish to pypi` is green, check
   <https://pypi.org/project/emcad/>. The new version number should show at
   the top, and **"Download files"** should list the same 7 files as above.
7. Anyone can now run `pip install --upgrade emcad`.

### Optional: mark the release in GitHub

Nice to have, but not required:

```bash
git checkout main
git pull
git tag v0.2.0
git push origin v0.2.0
```

(Use your real version number.)

---

## When things go wrong

| What you see | What it means | What to do |
|---|---|---|
| No **"Run workflow"** button | The workflow file isn't on `main` yet | Finish step 1.1 (merge the pull request) |
| A `wheel …` job has a **red cross** | Building or testing failed on that system | Don't upload. Click the red job to see the error, and ask for help with a screenshot. Nothing was uploaded. |
| `publish` says **"invalid-publisher"** or **"not a trusted publisher"** | A typo in step 1.2 or 1.3 | Compare every field letter by letter (`FennisRobert`, `emcad`, `wheels.yml`, `pypi`/`testpypi`), fix it, and run again |
| **"File already exists"** | That version was already uploaded on that site | Normal if you ran it twice. For a new upload, choose a new version number (see above) |
| The run sits on **"Waiting"** forever | It's waiting for your approval | Click **"Review deployments" → Approve and deploy** |
| The `macos-x86_64` job never starts ("waiting for a runner") | GitHub retired that machine type | Ask for help: it's a one-line change in `.github/workflows/wheels.yml` |
| You uploaded something broken to PyPI | It happens | Release a fixed version (e.g. `0.2.1`). On PyPI you can also "yank" the bad one: **Manage → Releases → Options → Yank**. That hides it from new installs without breaking people already using it. |

A run that fails before you click "Approve and deploy" **uploads nothing**. You
can't accidentally publish a broken build, because the upload step only runs
after every build and test job is green *and* you approve it.
