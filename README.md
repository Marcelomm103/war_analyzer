# war_analyzer

The existing analyzer programs remain separate modules under `src/` so their
current script behavior is preserved. Install the modules and optional
dependencies from this directory:

The Python analyzer package declares Python 3.11 or newer; CI validates the
3.11 minimum. Higher versions are in the declared range but are not claimed as
tested until added to the CI matrix. The same minimum applies to the vision
extra because its security-fixed Pillow pin requires Python 3.11.

```bash
python -m pip install -e ".[vision]"
python -m pip install -e ".[test]"
python -m pip install -e ".[quality]"
```

The `vision` extra pins the Python packages used for image processing, screen
capture, keyboard input, and OCR. Direct dependency versions are recorded in
`pyproject.toml`.

## System requirements

`pytesseract` is only a Python wrapper; the Tesseract OCR executable and its
trained-data files must be installed separately and available on `PATH`.
The current scripts do not pass a language explicitly, so Tesseract's default
English (`eng`) data must be available. On Debian or Ubuntu, install it with:

```bash
sudo apt install tesseract-ocr
```

Check the installation with `tesseract --version` and `tesseract --list-langs`.
If the executable is not on `PATH`, configure
`pytesseract.pytesseract.tesseract_cmd` before calling OCR.

The scripts use OpenCV highgui functions such as `imshow`; install the
`opencv-python` package from the `vision` extra and run them in a graphical
desktop session. They are not headless applications.

`pynput` is an optional vision dependency under LGPLv3 and is used by the
screen-capture prototype. Preserve its license notices when distributing it.

## Frontend

The TypeScript/React/Vite shell is in `pwa/`. Use Node.js 22.23.4 (see
`.nvmrc`) and npm 10.9.9:

```bash
cd pwa
npm ci
npm run dev
npm run typecheck
npm run build
```

To regenerate the checked-in contract schema and TypeScript types, first
install `war_game_engine` in the active Python environment, then run
`npm run generate:contracts` from `pwa/`.

This is a buildable frontend scaffold, not yet an installable/offline PWA.
Manifest, service-worker, session, voice, and acknowledgement behavior remain
in roadmap phase 13. The existing analyzer scripts are excluded from Ruff and
mypy until their characterization in step 10.1.
