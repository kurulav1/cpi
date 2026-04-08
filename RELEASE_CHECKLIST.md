# Release Checklist

Use this checklist before tagging a release.

## 1) Security and Secrets

- Ensure no real credentials are present in tracked files.
- Keep local secrets in `.env` only (ignored by git).
- Verify `.env.example` contains placeholders only.
- If any token was ever exposed, rotate/revoke it before release.

Quick scan:

```powershell
rg -n "hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|api[_-]?key|secret|token\\s*=" --glob "!web/node_modules/**" --glob "!third_party/**"
```

## 2) Clean Repository State

- Ensure generated outputs are ignored (`build/`, `artifacts/`, `web/node_modules/`, `web/dist/`, logs).
- Confirm there are no large model binaries in tracked source paths.

## 3) Build Validation

### Native runtime

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target llama_infer
```

### Web app

```powershell
cd web
npm ci
npm run build
cd ..
```

### Python tool scripts sanity

```powershell
python -m compileall tools
```

## 4) Runtime Smoke Checks

```powershell
.\build\Release\llama_infer.exe --help
.\build\Release\llama_infer.exe fake_model.ll2c --unknown-flag
```

Expected:

- Help/usage prints correctly.
- Invalid flags return a clear argument error.

## 5) Release Metadata

- Update `README.md` if CLI flags/config changed.
- Ensure version/tag notes include major runtime changes.
- Verify license choice is present and correct for publication target.

## 6) Tag and Publish

- Create annotated tag (`vX.Y.Z`).
- Publish release notes with:
  - supported platforms
  - known limitations
  - migration notes (if any)
