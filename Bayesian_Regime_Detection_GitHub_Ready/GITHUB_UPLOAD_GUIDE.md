# GitHub Upload Guide

1. Create a **private** GitHub repository.
2. Extract this ZIP.
3. Open PowerShell in the extracted folder.
4. Run:

```powershell
git init
git add .
git commit -m "Initial Bayesian regime detection engine"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not upload passwords, API keys, tokens or private credentials.
