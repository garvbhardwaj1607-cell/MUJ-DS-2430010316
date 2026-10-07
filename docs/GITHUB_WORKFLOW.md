# GitHub workflow (Batch F capstone guidelines)

This file maps each step of the guidelines to what has to be done on GitHub. Steps marked **(you)** need your GitHub account and cannot be done from code.

| Step | Action | Status |
|---|---|---|
| 1 Join the organisation | Accept the invitation, complete your GitHub profile | **(you)** |
| 2 Personal repository | Public repo named `MUJ-DS-2430010316` | **(you)**: push this folder (commands below) |
| 3 Instructor as collaborator | Settings -> Collaborators -> add the instructor account; do not remove it | **(you)** |
| 4 Repository structure | `README.md`, `assignments/`, `notebooks/`, `code/`, `resources/`, `presentations/`, `capstone/` | done |
| 5 README content | Name, registration number, branch, Batch F, project title, GitHub username, training programme | done (fill the marked fields) |
| 6 Weekly updates | Commit and push meaningful changes regularly | ongoing, see below |
| 7 Capstone repository | Team repo with all members and the instructor as collaborators | **(you / team)** |
| 8 Individual contributions | Code, documentation, testing, data collection, model development | recorded in the commit history |
| 9 GitHub Issues | Dataset Collection, Model Development, Testing, Documentation, Deployment | `docs/ISSUES.md`, `scripts/create_issues.sh` |
| 10 Pull requests | Branch -> PR -> review -> merge | template in `.github/pull_request_template.md` |
| 11 Deliverables | Source code, documentation, presentation, screenshots, results, installation guide | see `README.md` |
| 12 Final submission | Personal repo URL, capstone repo URL, presentation, other materials | **(you)** |

## Push this repository
```bash
cd MUJ-DS-2430010316
git remote add origin https://github.com/<your-username>/MUJ-DS-2430010316.git
git push -u origin main
```

## Create the issues
```bash
gh auth login
sh scripts/create_issues.sh <your-username>/MUJ-DS-2430010316
```

## Working on a future change (branch -> PR -> review -> merge)
```bash
git switch -c feature/colab-full-dataset
# ...edit, then:
git add -A && git commit -m "Add full-dataset Colab results (closes #9)"
git push -u origin feature/colab-full-dataset
gh pr create --fill          # ask a teammate/instructor to review, then merge on GitHub
```
Commit messages: short imperative summary, reference the issue (`closes #N`).

## Honest note on the commit history
The initial commits in this repository were created in one working session, in the order the project was built (data, models, evaluation, innovation, documentation). They are timestamped with the day they were made. The guidelines ask for continuous activity: keep committing as the open issues (full-dataset run, extra tests, deployment) are completed.
