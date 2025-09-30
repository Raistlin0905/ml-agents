# GitHub Contribution Guidelines

## Branch Naming Convention

- Features: `feature/{feature-name}`
- Bug fixes: `fix/{bug-description}`

## Workflow

### 1. Update Main Branch

```bash
git checkout main
git pull origin main
```

### 2. Create New Branch

For a feature:
```bash
git checkout -b feature/add-user-dashboard
```

For a bug fix:
```bash
git checkout -b fix/login-button-crash
```

### 3. Make Changes and Commit

```bash
git add .
git commit -m "Add user dashboard with profile display"
```

### 4. Push to GitHub

```bash
git push origin feature/add-user-dashboard
```

### 5. Create Pull Request

1. Go to https://github.com/Raistlin0905/ml-agents
2. Click "Compare & pull request"
3. Fill in title and description
4. Click "Create pull request"

### 6. Address Review Comments (if needed)

```bash
git add .
git commit -m "Address review comments"
git push origin feature/add-user-dashboard
```

### 7. After Merge

```bash
git checkout main
git pull origin main
git branch -d feature/add-user-dashboard
```

## Common Commands

```bash
# Check status
git status

# View branches
git branch -a

# Switch branch
git checkout branch-name

# View commit history
git log --oneline
```

## Complete Example

```bash
# Update main
git checkout main
git pull origin main

# Create branch
git checkout -b feature/add-leaderboard

# Make changes and commit
git add .
git commit -m "Add leaderboard page with top 10 scores"

# Push
git push origin feature/add-leaderboard

# Create PR on GitHub, wait for approval, merge

# Clean up
git checkout main
git pull origin main
git branch -d feature/add-leaderboard
```

## Rules

- Never commit directly to main
- Always use pull requests
- One feature or fix per branch
