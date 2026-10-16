#!/usr/bin/env bash
# Create the GitHub repo and push. Run once from the repo root on your own machine.
#   bash scripts/publish_to_github.sh <github-user-or-org>/<repo-name> [--public]
# Needs the GitHub CLI (https://cli.github.com) logged in: `gh auth login`.
set -euo pipefail
REPO=${1:?usage: publish_to_github.sh owner/name [--public]}
VIS=${2:---private}
gh repo create "$REPO" "$VIS" --source . --remote origin --description \
  "Six solvers for NxNx1 and NxNxN Rubik's puzzles (3<=N<=20) on one shared engine"
git push -u origin main
# protect main: PRs + green CI required
gh api -X PUT "repos/$REPO/branches/main/protection" --input - <<JSON || echo "(branch protection needs a paid plan for private repos; set it in Settings > Branches)"
{"required_status_checks":{"strict":true,"contexts":["test"]},"enforce_admins":false,
 "required_pull_request_reviews":{"required_approving_review_count":1},"restrictions":null}
JSON
echo "Done. Invite teammates: gh api -X PUT repos/$REPO/collaborators/<their-github-username>"
