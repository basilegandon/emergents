#!/bin/bash
# Run this script to add useful labels to your repository
# Usage: ./setup-labels.sh

REPO="Jojobarbarr/emergents"

# Delete default labels that we don't need
gh label delete "good first issue" --repo $REPO --yes || true
gh label delete "help wanted" --repo $REPO --yes || true

# Create priority labels
gh label create "priority:high" --color d73a4a --description "High priority" --repo $REPO || true
gh label create "priority:medium" --color fbca04 --description "Medium priority" --repo $REPO || true
gh label create "priority:low" --color 0075ca --description "Low priority" --repo $REPO || true

# Create type labels
gh label create "type:feature" --color a2eeef --description "New feature or request" --repo $REPO || true
gh label create "type:bug" --color d73a4a --description "Something isn't working" --repo $REPO || true
gh label create "type:docs" --color 0075ca --description "Documentation" --repo $REPO || true
gh label create "type:refactor" --color e4e669 --description "Code refactoring" --repo $REPO || true
gh label create "type:test" --color c2e0c6 --description "Testing related" --repo $REPO || true

# Create status labels
gh label create "status:blocked" --color b60205 --description "Blocked by other issues" --repo $REPO || true
gh label create "status:in-progress" --color fbca04 --description "Currently being worked on" --repo $REPO || true
gh label create "status:needs-review" --color 0052cc --description "Needs code review" --repo $REPO || true

# Create area labels
gh label create "area:genome" --color f9d0c4 --description "Genome module" --repo $REPO || true
gh label create "area:mutations" --color f9d0c4 --description "Mutations module" --repo $REPO || true
gh label create "area:ci-cd" --color 1d76db --description "CI/CD pipeline" --repo $REPO || true

echo "Labels created successfully!"
