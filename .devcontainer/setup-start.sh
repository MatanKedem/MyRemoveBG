#!/usr/bin/env bash
set -e

BASHRC="$HOME/.bashrc"

if ! grep -qF '# MyRemoveBG start command' "$BASHRC" 2>/dev/null; then
  cat >> "$BASHRC" <<'EOF'

# MyRemoveBG start command
start() {
  local project_dir="/workspaces/MyRemoveBG"
  if [[ ! -f "$project_dir/mvnw" ]]; then
    echo "MyRemoveBG workspace was not found at $project_dir"
    return 1
  fi
  (cd "$project_dir" && ./mvnw spring-boot:run)
}
EOF
fi

source "$BASHRC"
echo "הפקודה start מוכנה. מעכשיו אפשר להפעיל את MyRemoveBG עם: start"
