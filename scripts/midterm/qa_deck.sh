#!/usr/bin/env bash
# Validate and render the mid-term deck inside the LibreOffice container (isco-office-qa):
#   bash scripts/midterm/qa_deck.sh runs/midterm_reports/deck/ISCO_중간발표.pptx
# -> <deck dir>/qa/{validate.txt, text.md, slide-NN.jpg}
set -euo pipefail
DECK=$(realpath "$1")
DIR=$(dirname "$DECK")
NAME=$(basename "$DECK")
SKILL=/home/juk0324/.claude/skills/synced/e6b92801-4b10-458d-b522-c15c09f25ac8_eabd7edf-f36a-4e11-828c-0b283ca82af5/pptx
mkdir -p "$DIR/qa"
rm -f "$DIR"/qa/slide-*.jpg
FONTS=$(realpath "$(dirname "$0")/../../runs/midterm_reports/docker/fonts.conf")
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e FONTCONFIG_FILE=/fonts.conf -v "$FONTS:/fonts.conf:ro" \
  -v "$DIR:/work" -v "$SKILL:/skill:ro" isco-office-qa bash -c "
  cd /work
  python /skill/scripts/office/validate.py '$NAME' > qa/validate.txt 2>&1 || true
  python /skill/scripts/slide_text.py '$NAME' > qa/text.md 2>&1 || true
  cp '$NAME' /tmp/deck.pptx
  soffice -env:UserInstallation=file:///tmp/lo_profile --headless --convert-to pdf --outdir /tmp /tmp/deck.pptx > qa/convert.log 2>&1
  cp /tmp/deck.pdf qa/deck.pdf
  pdftoppm -jpeg -r 80 /tmp/deck.pdf qa/slide
"
tail -3 "$DIR/qa/validate.txt"
ls "$DIR"/qa/slide-*.jpg | wc -l
