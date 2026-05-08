#!/usr/bin/env bash
# cron_auto_train.sh — spúšťaj z crontabu, napr. každú hodinu
#
# Crontab záznam (crontab -e):
#   0 * * * * /home/vojtech/VideoTranslator_studio/mini_level11/cron_auto_train.sh
#
# Prípadne každé 4 hodiny:
#   0 */4 * * * /home/vojtech/VideoTranslator_studio/mini_level11/cron_auto_train.sh

set -euo pipefail

PYTHON="/home/vojtech/miniforge3/envs/chatterbox_env/bin/python3"
SCRIPT="/home/vojtech/VideoTranslator_studio/mini_level11/auto_train_trigger.py"

# Level 21 ingest adresáre — pridaj vlastné cesty ak máš viac zdrojov
INGEST_DIRS=(
    "/home/vojtech/VideoTranslator_studio/mini_level11/level21_ingest"
)

exec "$PYTHON" "$SCRIPT" \
    --ingest-dirs "${INGEST_DIRS[@]}" \
    --threshold 200 \
    --training-timeout 14400
