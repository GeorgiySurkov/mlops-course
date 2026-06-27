#!/usr/bin/env bash
# Stage 4: deploy the published Registry model with ClearML Serving. See README/Makefile
# for how to run it.
#
# The file server is addressed by this Mac's mDNS name (<LocalHostName>.local), not an IP.
# That name resolves both on the host (Bonjour) and from the Colima containers, and it
# follows your IP across DHCP changes, so there's no /etc/hosts edit and no hardcoded IP.
# The model weights and preprocess live under that host, so the serving container must be
# able to reach it.
set -euo pipefail
cd "$(dirname "$0")/.."

ENDPOINT="${ENDPOINT:-sentiment}"
VERSION="${VERSION:-1}"
SERVICE_NAME="${SERVICE_NAME:-sentiment-serving}"
HOST_PORT="${CLEARML_SERVING_HOST_PORT:-8085}"
ENV_FILE="infra/clearml-serving/serving.env"
PREPROCESS="$(pwd)/serving/preprocess.py"
COMPOSE="infra/clearml-serving/docker-compose.yml"

# project / model name come from config/config.yaml (single source of truth)
PROJECT=$(grep -E '^\s*project_name:' config/config.yaml | head -1 | sed -E 's/.*: *"?([^"]+)"?.*/\1/')
MODEL_NAME=$(grep -E '^\s*model_name:' config/config.yaml | head -1 | sed -E 's/.*: *"?([^"]+)"?.*/\1/')

# Stable, IP-independent host alias reachable from the host and the containers: this
# Mac's mDNS name. Override with SERVING_HOST=<name> if your setup differs.
SERVING_HOST="${SERVING_HOST:-$(scutil --get LocalHostName 2>/dev/null).local}"
[ "$SERVING_HOST" = ".local" ] && { echo "!! could not detect mDNS host name; set SERVING_HOST=<name>" >&2; exit 1; }
echo ">> SERVING_HOST=$SERVING_HOST  project='$PROJECT'  model='$MODEL_NAME'"

# Credentials from ~/clearml.conf. Handles both formats: HOCON (access_key: "...") and
# the JSON-ish block clearml-init writes ("access_key": "...").
read_conf() { python3 -c "import re,os;c=open(os.path.expanduser('~/clearml.conf')).read();m=re.search(r'$1'+r'\"?\s*[:=]\s*\"([^\"]+)\"',c);print(m.group(1) if m else '')"; }
ACCESS=$(read_conf access_key); SECRET=$(read_conf secret_key)
[ -z "$ACCESS" ] && { echo "!! no credentials in ~/clearml.conf — run 'uv run clearml-init'" >&2; exit 1; }

echo ">> creating serving service '$SERVICE_NAME'"
CREATE_OUT=$(clearml-serving create --name "$SERVICE_NAME")
echo "$CREATE_OUT"
SERVICE_ID=$(echo "$CREATE_OUT" | grep -oE 'id=[A-Za-z0-9]+' | head -1 | cut -d= -f2)
[ -z "$SERVICE_ID" ] && { echo "!! failed to parse serving service id" >&2; exit 1; }
echo ">> serving service id=$SERVICE_ID"

if [ -n "${MODEL_ID:-}" ]; then
  echo ">> adding endpoint '$ENDPOINT' from model-id $MODEL_ID"
  clearml-serving --id "$SERVICE_ID" model add --engine sklearn \
    --endpoint "$ENDPOINT" --version "$VERSION" \
    --model-id "$MODEL_ID" --preprocess "$PREPROCESS"
else
  echo ">> adding endpoint '$ENDPOINT' from published model '$MODEL_NAME' in '$PROJECT'"
  clearml-serving --id "$SERVICE_ID" model add --engine sklearn \
    --endpoint "$ENDPOINT" --version "$VERSION" \
    --project "$PROJECT" --name "$MODEL_NAME" --published --preprocess "$PREPROCESS"
fi

cat > "$ENV_FILE" <<EOF
CLEARML_WEB_HOST=http://${SERVING_HOST}:8080
CLEARML_API_HOST=http://${SERVING_HOST}:8008
CLEARML_FILES_HOST=http://${SERVING_HOST}:8081
CLEARML_API_ACCESS_KEY=${ACCESS}
CLEARML_API_SECRET_KEY=${SECRET}
CLEARML_SERVING_TASK_ID=${SERVICE_ID}
CLEARML_SERVING_HOST_PORT=${HOST_PORT}
CLEARML_EXTRA_PYTHON_PACKAGES=scikit-learn==1.4.2 numpy==1.26.4 scipy==1.13.1 joblib==1.4.2 pandas==2.2.2
EOF
echo ">> wrote $ENV_FILE"

docker compose --env-file "$ENV_FILE" -f "$COMPOSE" up -d
echo ">> serving stack up. Endpoint: http://localhost:${HOST_PORT}/serve/${ENDPOINT}/${VERSION}"
echo ">> first request is slow (model is pulled from the registry); then:"
echo "   curl -X POST http://localhost:${HOST_PORT}/serve/${ENDPOINT}/${VERSION} -H 'Content-Type: application/json' -d '{\"text\":\"a wonderful delightful film\"}'"
