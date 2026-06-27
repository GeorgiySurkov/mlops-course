#!/usr/bin/env bash
# Run a ClearML Agent (via uvx) on the students queue. The queue is auto-created.
# Each task gets a fresh venv with its auto-detected packages, fine for CPU text
# classification. The agent runs in an ephemeral uvx tool env, so there is no
# .venv-agent to manage.
#
# Prereqs: server up (make server-up) and ~/clearml.conf configured.
set -euo pipefail

QUEUE="${CLEARML_QUEUE:-students}"
AGENT_VERSION="${CLEARML_AGENT_VERSION:-2.0.7}"

echo ">> starting clearml-agent on queue '${QUEUE}' (foreground; Ctrl-C to stop)"
# clearml-agent still imports pkg_resources, which setuptools>=81 dropped, so pin <81.
exec uvx --with "setuptools<81" --from "clearml-agent==${AGENT_VERSION}" clearml-agent daemon \
  --queue "$QUEUE" --create-queue --foreground
