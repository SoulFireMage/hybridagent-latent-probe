#!/bin/sh
# Runs inside a Hugging Face Job (image python:3.11, one L4 GPU).
# Installs HybridAgent, runs each question in questions/$SET.txt through the
# Interpretable Latent Mode inspector, and prints every trace as plain text to
# the job log (the HTML reports only live inside the container).
#
#   SET     question file to use (default: traps)  -> questions/traps.txt
#   MODEL   default: togethercomputer/HybridAgent-8B-Agentic
#   REPO    where to fetch the question file and dumper from
set -u
SET="${SET:-traps}"
MODEL="${MODEL:-togethercomputer/HybridAgent-8B-Agentic}"
# TEMP    optional sampling temperature; when set, question N also gets --seed N
TEMP="${TEMP:-}"
REPO="${REPO:-https://raw.githubusercontent.com/SoulFireMage/hybridagent-latent-probe/main}"

# The search tool imports tavily at start-up even if you never search.
# Any non-empty value lets it load; searches then fail harmlessly.
export TAVILY_API_KEY="${TAVILY_API_KEY:-unused-placeholder-key}"
export HF_HUB_DISABLE_PROGRESS_BARS=yes

curl -sSL "$REPO/questions/$SET.txt" -o /tmp/questions.txt
curl -sSL "$REPO/dump_trace.py"      -o /tmp/dump_trace.py

git clone -q --depth 1 https://github.com/togethercomputer/HybridAgent.git /tmp/HybridAgent
cd /tmp/HybridAgent

pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 2>/dev/null
# Install pins one at a time: one bad pin (deepspeed, training-only) aborts a
# whole `pip install -r` and silently leaves transformers/tavily missing.
for P in $(grep -vE '^#|^torch|deepspeed|^$' requirements.txt); do
  pip install -q "$P" >/dev/null 2>&1 || echo "FAILED PIN $P"
done
pip install -q -e . --no-deps 2>/dev/null

# Show the inspector's options (sampling, temperature, ...) for the record.
python -m interpretable_latent_mode --help 2>&1 | head -n 70

N=0
while IFS= read -r Q; do
  [ -z "$Q" ] && continue
  N=$((N+1))
  SAMPLING=""
  [ -n "$TEMP" ] && SAMPLING="--temperature $TEMP --seed $N"
  python -m interpretable_latent_mode --model "$MODEL" --question "$Q" $SAMPLING \
      --output "/tmp/r$N.html" >/dev/null 2>&1 || echo "INSPECTOR FAILED on Q$N"
done < /tmp/questions.txt

python -u /tmp/dump_trace.py /tmp/r*.html
echo DONE
