"""Light LLM client for the scheduling agents.

Design goals:
  * light  — small model (claude-haiku-4-5 by default), one small call per
             agent per scheduling window, compact JSON in and out.
  * reproducible — every real API response is recorded on disk keyed by a
             hash of the request; re-running an experiment replays the
             recorded responses at zero cost.
  * offline — mode='mock' answers every query with deterministic
             heuristics of the same JSON shape, so the full pipeline runs
             (and is testable) without any API access. The mock also
             serves as the "no-LLM" ablation.

Structured outputs (output_config.format with a JSON schema) guarantee
that responses parse; there is no free-text fragility in the loop.
"""

import hashlib
import json
import os
import time

from .env_config import load_env, read_claude_oauth_token

OAUTH_BETA_HEADER = 'oauth-2025-04-20'


class LLMClient:
    def __init__(self, mode='mock', model='claude-haiku-4-5',
                 cache_dir='results/llm_cache', max_tokens=512):
        self.mode = mode
        self.model = model
        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latencies = []          # wall-clock seconds per live API call
        self._client = None

        self._use_oauth = False
        if mode == 'api':
            os.makedirs(cache_dir, exist_ok=True)
            load_env()  # pull any ANTHROPIC_* vars from the .env secret file
            self._connect()

    def _connect(self):
        """Resolve credentials in priority order: explicit API key, explicit
        OAuth auth token, then the Claude Pro/Max OAuth login on disk. Falls
        back to mock mode if nothing works."""
        try:
            import anthropic
        except Exception as exc:
            print(f"[llm_client] anthropic SDK unavailable ({exc}); using mock")
            self.mode = 'mock'
            return

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        auth_token = os.environ.get('ANTHROPIC_AUTH_TOKEN')

        if api_key:
            self._client = anthropic.Anthropic()
            source = 'API key'
        else:
            if not auth_token:
                auth_token, expired = read_claude_oauth_token()
                if auth_token and expired:
                    print("[llm_client] Claude OAuth token looks expired; "
                          "trying anyway (open Claude Code to refresh if this "
                          "fails)")
            if not auth_token:
                print("[llm_client] no credentials found (no ANTHROPIC_API_KEY, "
                      "no ANTHROPIC_AUTH_TOKEN, no Claude OAuth login); using mock")
                self.mode = 'mock'
                return
            self._client = anthropic.Anthropic(
                auth_token=auth_token,
                default_headers={'anthropic-beta': OAUTH_BETA_HEADER},
            )
            self._use_oauth = True
            source = 'OAuth (Claude subscription)'

        if not self._probe():
            self.mode = 'mock'
            return
        print(f"[llm_client] API connected via {source}, model={self.model}")

    def _probe(self):
        """One tiny call to confirm the credentials actually authorize
        inference. Returns True on success."""
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{'role': 'user', 'content': 'ping'}],
            )
            return True
        except Exception as exc:
            print(f"[llm_client] credential probe failed ({exc}); using mock")
            return False

    # ------------------------------------------------------------------
    def query(self, system, user, schema, mock_fn):
        """Ask the LLM for a structured decision.

        system:  role prompt for the agent
        user:    compact JSON state summary
        schema:  JSON schema the answer must satisfy
        mock_fn: deterministic fallback producing the same shape

        Returns a dict conforming to `schema`.
        """
        self.calls += 1
        if self.mode == 'mock':
            return mock_fn()

        key = hashlib.sha256(
            json.dumps([self.model, system, user, schema], sort_keys=True).encode()
        ).hexdigest()[:24]
        cache_path = os.path.join(self.cache_dir, f'{key}.json')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        try:
            _t0 = time.time()
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config={'format': {'type': 'json_schema', 'schema': schema}},
                messages=[{'role': 'user', 'content': user}],
            )
            self.latencies.append(time.time() - _t0)
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            text = next(b.text for b in response.content if b.type == 'text')
            result = json.loads(text)
        except Exception as exc:
            print(f"[llm_client] API call failed ({exc}); using mock decision")
            return mock_fn()

        with open(cache_path, 'w') as f:
            json.dump(result, f)
        return result
