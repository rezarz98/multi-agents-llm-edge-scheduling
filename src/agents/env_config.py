"""Zero-dependency loader for the .env secret file.

Searches upward from the current directory for a `.env` file and loads
KEY=VALUE lines into os.environ (without overwriting variables already
set in the real environment). Keeps the Anthropic API key out of source
code and out of version control.
"""

import json
import os
import time

CLAUDE_CREDS = os.path.expanduser('~/.claude/.credentials.json')


def read_claude_oauth_token(path=CLAUDE_CREDS):
    """Read the current OAuth access token from the Claude Code credential
    store (a Claude Pro/Max login). Reading it live means we always get the
    token Claude Code keeps refreshed. Returns (token, expired) or
    (None, False) if unavailable. `expired` is advisory — clock skew and
    refresh timing mean an 'expired' token may still work and vice versa."""
    try:
        with open(path) as f:
            oauth = json.load(f).get('claudeAiOauth', {})
    except (OSError, ValueError):
        return None, False
    token = oauth.get('accessToken')
    if not token:
        return None, False
    expires_at = oauth.get('expiresAt')  # epoch milliseconds
    expired = bool(expires_at) and (expires_at / 1000.0) < time.time()
    return token, expired


def load_env(max_levels=4):
    """Load the nearest `.env` into os.environ. Returns the path loaded,
    or None if no file was found. Existing environment variables win, so
    an explicitly exported ANTHROPIC_API_KEY overrides the file."""
    here = os.path.abspath(os.getcwd())
    for _ in range(max_levels + 1):
        candidate = os.path.join(here, '.env')
        if os.path.isfile(candidate):
            _load_file(candidate)
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def _load_file(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def has_api_key():
    return bool(os.environ.get('ANTHROPIC_API_KEY'))
