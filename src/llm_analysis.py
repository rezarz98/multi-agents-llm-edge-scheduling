"""LLM control-plane evaluation: control-plane latency,
decision-distribution of the monitor agent, and a rationale sample.
Runs HAS once on a fresh instance with live model calls.
"""
import json
import statistics as st

from lib.instance_gen import make_instance
from lib.sim_core import make_edge_servers, preset_for
from schedulers.cnp_core import ContractNetScheduler

df = make_instance(101, 1)                      # fresh, uncached instance
servers = make_edge_servers(preset=preset_for(df))
sched = ContractNetScheduler(servers, df.copy(), window_size=40,
                             consult_llm=True, monitor_kind='llm',
                             llm_mode='api', review_every=2, seed=42)
metrics = sched.run()
sched.bus.save('results/has_analysis_trace.jsonl')

lat = sched.llm.latencies
trace = sched.bus.trace
adjusts = [m for m in trace if m['type'] == 'PolicyAdjust']
edge_dec = [m for m in trace if m['type'] == 'PolicyDecision' and m['sender'].startswith('edge')]

# how often the monitor actually changed the parameter vector
changes, prev = 0, None
for a in adjusts:
    p = (a['params']['order_policy'], a['params']['ntc_defer_windows'])
    if prev is not None and p != prev:
        changes += 1
    prev = p

report = {
    'llm_calls': sched.llm.calls,
    'live_api_calls': len(lat),
    'latency_s': {
        'mean': round(st.mean(lat), 3) if lat else None,
        'median': round(st.median(lat), 3) if lat else None,
        'p95': round(sorted(lat)[int(0.95 * len(lat)) - 1], 3) if lat else None,
        'max': round(max(lat), 3) if lat else None,
    },
    'input_tokens': sched.llm.input_tokens,
    'output_tokens': sched.llm.output_tokens,
    'monitor_reviews': len(adjusts),
    'monitor_param_changes': changes,
    'edge_decline_ntc_rate': round(
        sum(1 for e in edge_dec if e['decision'].get('accept_ntc') is False)
        / max(1, len(edge_dec)), 3),
    'rationales_sample': [
        {'params': a['params'], 'rationale': a['rationale']} for a in adjusts[:6]
    ],
}
with open('results/llm_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps({k: v for k, v in report.items() if k != 'rationales_sample'}, indent=2))
print("\nmonitor rationales:")
for r in report['rationales_sample']:
    print(f"  {r['params']['order_policy']}/defer{r['params']['ntc_defer_windows']}: {r['rationale']}")
