"""In-process message bus with a full JSONL trace.

The trace is a paper artifact: it records every announce/bid/award and
every policy decision the agents take, so a negotiation can be replayed
and visualised after a run.
"""

import json


class MessageBus:
    def __init__(self):
        self.trace = []

    def send(self, message):
        self.trace.append(message.to_dict())
        return message

    @property
    def count(self):
        return len(self.trace)

    def save(self, path):
        with open(path, 'w') as f:
            for entry in self.trace:
                f.write(json.dumps(entry, default=str) + '\n')
