from pathlib import Path

def gen_filenames(top_dir):
    yield from Path(top_dir).rglob("*.txt")

def gen_lines(filenames):
    for filename in filenames:
        yield from open(filename)

import re
from datetime import datetime

class apacheParser:
    # 179.26.146.28 - - [Jul 13 2026, 07:27:32] "GET /checkout HTTP/1.1" 304 0 "https://www.twitter.com" "Mozilla/5.0"
    log_regex = r'(\S+) (\S+) (\S+) \[(.*?)\] "(\S+) (\S+) (\S+)" (\S+) (\S+) "(\S+)" "(.*?)"'
    schema = (("src_ip",), ("-",), ("user",), ("timestamp", lambda t: datetime.strptime(t, "%b %d %Y, %H:%M:%S")),
              ("method",), ("url",), ("http_version",), ("code", lambda c: int(c) if c != "-" else 0),
              ("size", lambda s: int(s) if s != "-" else 0), ("referrer",), ("user_agent",))
    def __init__(self):
        self.logpat = re.compile(self.log_regex)

    def parse(self, line):
        matches = self.logpat.match(line)
        res = {k[0]: v if len(k) == 1 else k[1](v) for k,v in zip(self.schema, matches.groups())}
        return res

class accumulator:
    def __init__(self, field):
        self.field = field
        self.value = 0

    def __call__(self, log):
        self.value += log.get(self.field, 0)

from collections import defaultdict

class histogram:
    def __init__(self, field):
        self.field = field
        self.value = defaultdict(int)

    def __call__(self, log):
        self.value[log.get(self.field, "-")] += 1
    
def gen_parse(lines, parser):
    for line in lines:
        yield parser.parse(line)

def gen_process(parseds, *processors):
    for f in parseds:
        for p in processors:
            p(f)
            
class filters:
    def __init__(self, field, op, val):
        self.field = field
        self.op = op
        self.val = val

    def __call__(self, log):
        return log if self.op(log.get(self.field, 0), self.val) else None
            
def gen_process_chain(parseds, chains):
    for p in parseds:
        for c in chains:
            proc_chain(p, c)

def proc_chain(line, chain):
    l = chain[0](line)
    if l and len(chain) > 1:
        for c in chain[1]:
            proc_chain(l, c)
            
f = gen_filenames("/Users/kousiknandy/Workspace/ractice/log_processor/logs/apache/")
l = gen_lines(f)
p = gen_parse(l, apacheParser())

import operator

pipeline = [
    [
        filters("method", operator.eq, "PUT"),
        [
            [histogram("code")],
            [accumulator("size")]
        ]
    ],
    [
        filters("code", operator.ge, 500),
        [
            [histogram("url")]
        ]
    ]
]

gen_process_chain(p, pipeline)

pipeline[1][1][0][0].value.keys()
