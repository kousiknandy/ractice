from pathlib import Path

def gen_filenames(top_dir, path_filter):
    yield from Path(top_dir).rglob(path_filter)

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

    def __call__(self, line):
        matches = self.logpat.match(line)
        res = {k[0]: v if len(k) == 1 else k[1](v) for k,v in zip(self.schema, matches.groups())}
        return res

import csv

class csvParser:
    def __init__(self):
        self.headers = None

    def __call__(self, log):
        if not self.headers:
            self.headers = list(csv.reader([log]))[0]
            return {k: None for k in self.headers}
        values = list(csv.reader([log]))[0]
        return dict(zip(self.headers, values))
    
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

class tee:
    def __call__(self, log):
        print(log)
        
def gen_parse(lines, parser):
    for line in lines:
        yield parser(line)

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
            
f = gen_filenames("/Users/kousiknandy/Workspace/ractice/log_processor/logs/apache/", "*.txt")
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


f = gen_filenames("/Users/kousiknandy/Workspace/ractice/log_processor/logs/apps/", "*.csv")
l = gen_lines(f)
p = gen_parse(l, csvParser())
pipeline2 = [
    [
        filters("Daily Charge (GBP)", operator.eq, "-14.50"),
        [
            [tee()]
        ]
    ]
]
gen_process_chain(p, pipeline2)
