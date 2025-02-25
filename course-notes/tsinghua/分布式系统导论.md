Two key words:
1. Abstraction
2. Tradeoff

Tradeoff between throughput and latency
latency for a single user, throughput for the whole system
E.g. large batch size will benefit the throughput but not the latency, the serving system focus on throughput because this brings more money

Different upper system use similar infrastructures (store, compute, coordinate)

Learn how to debug in distributed systems in the labs

Vertical: scale-up
Horizontal: scale-out (multiple machines)

Strong consistency can only be defined if we have a globally synchronized clock time steps that assign a time step to each operation. This is very expensive.

Internet generally provides eventual consistency.

