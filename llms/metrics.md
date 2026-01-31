# Metrics

Metrics to measure the effectiveness and efficiency of LLMs for designing a protocol.

| Indicator | Type | Desc |
| --- | --- | --- |
| One-Shot | Bool | Give correct answer directly |
| Feedback | Times | Times of feedback to answer correctly (limit: 10 times, `inf` means cannot achieve) |
| Elapse | Int | How long it takes to answer correctly |
| Small-Scale | Bool | Can it work under a basic configuration |
| Large-Scale | Bool | Can it generalize to more-nodes configuration |
