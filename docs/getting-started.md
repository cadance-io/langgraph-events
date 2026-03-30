# Getting Started

## Minimal Example

```python
from langgraph_events import Event, EventGraph, on


class MessageReceived(Event):
    text: str


class MessageClassified(Event):
    label: str


class ReplyProduced(Event):
    text: str


@on(MessageReceived)
def classify(event: MessageReceived) -> MessageClassified:
    if "help" in event.text.lower():
        return MessageClassified(label="support")
    return MessageClassified(label="general")


@on(MessageClassified)
def respond(event: MessageClassified) -> ReplyProduced:
    if event.label == "support":
        return ReplyProduced(text="Routing you to support...")
    return ReplyProduced(text="Thanks for your message!")


graph = EventGraph([classify, respond])
log = graph.invoke(MessageReceived(text="I need help with my order"))

print(log.latest(ReplyProduced))
```

## How It Works

1. A seed event enters the graph.
2. The dispatcher matches events to subscribed handlers.
3. Handler outputs become new events.
4. Processing repeats until there are no pending events, or a `Halted` event appears.

```text
seed -> dispatch -> handlers -> dispatch -> ... -> end
```

## Useful Operations

```python
# Sync
log = graph.invoke(MessageReceived(text="hello"))

# Async
log = await graph.ainvoke(MessageReceived(text="hello"))

# Stream produced events
for event in graph.stream_events(MessageReceived(text="hello")):
    print(event)
```

See [Concepts](concepts.md) for event log queries, reducers, interruption/resume, and fan-out.
