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
# ReplyProduced(text='Routing you to support...')
```

## How It Works

`EventGraph` compiles your handlers into a LangGraph `StateGraph` with a hub-and-spoke reactive loop:

```
seed event
    |
    v
[seed] --> [dispatch] --> handler_a --+
                ^          handler_b --+
                |                      |
             [router] <----------------+
                |
                v
             [dispatch] --> handler_c --+
                ^                       |
                |                       |
             [router] <-----------------+
                |
                v
             [dispatch] --> END (no pending events)
```

1. A **seed event** enters the graph.
2. The **router** collects new events, then **dispatch** matches each to subscribed handlers via `isinstance`. Matched handlers run and emit new events.
3. The loop repeats until no handler matches or a `Halted` event appears.

## Useful Operations

```python
# Synchronous
log = graph.invoke(MessageReceived(text="hello"))

# Multiple seed events
log = graph.invoke([
    SystemPromptSet.from_str("You are helpful"),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])

# Async
log = await graph.ainvoke(MessageReceived(text="hello"))

# Stream produced events
for event in graph.stream_events(MessageReceived(text="hello")):
    print(event)

# Stream with reducer snapshots
for frame in graph.stream_events(MessageReceived(text="hello"), include_reducers=True):
    print(frame.event, frame.reducers["messages"])
```

See [Concepts](concepts.md) for event log queries, reducers, interruption/resume, and fan-out.
