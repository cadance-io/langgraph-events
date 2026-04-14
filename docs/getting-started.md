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

## Running the Graph

```python
# Synchronous — returns the full EventLog when the graph completes
log = graph.invoke(MessageReceived(text="hello"))

# Multiple seed events — useful for system prompts + user input
log = graph.invoke([
    SystemPromptSet.from_str("You are helpful"),
    UserMessageReceived(message=HumanMessage(content="Hi")),
])

# Async — same API, awaitable
log = await graph.ainvoke(MessageReceived(text="hello"))

# Stream events as they're produced — for live UI updates
for event in graph.stream_events(MessageReceived(text="hello")):
    print(event)

# Stream with reducer snapshots — see accumulated state each round
for frame in graph.stream_events(MessageReceived(text="hello"), include_reducers=True):
    print(frame.event, frame.reducers["messages"])
```

## Common Tasks

| I want to... | Reach for... | Docs |
|---------------|-------------|------|
| Query past events in a handler | `EventLog` (`log.filter()`, `log.latest()`) | [Concepts](concepts.md#eventlog) |
| Accumulate message history | `message_reducer()` | [Reducers](reducers.md#message_reducer) |
| Fan out parallel work | `Scatter` | [Control Flow](control-flow.md#scatter) |
| Pause for human approval | `Interrupted` + `graph.resume()` | [Control Flow](control-flow.md#interrupted-resumed) |
| Stop the graph early | Return a `Halted` subclass | [Concepts](concepts.md#halted) |
| Catch handler exceptions | `@on(..., raises=MyError)` + `@on(HandlerRaised, exception=MyError)` | [Control Flow](control-flow.md#handler-exceptions) |
| Stream LLM tokens in real time | `astream_events(include_llm_tokens=True)` | [Streaming](streaming.md) |
| Connect to an AG-UI frontend | `AGUIAdapter` | [AG-UI Adapter](agui.md) |

See [Concepts](concepts.md) for the core model, then explore the topics above as needed.
