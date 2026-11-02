# Patterns

Runnable examples in `examples/`. Diagrams auto-generated from each example's `graph.namespaces()` (always in sync with code).

| If you need…                          | See                                       | Related docs                                                          |
| ------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------- |
| Invariants + pinned handlers          | [Order](#order)                       | [control-flow](control-flow.md#invariants)                            |
| Approval on a paused thread           | [Expense](#expense-hitl)                  | [control-flow](control-flow.md#interrupted-resumed)                   |
| Tool-calling agent + AG-UI streaming  | [Conversation](#conversation-agui)        | [agui](agui.md), [reducers](reducers.md#message_reducer)              |
| Supervisor loop / fan-in              | [Task](#supervisor)                       | [reducers](reducers.md#reducer)                                       |
| Scatter fan-out / map-reduce          | [Batch](#scatter-fan-out)                 | [control-flow](control-flow.md#scatter)                               |
| Safety gates + live streaming         | [Content](#content-pipeline)              | [streaming](streaming.md), [concepts](concepts.md#system-events)      |
| Retries + escalation via `retry=`     | [Question](#error-recovery)               | [control-flow](control-flow.md#retries)                               |
| Renamed event class, old checkpoints  | [event_migration.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/event_migration.py) | [event-migrations](event-migrations.md) |
| Agent diagnoses a run with `query_log` | [reflection_agent.py](https://github.com/cadance-io/langgraph-events/blob/main/examples/reflection_agent.py) | [reflection](reflection.md) |

<!-- autogen:start:legend -->
<details markdown="1">
<summary>🗝️ Diagram vocabulary</summary>

```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    classDef ctx fill:none,stroke:#9ca3af,color:#9ca3af,stroke-dasharray:3 3
    classDef muted fill:none,stroke:#9ca3af,color:#9ca3af,stroke-dasharray:1 3,font-style:italic
    subgraph Example["Namespace"]
      direction LR
      Command{{Command}}:::cmd
      DomainEvent(DomainEvent):::devt
      Halted([Halted]):::halt
      Invariant{Invariant}:::inv
      Rejected(Rejected):::devt
      Reducer[(Reducer)]:::reducer
    end
    IntegrationEvent[/IntegrationEvent/]:::intg
    SystemEvent([SystemEvent]):::syst
    Context(Context: outside a focus):::ctx
    Muted(Muted: faded by the caller):::muted
    _seed_[ ]:::entry ==> Command
    Command --> DomainEvent
    Command -.->|"(raises)"| SystemEvent
    Command -.->|scatter| IntegrationEvent
    Command -.- Halted
    Command -.->|invariant| Invariant -.->|reactor| Rejected
    DomainEvent -->|"reactor [orchestrate]"| Command
    Command -->|"[chain]"| Command
    Command -.->|"(retry)"| SystemEvent
    DomainEvent -.->|folds| Reducer
    linkStyle 2 stroke:#6b7280,stroke-dasharray:3 3
    linkStyle 3 stroke:#7c3aed,stroke-width:2.5px,stroke-dasharray:8 3
    linkStyle 4 stroke:#9ca3af,stroke-dasharray:3 3
    linkStyle 5,6 stroke:#c2410c,stroke-dasharray:4 2
    linkStyle 7 stroke:#0369a1,stroke-width:3px
    linkStyle 8 stroke:#b91c1c,stroke-width:2px,stroke-dasharray:5 3
    linkStyle 9 stroke:#0891b2,stroke-dasharray:2 4
    linkStyle 10 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

</details>
<!-- autogen:end -->

## Invariants & handlers (Order namespace) { #order }

- Commands: `Place`, `Ship`. Outcomes: `Placed`, `Rejected`, `Shipped`.
- Invariants: `CustomerNotBanned`, `OrderTotalWithinLimit`. Pinned handlers turn a violation into a namespace-level `Order.Rejected`.
- State: `current_status` (`ScalarReducer` as a namespace attribute).

<!-- autogen:start:order -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    subgraph Order["Order namespace"]
        direction LR
        Place{{Place}}:::cmd
        Placed(Placed):::devt
        Rejected(Rejected):::devt
        Ship{{Ship}}:::cmd
        Shipped(Shipped):::devt
        CustomerNotBanned{CustomerNotBanned}:::inv
        OrderTotalWithinLimit{OrderTotalWithinLimit}:::inv
        _reducer_current_status[(current_status)]:::reducer
    end
    _e0_[ ]:::entry ==> Place
    _e1_[ ]:::entry ==> Ship
    Place --> Placed
    Ship --> Shipped
    Place -.->|folds| _reducer_current_status
    Placed -.->|folds| _reducer_current_status
    Rejected -.->|folds| _reducer_current_status
    Ship -.->|folds| _reducer_current_status
    Shipped -.->|folds| _reducer_current_status
    CustomerNotBanned -.->|explain_banned| Rejected
    OrderTotalWithinLimit -.->|explain_over_limit| Rejected
    Place -.->|invariant| CustomerNotBanned
    Place -.->|invariant| OrderTotalWithinLimit
    linkStyle 9,10,11,12 stroke:#c2410c,stroke-dasharray:4 2
    linkStyle 4,5,6,7,8 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/order.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/order.graph.md)
<!-- autogen:end -->

The inline handler can carry any name, such as `place` or `ship`. See [inline command handlers](concepts.md#inline-command-handlers).

## Approval on a paused thread (Expense namespace) { #expense-hitl }

- An LLM extracts expense data. A policy handler auto-approves small expenses or pauses with [`Interrupted`](control-flow.md#interrupted-resumed) for manager review.
- Resume with an `Approve` or `Reject` command.

<!-- autogen:start:expense_approval -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    subgraph Expense["Expense namespace"]
        direction LR
        Approve{{Approve}}:::cmd
        Approved(Approved):::devt
        Invalidated(Invalidated):::devt
        Reject{{Reject}}:::cmd
        Rejected(Rejected):::devt
        Submit{{Submit}}:::cmd
        Submitted(Submitted):::devt
        _reducer_status[(status)]:::reducer
    end
    ApprovalRequired([ApprovalRequired]):::syst
    _e0_[ ]:::entry ==> Reject
    _e1_[ ]:::entry ==> Submit
    Submit --> Submitted
    Submit --> Invalidated
    Approve --> Approved
    Reject --> Rejected
    Submitted -->|"check_policy [orchestrate]"| Approve
    Submitted -->|check_policy| ApprovalRequired
    Approved -.->|folds| _reducer_status
    Rejected -.->|folds| _reducer_status
    Invalidated -.->|folds| _reducer_status
    Submitted -.->|folds| _reducer_status
    linkStyle 6 stroke:#0369a1,stroke-width:3px
    linkStyle 8,9,10,11 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/expense_approval.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/expense_approval.graph.md)
<!-- autogen:end -->

## Tool-calling + AG-UI (Conversation namespace) { #conversation-agui }

ReAct tool-calling agent with an `AGUIAdapter` built for **AG-UI frontend tools** (CopilotKit `useFrontendTool`). `main()` runs the graph directly with `graph.ainvoke` and no frontend tool registered. It does not drive `adapter.stream`.

- Frontend-declared tools bound to the LLM via `build_langchain_tools`.
- Tool calls stream as `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`. Results return via `detect_new_tool_results` → `ToolsExecuted`.
- `DomainEvent + MessageEvent` mixin with [`message_reducer()`](reducers.md#message_reducer).
- For the handler-initiated `FrontendToolCallRequested` pattern, see [AG-UI](agui.md#handler-initiated-frontend-tools).

<!-- autogen:start:conversation -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    subgraph Conversation["Conversation namespace"]
        direction LR
        AnswerProduced(AnswerProduced):::devt
        Send{{Send}}:::cmd
        Sent(Sent):::devt
    end
    LLMResponded[/LLMResponded/]:::intg
    ToolsExecuted[/ToolsExecuted/]:::intg
    _reducer_messages[(messages)]:::reducer
    _e0_[ ]:::entry ==> Send
    _e1_[ ]:::entry ==> ToolsExecuted
    Send --> Sent
    Sent -->|call_llm| LLMResponded
    ToolsExecuted -->|call_llm| LLMResponded
    LLMResponded -->|finalize_answer| AnswerProduced
    Sent -.->|folds| _reducer_messages
    LLMResponded -.->|folds| _reducer_messages
    ToolsExecuted -.->|folds| _reducer_messages
    linkStyle 6,7,8 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/conversation.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/conversation.graph.md)
<!-- autogen:end -->

## Supervisor fan-in (Task namespace) { #supervisor }

- `Task.Run` starts the supervisor loop. The supervisor dispatches sub-commands `Task.Research` / `Task.Code` or emits the terminal `Task.Finalized` fact.
- A custom [`Reducer`](reducers.md#reducer) projects specialist outputs into shared context.

<!-- autogen:start:supervisor -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    subgraph Task["Task namespace"]
        direction LR
        Code{{Code}}:::cmd
        Completed(Completed):::devt
        Finalized(Finalized):::devt
        Produced(Produced):::devt
        Research{{Research}}:::cmd
        Run{{Run}}:::cmd
    end
    _reducer_context_parts[(context_parts)]:::reducer
    _e0_[ ]:::entry ==> Run
    Run -->|"supervisor [orchestrate]"| Research
    Run -->|"supervisor [orchestrate]"| Code
    Run -->|supervisor| Finalized
    Completed -->|"supervisor [orchestrate]"| Research
    Completed -->|"supervisor [orchestrate]"| Code
    Completed -->|supervisor| Finalized
    Produced -->|"supervisor [orchestrate]"| Research
    Produced -->|"supervisor [orchestrate]"| Code
    Produced -->|supervisor| Finalized
    Research --> Completed
    Code --> Produced
    Produced -.->|folds| _reducer_context_parts
    Completed -.->|folds| _reducer_context_parts
    Run -.->|folds| _reducer_context_parts
    linkStyle 1,2,4,5,7,8 stroke:#0369a1,stroke-width:3px
    linkStyle 12,13,14 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/supervisor.graph.md)
<!-- autogen:end -->

## Scatter fan-out (Batch namespace) { #scatter-fan-out }

- `Batch.Summarize` fans out to per-document work via [`Scatter[DocDispatched]`](control-flow.md#scatter).
- Gather handler uses `EventLog.filter()` to wait for all `DocSummarized` facts, then emits `Batch.Summarized`.

<!-- autogen:start:map_reduce -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    subgraph Batch["Batch namespace"]
        direction LR
        DocDispatched(DocDispatched):::devt
        DocSummarized(DocSummarized):::devt
        Summarize{{Summarize}}:::cmd
        Summarized(Summarized):::devt
    end
    _e0_[ ]:::entry ==> Summarize
    Summarize -.->|split_batch| DocDispatched
    DocDispatched -->|summarize_one| DocSummarized
    DocSummarized -->|gather_summaries| Summarized
%% Side-effect handlers: audit_trail (Event)
    linkStyle 1 stroke:#7c3aed,stroke-width:2.5px,stroke-dasharray:8 3
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/map_reduce.graph.md)
<!-- autogen:end -->

## Safety gates + streaming (Content namespace) { #content-pipeline }

- Inline `Content.Process.handle` classifies text (keyword-based, no LLM).
- External handlers gate approval, emitting `Content.Blocked` ([`Halted`](concepts.md#system-events) subtype) or `Content.Approved`.
- Live streaming via `astream_events()`.

<!-- autogen:start:content_pipeline -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    classDef reducer fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e
    subgraph Content["Content namespace"]
        direction LR
        Analyzed(Analyzed):::devt
        Approved(Approved):::devt
        Blocked([Blocked]):::halt
        Classified(Classified):::devt
        Process{{Process}}:::cmd
    end
    _reducer_stages[(stages)]:::reducer
    _e0_[ ]:::entry ==> Process
    Process --> Classified
    Classified -->|gate| Blocked
    Classified -->|gate| Approved
    Approved -->|analyze| Analyzed
    Analyzed -.->|folds| _reducer_stages
    Approved -.->|folds| _reducer_stages
    Classified -.->|folds| _reducer_stages
    linkStyle 5,6,7 stroke:#0ea5e9,stroke-width:1.5px,stroke-dasharray:2 2
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/content_pipeline.graph.md)
<!-- autogen:end -->

## Retries & escalation (Question namespace) { #error-recovery }

- `Ask` declares `raises=` and `retry=RetryPolicy(max_attempts=3)`. Retries emit `HandlerRetried`. The third failure emits `HandlerRaised`, which `give_up` turns into `Question.RetriesExhausted` (`Halted`). See [retries](control-flow.md#retries).

<!-- autogen:start:error_recovery -->
```mermaid
graph LR
    classDef entry fill:none,stroke:none,color:none
    classDef cmd fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    classDef devt fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef intg fill:#ede9fe,stroke:#6d28d9,color:#4c1d95
    classDef syst fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef halt fill:#fef3c7,stroke:#b45309,color:#78350f,stroke-width:3px,stroke-dasharray:4 2
    classDef inv fill:#ffedd5,stroke:#c2410c,color:#7c2d12
    subgraph Question["Question namespace"]
        direction LR
        Answered(Answered):::devt
        Ask{{Ask}}:::cmd
        RetriesExhausted([RetriesExhausted]):::halt
    end
    HandlerRaised([HandlerRaised]):::syst
    HandlerRetried([HandlerRetried]):::syst
    _e0_[ ]:::entry ==> Ask
    Ask -.->|"(raises)"| HandlerRaised
    Ask -.->|"(retry)"| HandlerRetried
    Ask --> Answered
    HandlerRaised -->|give_up| RetriesExhausted
    linkStyle 1 stroke:#6b7280,stroke-dasharray:3 3
    linkStyle 2 stroke:#0891b2,stroke-dasharray:2 4
```

[Full code](https://github.com/cadance-io/langgraph-events/blob/main/examples/error_recovery.py) · [Raw diagrams on GitHub](https://github.com/cadance-io/langgraph-events/blob/main/examples/error_recovery.graph.md)
<!-- autogen:end -->

