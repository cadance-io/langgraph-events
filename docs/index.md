# langgraph-events

Opinionated event-driven abstraction for LangGraph with a **DDD-aligned event taxonomy**. State IS events.

!!! warning "Experimental (v0.47.2)"
    This is an early-stage personal project, not a supported product. The API will change without notice or migration path. Do not depend on this for anything that you cannot rewrite with little effort.

Model your domain as namespaces with commands and outcomes, colocate the handler on the command, and let `EventGraph` derive the topology. The quick start is in the [README](https://github.com/cadance-io/langgraph-events#quick-start). The install lines are in [Getting Started](getting-started.md#install).

## What the graph looks like

`graph.namespaces().mermaid()` on the [`order`](patterns.md#order) example:

<!-- autogen:start:hero -->
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
<!-- autogen:end -->

## Navigate

- [Getting Started](getting-started.md)
- [Concepts](concepts.md)
- [Control Flow](control-flow.md) — fan-out, paused threads, exceptions, retries, invariants
- [Reducers](reducers.md) — namespace-scoped or graph-wide
- [Patterns](patterns.md)
- [Streaming](streaming.md)
- [AG-UI Adapter](agui.md)
- [Reflection](reflection.md) — `graph.reflect(log)` and the `query_log` tool: an LLM queries the run's facts lazily instead of holding the log in context
- [Event Migrations](event-migrations.md) — keep old checkpoints working after a rename
- [Migration Gate and Deploys](migration-gate.md) — the release gate, the deploy steps and the store rewrite
- [Retiring an Interrupted Subclass](retiring-interrupted.md) — delete a pause class and keep each paused thread resumable
- [API Reference](api.md)
