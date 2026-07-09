"""The EthicLens agent layer: schema inference, audit planning, execution, and narrative.

Five stages (see build plan), each independently testable:

1. ``schema_inference`` — LLM proposes column roles; a human must confirm.
2. ``audit_planner`` — deterministic rules engine; decides which metrics apply. No LLM.
3. ``executor`` — wires a confirmed plan into ``fairness_core.run_audit``. No LLM.
4. ``narrative`` — LLM explains the scorecard; a deterministic validator grounds every number.
5. Q&A chat (future work) will reuse the same grounding validator against stored audit records.
"""
