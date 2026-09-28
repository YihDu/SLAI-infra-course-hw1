# Agent traces

Submission requires at least one nonempty exported agent session trace in this folder.

Devin CLI exports sessions in ATIF format:

```sh
devin list                          # find session ids for this directory
devin -r <session-id> --export agent-trace/<session-id>.json
```

Resuming with `--export` writes the full conversation after each turn. Export every
session used for this project before packaging the submission ZIP.
