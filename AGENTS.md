# Agent Working Guidelines

## Small fixes

- For small, localized bug fixes or configuration changes, keep the workflow lightweight: make the minimal change and run only focused validation.
- Do not automatically run a full test suite, invoke a multi-agent code review, or commit changes unless the user explicitly asks for those steps.

## Training operations

- Report and record training timestamps in Beijing time (`Asia/Shanghai`, UTC+8) unless the user explicitly requests another timezone.
