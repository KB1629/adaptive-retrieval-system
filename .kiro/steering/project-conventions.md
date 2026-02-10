# Project Conventions - General Steering Guide

## Overview

This document establishes conventions for all development work in this workspace. Follow these guidelines to maintain consistency, quality, and scalability.

---

## Code Organization

### Directory Structure
- Keep folders organized and scalable
- Group related functionality together
- Avoid deeply nested structures (max 3-4 levels)
- Use clear, descriptive names

### File Naming
- Python: `snake_case.py`
- TypeScript/JavaScript: `camelCase.ts` or `kebab-case.ts`
- Configuration: `kebab-case.yaml` or `kebab-case.json`
- Documentation: `UPPER_CASE.md` for root docs, `kebab-case.md` for others

### Module Design
- Each module should have a single responsibility
- Keep files under 300 lines when possible
- Extract shared utilities to `utils/` or `common/`
- Use interfaces/protocols for abstraction

---

## Code Quality

### Best Practices
- Write self-documenting code with clear names
- Add type hints (Python) or TypeScript types
- Include docstrings for public APIs
- Handle errors gracefully with meaningful messages
- Avoid magic numbers - use named constants

### Testing
- Write tests alongside implementation
- Use property-based testing for invariants
- Unit tests for specific examples and edge cases
- Integration tests for component interactions
- Run tests before committing

### Performance
- Profile before optimizing
- Document performance expectations
- Consider memory constraints (especially for M1 Pro)
- Use batch processing where appropriate

---

## Documentation

### README Updates
Update the README when:
- A medium-level milestone is completed
- The project plan changes significantly
- New features are added
- Installation/setup steps change

### Changelog Maintenance
Maintain CHANGELOG.md for:
- Architecture decisions and rationale
- Breaking changes
- Assumption changes
- Direction pivots

### Inline Documentation
- Document "why" not "what"
- Explain non-obvious decisions
- Note assumptions and constraints
- Reference external resources (papers, docs)

---

## Version Control

### Commit Messages
- Use clear, descriptive messages
- Reference task numbers when applicable
- Keep commits focused and atomic

### Branch Strategy
- Main branch should always be stable
- Feature branches for new work
- Clean up after merging

---

## Configuration Management

### Environment
- Use configuration files (YAML/JSON) for settings
- Support environment variable overrides
- Never hardcode secrets or paths
- Provide sensible defaults

### Hardware Awareness
- Auto-detect available hardware (MPS/CUDA/CPU)
- Graceful fallback when resources unavailable
- Document hardware requirements

---

## Error Handling

### Principles
- Fail fast with clear error messages
- Log errors with context for debugging
- Provide fallback behavior where appropriate
- Don't swallow exceptions silently

### Logging
- Use structured logging
- Include timestamps and context
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Don't log sensitive information

---

## Long-Running Project Mindset

This workspace contains evolving research projects. Approach work with:

1. **Incremental Progress**: Build in small, testable increments
2. **Documentation First**: Document decisions as they're made
3. **Flexibility**: Requirements may change based on findings
4. **Reproducibility**: Ensure experiments can be repeated
5. **Traceability**: Link code to requirements and design

---

## Quality Checklist

Before completing any significant work:
- [ ] Code follows project conventions
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Error handling implemented
- [ ] No hardcoded values
- [ ] Performance considered
- [ ] Changes logged if significant
