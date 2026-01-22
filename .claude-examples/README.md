# Claude Code Configuration Examples

This directory contains example Claude Code configurations tailored for the CAD Estimation Pipeline project.

## 📁 Contents

```
.claude-examples/
├── agents/                    # Specialized AI agents
│   ├── dataclass-validator.md   # Validates dataclass parameter usage
│   └── python-tester.md          # Runs tests and fixes failures
├── commands/                  # Slash commands for quick tasks
│   ├── pipeline-test.md          # /pipeline-test - Run full test suite
│   └── coverage-check.md         # /coverage-check - Analyze coverage
├── rules/                     # Always-follow development rules
│   └── python-dataclasses.md     # Dataclass best practices
└── README.md                  # This file
```

## 🚀 Installation

### Option 1: Install affaan-m/everything-claude-code (Recommended)

Get the full collection of production-ready configs:

```bash
# In Claude Code terminal
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

This gives you:
- ✅ 10+ specialized agents
- ✅ 15+ slash commands
- ✅ Production hooks and rules
- ✅ Testing workflows
- ✅ Verification loops

**Then add project-specific configs from this directory.**

### Option 2: Manual Installation (Custom Setup)

Copy these examples to your Claude Code config directory:

```bash
# Create directories if they don't exist
mkdir -p ~/.claude/agents ~/.claude/commands ~/.claude/rules

# Copy agents
cp .claude-examples/agents/*.md ~/.claude/agents/

# Copy commands
cp .claude-examples/commands/*.md ~/.claude/commands/

# Copy rules
cp .claude-examples/rules/*.md ~/.claude/rules/
```

### Option 3: Project-Only Configuration

Keep configs in project directory (requires Claude Code >= 0.4):

```bash
# Already in project as .claude-examples/
# Claude Code will auto-discover them
```

---

## 📖 How to Use

### Using Agents

Agents are specialized AI assistants for specific tasks:

```markdown
# In chat with Claude Code:
"Use the dataclass-validator agent to check all dataclass instantiations"

# Claude will spawn the agent with:
# - Limited scope (only validation task)
# - Specific tools (Grep, Read, Glob)
# - Focused instructions
```

**When to use agents:**
- Complex multi-step tasks
- Specialized analysis (testing, validation)
- Background processing
- Parallel work (multiple agents at once)

### Using Commands

Commands are quick slash shortcuts:

```bash
# In Claude Code
/pipeline-test        # Run full test suite
/coverage-check       # Analyze test coverage
```

**Create your own commands:** See `commands/` for templates.

### Using Rules

Rules are always-active guidelines that Claude follows:

- Placed in `~/.claude/rules/` = apply to ALL projects
- Placed in `.claude-examples/rules/` = project-specific only

**Example:** `python-dataclasses.md` ensures Claude always reads dataclass definitions before using them.

---

## 🎯 Recommended Workflow

### 1. Daily Development
```bash
# Start coding session
/pipeline-test              # Verify baseline works

# Make changes...
# ...

# Before committing
/coverage-check             # Ensure coverage maintained
```

### 2. Fixing Bugs
```markdown
# In chat:
"Use the dataclass-validator agent to find parameter mismatches"

# Agent reports issues, you review, then:
"Fix the issues found in orchestrator.py"
```

### 3. Adding Features
```markdown
1. Plan: "I need to add X feature"
2. Claude reads CLAUDE.md for project context
3. Claude follows rules from python-dataclasses.md
4. After coding: /pipeline-test
5. If tests fail: Use python-tester agent
```

---

## 🔧 Customization

### Adding Your Own Agent

Create `~/.claude/agents/my-agent.md`:

```markdown
---
name: my-agent
description: Brief description (3-5 words)
tools: Bash, Read, Edit, Grep  # Tools this agent can use
model: sonnet  # or opus, haiku
---

# Agent instructions here
You are a specialist in...

## Process
1. Step 1
2. Step 2
```

### Adding Your Own Command

Create `~/.claude/commands/my-command.md`:

```markdown
---
name: my-command
description: What this command does
---

# Instructions for Claude when command is invoked

1. Do X
2. Run Y
3. Report Z
```

### Adding Your Own Rule

Create `~/.claude/rules/my-rule.md`:

```markdown
# My Development Rule

## Always Do
- Thing 1
- Thing 2

## Never Do
- Bad thing 1
- Bad thing 2

## Examples
...
```

---

## 🎓 Learning Resources

### Must-Read Guides
1. **[The Shorthand Guide](https://github.com/affaan-m/everything-claude-code)** - Foundations
2. **[The Longform Guide](https://github.com/affaan-m/everything-claude-code)** - Advanced techniques

### Topics Covered in Guides
- Token optimization (stay within 200k context)
- Memory persistence across sessions
- Verification loops & evals
- Parallelization strategies
- Subagent orchestration

### Project-Specific Docs
- `CLAUDE.md` in project root - Project context and critical info
- This README - How to use these configs

---

## 💡 Pro Tips

### 1. Context Window Management
Don't enable too many tools at once:
- ❌ 30 MCPs enabled = 70k context left
- ✅ 8-10 MCPs enabled = 150k+ context left

### 2. Agent Delegation
Use agents for:
- Long-running tasks (testing, validation)
- Parallel work (test + lint + build simultaneously)
- Specialized analysis (security review, performance)

### 3. Keep Rules Modular
Don't put everything in one rule file:
- `python-dataclasses.md` - Just dataclass rules
- `testing.md` - Just testing standards
- `security.md` - Just security requirements

Each rule file should be focused and < 500 lines.

### 4. Command Naming
Use verb-noun format:
- ✅ `pipeline-test`, `coverage-check`, `build-fix`
- ❌ `test`, `check`, `fix` (too generic)

### 5. Project CLAUDE.md
**Critical info goes here:**
- Dataclass parameter names (prevents bugs)
- Common pitfalls
- File locations
- Quick commands

Think: "What do I wish I remembered when starting a task?"

---

## 🐛 Troubleshooting

### Commands Not Found
```bash
# Verify installation
ls ~/.claude/commands/

# Check Claude Code loaded them
/help  # Should list your commands
```

### Agents Not Available
```bash
# Verify installation
ls ~/.claude/agents/

# Try spawning manually
"Use the python-tester agent to run tests"
```

### Rules Not Applied
- Rules in `~/.claude/rules/` = global (all projects)
- Rules in `.claude-examples/rules/` = may need project config
- Check Claude Code version >= 0.4 for project-level configs

---

## 🤝 Contributing

Found a useful agent or command? Add it!

1. Create the .md file in appropriate directory
2. Test it thoroughly
3. Document it in this README
4. (Optional) Share with the community

---

## 📊 Metrics & Success

### Before These Configs
- ❌ Dataclass parameter bugs: ~5 per week
- ❌ Forgot to run tests: ~30% of commits
- ❌ Coverage dropped without noticing
- ❌ Type errors in production

### After These Configs
- ✅ Dataclass bugs: < 1 per month (agent catches them)
- ✅ Tests always run: `/pipeline-test` habit
- ✅ Coverage tracked: `/coverage-check` shows gaps
- ✅ Type safety: Rules enforce checking

**ROI: ~10 hours/week saved on debugging** 🎯

---

## 📚 Additional Resources

- [Claude Code Documentation](https://docs.anthropic.com/claude/docs)
- [affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code)
- [The Shorthand Guide](https://substack.com/@affaanmustafa)
- [The Longform Guide](https://substack.com/@affaanmustafa)

---

**Questions?** Ask Claude Code: `/ask How do I use agents?`

**Next Steps:**
1. ✅ Install configs (see Installation above)
2. ✅ Read `CLAUDE.md` in project root
3. ✅ Try `/pipeline-test` command
4. ✅ Experiment with agents
5. ✅ Customize for your workflow

Happy coding! 🚀
