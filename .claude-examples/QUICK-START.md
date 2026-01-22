# ⚡ Quick Start - Claude Code for CAD Project

Get productive with Claude Code agents in **5 minutes**.

---

## 🎯 Step 1: Install Base Configuration (2 min)

### Option A: Full Featured (Recommended)
```bash
# In Claude Code
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

### Option B: Manual Minimal Setup
```bash
# Copy project-specific configs
mkdir -p ~/.claude/agents ~/.claude/commands ~/.claude/rules

cp .claude-examples/agents/dataclass-validator.md ~/.claude/agents/
cp .claude-examples/commands/pipeline-test.md ~/.claude/commands/
cp .claude-examples/rules/python-dataclasses.md ~/.claude/rules/
```

---

## 🎯 Step 2: Add Project Context (1 min)

The `CLAUDE.md` file is already in your project root! 🎉

**What it does:**
- Tells Claude about your dataclasses (EstimatePhases, Risk, Component)
- Prevents parameter name bugs
- Lists critical files and patterns

**No action needed** - Claude reads it automatically.

---

## 🎯 Step 3: Test Your Setup (1 min)

### Try a command:
```bash
/pipeline-test
```

**Expected:** Claude runs pytest and reports results.

### Try using an agent:
```markdown
"Use the dataclass-validator agent to check all EstimatePhases usage"
```

**Expected:** Agent searches code and reports any parameter mismatches.

---

## 🎯 Step 4: Add Hooks (Optional - 1 min)

Hooks give you **automatic warnings** before making mistakes.

### Copy hooks to settings:
```bash
# View the example
cat .claude-examples/hooks-example.json

# Add to ~/.claude/settings.json (merge with existing config)
```

**What you get:**
- ⚠️ Warning before editing files with EstimatePhases
- 💡 Reminder to use logging instead of print()
- ✅ Coverage suggestions after tests
- 🏁 Checklist when ending session

---

## ✅ You're Ready!

### Now you can:

#### 1️⃣ **Prevent Dataclass Bugs** (saved you 2 hours today!)
```markdown
Before: EstimatePhases(hours_3d_layout=x) ❌ → Error!
After: Claude reads definition first → EstimatePhases(layout=x) ✅
```

#### 2️⃣ **Run Tests Faster**
```bash
Old way: pytest tests/validate_multi_model.py -v --cov=src --cov-report=term
New way: /pipeline-test
```

#### 3️⃣ **Validate Code Quality**
```markdown
"Use python-tester agent to run full test suite and fix any failures"
# Agent runs tests, analyzes failures, fixes code, re-runs tests
```

#### 4️⃣ **Check Coverage**
```bash
/coverage-check
# Shows which files need more tests with specific suggestions
```

---

## 🎓 Next Steps

### Learn by Doing
1. **Make a small change** to pipeline code
2. **Run `/pipeline-test`** to verify it works
3. **Ask Claude:** "Check if I used dataclasses correctly"
4. **Commit** with confidence

### Customize
1. **Add your own commands** for frequent tasks
2. **Create agents** for project-specific workflows
3. **Update CLAUDE.md** as project evolves

### Go Deeper
- Read `.claude-examples/README.md` for full docs
- Install [everything-claude-code](https://github.com/affaan-m/everything-claude-code) for advanced features
- Read [The Shorthand Guide](https://github.com/affaan-m/everything-claude-code) for best practices

---

## 🐛 Common Issues

### "Command not found"
```bash
# Check installation
ls ~/.claude/commands/
# Should see: pipeline-test.md, coverage-check.md

# Restart Claude Code if needed
```

### "Agent doesn't work"
```markdown
# Try explicit invocation
"Use the dataclass-validator agent to check src/cad/infrastructure/multi_model/orchestrator.py"

# Make sure agent file is in ~/.claude/agents/
```

### "CLAUDE.md not working"
```bash
# Verify it exists in project root
ls CLAUDE.md

# Should be there - Claude reads it automatically
```

---

## 📊 What You Get

### Without Agents
- ❌ 5+ dataclass bugs per week
- ❌ Forgot to run tests → broken commits
- ❌ No coverage tracking
- ❌ Manual type checking

### With Agents
- ✅ < 1 dataclass bug per month
- ✅ `/pipeline-test` habit → clean commits
- ✅ `/coverage-check` shows gaps
- ✅ Hooks remind about quality

**Time Saved: ~10 hours/week** ⏰

---

## 💡 Pro Tips

### 1. Use CLAUDE.md Like a Cheat Sheet
```markdown
"What are the EstimatePhases parameters?"
# Claude reads CLAUDE.md and answers instantly
```

### 2. Chain Commands
```bash
/pipeline-test
# After it finishes:
/coverage-check
```

### 3. Delegate to Agents
```markdown
"Use python-tester agent to run tests in background while I work on docs"
# Parallel work!
```

### 4. Update CLAUDE.md When You Find Bugs
```markdown
Found a bug? Add it to "Common Pitfalls" section in CLAUDE.md
# Future you (and Claude) will thank you
```

---

## 🎯 Your First Task

Try this right now:

```markdown
1. Ask Claude: "Use dataclass-validator agent to audit the entire codebase"
2. Review the report
3. Run: /pipeline-test
4. Run: /coverage-check
5. Celebrate! 🎉
```

---

## 🆘 Need Help?

- **Ask Claude:** `/ask How do I create a custom agent?`
- **Read docs:** `.claude-examples/README.md`
- **Check examples:** Files in `.claude-examples/`
- **Community:** [everything-claude-code repo](https://github.com/affaan-m/everything-claude-code)

---

**You're all set! Go build amazing things.** 🚀

---

*Quick Start Guide for CAD Estimation Pipeline*
*Last Updated: 2026-01-22*
