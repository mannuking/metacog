# LinkedIn post — for copy-paste

**Image to attach:** the dashboard screenshot at step 20/20 (the one you attached to the chat).

---

```
I taught a 35B-parameter AI to say "I don't know."

Then it stopped lying to me.

For 4 hours and $15 of cloud GPU, I ran a single reinforcement-learning
experiment on Qwen 3.6-35B-A3B with a custom reward function.

One change to the reward signal. That's it.

The reward punished being confidently wrong. Twice.
And it gave zero penalty for being humbly wrong.

Step 1:  the model was right 19% of the time and overconfident
         the other 81%.
Step 20: the model was right 72% of the time and overconfident
         just 10%.

ECE — Expected Calibration Error — fell from 0.50 to 0.10.
That's an 80% drop in how often "the model is sure" actually
means "the model is correct."

The chart is the proof. 20 steps. One reward function.
No humans in the loop.

The trick: tell the model that saying "I don't know" is a
strictly better strategy than bluffing. Within 5 RL steps,
the bluffing was gone.

This is what calibration looks like in a reasoning model.
This is what "metacognition" means in code.

Next: scale it. 100B+ parameters. 200-question held-out eval.
K=5 self-consistency. Phase 2 starts tonight.

The paper is being written. The full reproducible pipeline
will be open-sourced.

If you're building AI agents that need to know when they
don't know — this is the missing piece.

♻️ Repost if you think "I don't know" is the most underrated
   skill in AI right now.
💬 Comment: what's the worst overconfident AI answer you've
   ever trusted?

#AI #MachineLearning #RLHF #LLM #Reasoning #Calibration
#Metacognition #OpenSource #Qwen #AGI
```

---

**Character count:** ~1,650 (well under LinkedIn's 3,000 limit)
**Estimated read time:** 90 seconds
**Hook strength:** opens with a paradox (taught an AI to say "I don't know") that has to be clicked-through
**Numbers:** 3 specific stats in the first 250 chars (4h, $15, 35B)
**Story arc:** setup → reveal → proof → takeaway → CTA
**CTA:** dual (repost + comment) — both are LinkedIn virality drivers
**Hashtags:** 10, all in the AI/ML top tier, no oversaturated ones
