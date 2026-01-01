# Annotation Training Guideline
## Fluency and Imperceptibility Assessment

Thank you for participating in this annotation task. This guideline explains how to rate documents consistently and objectively. Please read everything carefully before starting.

---

## Important Notice (Please Read First)

- This task must be completed using your own human judgment.
- **Do not use ChatGPT, AI language models, grammar checkers, writing assistants, or any automated tools** to evaluate or rewrite the passages.
- Using automated tools would invalidate the study results.
- All assessments must be based solely on your personal, human evaluation of fluency and naturalness.

---

## 1. Objective of the Task

You will be shown short passages. For each passage, you must provide:

### Fluency Rating (1 to 5)
A score from 1 to 5 measuring grammatical quality, readability, and coherence.

### Imperceptibility Label (0 or 1)
A binary label indicating whether the text appears normal or suspicious/manipulated.

- **1 = Normal** (no sign of manipulation)
- **0 = Attacked** (appears altered, injected, or unnatural in a way that suggests manipulation)

---

## 2. Annotation Process (Step by Step)

For each document:

1. **Read the entire passage once** without judging it.
2. **Assign Fluency Rating (1 to 5).**  
   Use the fluency scale definitions and examples below.
3. **Assign Imperceptibility Label (0 or 1).**  
   Use the labeling criteria and examples below.
4. Move to the next item.  
   Do not revisit previous items unless instructed.

**Important:**
- Always assign **fluency first**, **imperceptibility second**, to avoid influencing the fluency rating.
- If you feel tired or unfocused, take a break. Annotation quality is more important than speed.

---

## 3. Fluency Rating Scale (1 to 5)

Rate how natural, grammatical, and coherent the text is. Below are definitions and synthetic, unbiased examples for training.

### Score 5: Excellent Fluency
Highly coherent, natural grammar, smooth and clear.

**Example:**  
To renew your permit, submit the updated application form along with the verification documents listed on the requirements page. Once everything is approved, your new card will be mailed to the address on file. Most renewals take between five and ten business days.

### Score 4: Good Fluency
Mostly clear and natural, with minor awkwardness or small grammar issues.

**Example:**  
When logging in to your account, make sure you are using the most recent password you created. If you still cannot access your profile, try resetting your password or checking if your browser allows cookies.

### Score 3: Moderate Fluency
Understandable overall but contains noticeable grammar issues, awkward phrasing, or repetitive structure.

**Example:**  
To check if your item shipped, go to the order page and look for the tracking number. Sometimes the tracking does not update for a while, which can make people confused about where their package is.

### Score 2: Low Fluency
Frequent grammar problems, unclear sentence construction, difficult flow, but meaning is still mostly recoverable.

**Example:**  
The login reset page not working right now because server having issue and users is getting message that no match found even when typing correct info. Email sometimes send but not always, so people get confused what to do next.

### Score 1: Poor Fluency
Severely broken grammar, confusing structure, partially incoherent.

**Example:**  
Account reset email not go send and user many time trying but system no telling what wrong or where click next for making fix happening. Login page working maybe later but not sure doing correct.

---

## 4. Imperceptibility (1 = Normal, 0 = Attacked)

This evaluates whether the passage looks naturally written or appears manipulated.

A manipulated passage may include:
- Unnatural insertions
- Off-topic statements
- Keyword stuffing
- Nonsense content
- Abrupt changes that break coherence

These are only examples. Real cases may vary.

### Label 1 = Normal (Plausible Human-Written Text)
The document appears natural and free from any noticeable inconsistencies or signs of manipulation.

**Example A:**  
The city administration introduced new water conservation guidelines and encouraged residents to attend upcoming workshops.

**Example B:**  
The town library was renovate last year and now more easy for people to access the new children section.

### Label 0 = Attacked / Manipulated
The document contains noticeable inconsistencies or signs of manipulation that could raise suspicion or affect the trustworthiness of the content.

**Example A: Spam-like insertion**  
The committee submitted several recommendations for the next planning cycle. Best cheap deals at discount-offer-now dot com today.

**Example B: Keyword or term spamming**  
Volunteers restored the hiking trail. Trail hiking trail best trail hiking now trail.

**Example C: Nonsensical keyword insertion**  
Cloud-velocity banana index protocol system rapid blue update. The service form must be completed before submission.

**Example D: Term spamming**  
To reset your password, open the account menu and select “Reset Password.” Password reset password help password fix password account reset right now password.

**Example E: Awkward topic shift**  
"Foreclosure can damage your credit score significantly. How do I log in? Shipping Options. Returns Policy."

**Note:** The presence of characters **“â”** is a Unicode encoding issue and should be ignored. Do not label a passage as Attacked based on this alone unless there are additional clear signs of manipulation.

---

## Important Final Note

The examples above illustrate common symptoms of manipulated passages. However, real manipulated passages may look different. Use your judgment to determine whether the passage feels suspicious, artificial, or unnatural, even if it does not match any example exactly.

Your role is to judge overall naturalness, not pattern-match. Also, do not assume a passage is manipulated unless you notice clear signs that break naturalness.

---

## Avoid Bias Based on Content Familiarity

Please do not evaluate passages differently based on how much you know about the topic. A document about firearms, real estate licensing, biology, history, or online shopping must be judged the same way.

Your role is to assess the quality, coherence, and naturalness of the writing, not your comfort with or knowledge of the subject.

We intentionally remind annotators that the goal is to evaluate general readability and trustworthiness, not to reward or penalize specific topics.

---

## Thank You

Your careful work is essential for evaluating how well manipulated documents appear to humans. If anything in this guideline is unclear, please reach out before beginning the task.

---

## Fluency Cheat Sheet

| Score | Label | Quick definition |
|---:|---|---|
| 5 | Excellent | Very clear and easy to read; grammar/spelling/punctuation are correct or have only tiny issues; sentences flow naturally and ideas are logically connected. |
| 4 | Good | Generally clear and natural; a few minor grammar mistakes, slightly awkward phrases, or mild repetition may appear, but readability stays high and the main ideas are easy to follow. |
| 3 | Moderate | Understandable overall but noticeably uneven; may have clumsy sentence structure, repetitive patterns, or abrupt transitions; grammar is mostly OK, but the writing does not feel smooth or polished. |
| 2 | Low | Frequent issues that reduce readability; many awkward or incomplete sentences, disorganized structure, confusing phrasing; meaning is still recoverable, but it takes effort. |
| 1 | Poor | Very difficult to understand; severe grammar/spelling errors, broken words, or corrupted text; sentences may be incomplete or scrambled; overall message is unclear or only partially recoverable. |
