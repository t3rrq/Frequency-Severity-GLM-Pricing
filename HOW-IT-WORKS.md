# How this works (plain language)

This project asks a simple question about **car liability insurance** in France:

> If we look at who the driver is, what they drive, and where they live, can we tell **who is likely to cost the insurer more** — better than charging everyone the same?

The technical write-up is in [ANALYSIS.md](ANALYSIS.md). This page is for anyone who does not work in insurance math.

## The everyday idea

Imagine you run a garage that also sells a promise: “If this car hurts someone, we pay.” You cannot charge every customer the same if some groups crash much more often. You also cannot wait until after the crash to set the price. You need a **guess of the bill before it happens**.

That guess is not the sticker price on a website. It is only the **expected claim cost** — no office expenses, no profit, no discounts. Actuaries call that a *pure premium*. Think of it as “what we think this policy will cost in claims.”

## Two questions, then multiply

A claim has two parts, like a restaurant bill:

1. **How often** does someone order? (How often does a policy have a claim?)
2. **How large** is the check when they do? (How many euros is a typical claim?)

Expected cost ≈ (expected number of claims) × (expected size of a claim).

We estimate those two pieces separately, then multiply. That is the whole model.

We use information already on the policy: area, region, how powerful and how old the car is, driver age, a **bonus-malus** score (a French no-claims / at-fault record — higher usually means a worse history), brand, fuel type, and how densely populated the area is.

## How we know it is not just a story we told ourselves

We hide **20%** of the policies, build the guess on the other **80%**, then score the hidden group. That is like studying on last year’s customers and grading yourself on next year’s.

We compare to a **flat rate**: take all the claim euros in that hidden group and spread them by how long each policy was in force. A car insured for a full year gets more of the pot than a car insured for a month. That is the “everyone the same per year” baseline.

Then we line people up from “we think you will cost the most” to “we think you will cost the least” and look at **where the real claim euros actually went**.

## What we found

On that hidden 20% (about 136,000 policies):

- Real claims added up to about **€11.1 million**.
- The model’s expected costs added up to about **€12.3 million** (a bit high — we are better at **ranking** people than at hitting the exact euro total).
- The **10% of policies the model called most expensive** had about **1.5 times** as many claim euros per year of cover as the average. That is the main result: the ranking is not random.
- The ranking beats the flat rate. In everyday terms: **using the rating factors finds the costly policies better than ignoring them.**

Most of that signal is **how often** claims happen, not how large they are. People with a worse bonus-malus record have far more claims (about **0.08** per year of cover at the best band vs about **0.35** at the worst). Young drivers also have more claims. Claim *size* barely moves with those same factors.

## What this is not

- It is not “the price you would pay at a broker.” Real prices add expenses, tax, profit, and competition.
- It is not a guarantee. Insurance is luck plus pattern. A cheap-looking policy can still have one huge claim.
- The French files we use do not perfectly match “number of claims” on the policy to “list of claim amounts.” We did not paper over that.
- When we rank by **cost for this short policy term**, short-duration policies bunch at the bottom. A few large claims there make that group look wild. That is a ranking quirk, not “the cheapest 10% are six times worse.”

## One picture to remember

```
Who crashes more often?  ×  How big is a crash?  →  Expected cost
         (the strong part)        (the weak part)

Then: did the expensive-looking people actually cost more on data we hid?
Answer: yes, enough to beat “same price per year for everyone.”
```

If you want the formulas, Gini numbers, and charts, read [ANALYSIS.md](ANALYSIS.md).
