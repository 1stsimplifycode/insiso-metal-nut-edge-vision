# Three-minute spoken explanation (plain English)

A factory makes small metal nuts. Some come out bent, scratched, stained, or upside down. The customer
wants a camera that looks at every nut, throws out the bad ones, shows where the problem is, and does all
of this on a small computer next to the machine, not on the internet.

I used a public research collection of photos of metal nuts. It has 242 good nuts and 93 bad ones. Each bad
photo comes with a mask: a black-and-white picture that marks exactly which pixels are damaged. The program I
trained draws rectangles, so I wrote code that turns each marked area into the tightest rectangle around it,
and tests to check that every damaged pixel ends up inside a rectangle.

Before training anything, I divided the photos into three groups: one to learn from, one to make choices
with, and one locked away for the final test. No photo appears in two groups, and I saved a fingerprint of
the split so it cannot quietly change. The final test group has 29 bad nuts and 40 good nuts.

I trained a small, fast object-detection model, about two and a half million numbers, on my laptop's
graphics card. On the locked test photos, it found most damage and never accused a good nut. But when I looked
at its mistakes one by one, the interesting part appeared. Six bad nuts would have been passed. In almost
every case the model *had* drawn a rectangle in the right place. It just was not confident enough to pass the
cut-off I had picked with the usual textbook rule. The rule balances two kinds of mistakes equally, but in a
factory, letting a bad part through is usually far worse than double-checking a good one.

So my main improvement changed no learning at all. I picked the cut-off differently: the lowest
confidence at which the model still raises no false alarms on the *choice* photos. On the locked test, all 29
bad nuts were caught and no good nut was rejected. I am careful about what that proves: with only 29 bad nuts,
the true catch rate could still be as low as about 88 percent.

I also tried a second idea: some damage marks are split into several small pieces, so I merged nearby pieces
before training. The rectangles became tidier, but the program then missed one stained nut completely. That
idea did not help, and I report it as a result, not something to hide.

For the factory computer, I converted the model to a standard portable format, ONNX, and checked that it
gives the same answers as the original, to six decimal places. On my laptop's processor the model takes about
40 to 60 thousandths of a second per photo, and the whole inspection, including reading the photo, about 56
to 78. A compressed eight-bit version was about 1.8 times faster with the same results.

Finally, I tried a method that learns only from good nuts and flags anything unusual. It never saw a single
bad nut, yet it caught all 29, though it also rejected 4 good ones. That is a useful backup for kinds of
damage nobody has photographed yet.

The honest summary: a working, tested prototype with measured results, a clear picture of where it fails,
and a list of what must be checked on a real production line before anyone should trust it.
