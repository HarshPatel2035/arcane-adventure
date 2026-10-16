# Arcane Adventure play-through checklist

Close the older game and run:

```bash
cd "$HOME/Desktop/Hand Gesture Game "
./.venv/bin/python arcane_adventure.py
```

The title should be **Arcane Adventure - The Enchanted Grove**. Select Adventure and a
companion at camp; press Space. Start with motion spells OFF.

1. **Feedback and movement:** move left/right; lower your hand to stop. Look at the
   recognized spell, confidence, cooldown and mana messages. Compare with A/D and keys
   1–5 if a gesture is unreliable. Losing camera tracking should not replay old spells.
2. **Upgrades:** complete the first wave. Combat must freeze while three cards appear.
   Move your hand to highlight one, hold steady, then make a fist after releasing the
   entry pose. The selection should happen once. Mouse, keys 1–3 and Space also work.
   Watch for the selected effect (two fireballs, three counter shots, longer chain, etc.).
3. **Enemies:** identify gold-marked archers, clustered swarms, shield-shaped golems and
   plus-marked healers. Fireballs should bounce off armored golems; pointing exposes
   them for 4 seconds. Try lightning and splash attacks against groups.
4. **Heart Tree:** protect it for 35 seconds. Enemies reaching it reduce its health.
   Push them back with telekinesis. The encounter should fail if the tree reaches zero.
5. **Portals:** move beneath each of the three purple portals and make a fist. A shield
   held there for 0.9 seconds should fill its bar and seal it. Seal all three to advance.
6. **Props:** line up the pale cursor near a rock/seed/crystal and point. Once lifted,
   the wizard should stay still while your hand drags the prop. Palm throws it; lowering
   your hand places it. Move a rock into a shot's path, and throw a seed at a group.
   Point -> palm should turn the throw into a stronger Meteor attack.
7. **Guardian:** break both purple crystals; ordinary attacks should not hurt the
   protected boss first. Then counter a pink shot into its mirror shield and attack in
   the 4-second opening. Below half health, enter the green lane before the storm hits.
   Defeat it to reach victory. R restarts; Return to camp lets you change your build.
8. **Companions/unlocks:** compare Owl mana collection, Dragon hit blocking and Spirit
   target marks/4-second combos. After victory, equip Gold at camp. After three guardian
   defeats, check Rose. Quit/relaunch and verify unlocks remain.
9. **Optional motions:** pause with P, enable motion with O, then resume. Sweep an open
   palm for Wind Slash; make a circle with a peace sign for Rune Ward; hold a steady
   fist for 1.3 seconds for Charged Ward. Check mana use and cooldowns. Z/X/V provide
   keyboard comparisons. Small movements/two hands should not accidentally trigger them.
10. **Pause and Endless:** pause during a projectile warning, drag or storm. Timers and
    objects should freeze, then resume cleanly. Try Endless from camp and verify another
    guardian appears after defeat and upgrade selection.

11. **Calibration and difficulty:** at camp press C and sweep an open palm across your
    comfortable range for 4 seconds. Start a run: reaching your comfortable edge
    should move the wizard to the arena wall. Press F to try Easy and Hard; Hard
    hits should hurt more and enemies should arrive faster. Relaunch and check
    both settings are remembered.
12. **Top scores:** finish two runs. The end screen should list the top runs and show
    "New high score!" only when you beat the best saved score.

Quick guardian check:

```bash
./.venv/bin/python arcane_adventure.py --boss-practice
```

Keyboard-only check:

```bash
./.venv/bin/python arcane_adventure.py --no-camera
```

In keyboard mode, 4 lifts a nearby prop, arrows drag it, and 1 throws. Normal spells are
1 Fireball, 2 Shield, 3 Lightning, 4 Telekinesis, 5 Ultimate. P opens the help/pause screen.

Automated checks cover the mechanics and progression. Your play-through should focus on
recognition reliability, menu confirmation, motion sensitivity, readability, sound and
whether the tree/guardian difficulty feels fair. Note the encounter and action for any bug.
