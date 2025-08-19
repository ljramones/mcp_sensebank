# Sense Bank / Strands Test Suite

This file groups handy test commands for your `strands_chat.py` REPL.  
Keep the MCP server running in **Terminal A** and run the chat loop in **Terminal B**.

---

## Quick adds (Heian Japan + a couple Andes)

```
/add term="rain on paulownia leaves" category=smell locale=Japan era=Heian weather=rain register=court notes="metallic-cool sweetness from bruised leaves"
/add term="silk sleeve whisper" category=sound locale=Japan era=Heian weather=any register=court notes="brushed along lacquer rail"
/add term="wet lacquer rail" category=touch locale=Japan era=Heian weather=rain register=court notes="cool slickness under sleeve"
/add term="salt pickled plum" category=taste locale=Japan era=Heian weather=any register=court notes="briny tartness lingering on tongue"
/add term="bamboo downspout plink" category=sound locale=Japan era=Heian weather=rain register=common notes="plink | plunk pattern under eaves" 
/add term="scroll backing paste" category=smell locale=Japan era=Heian weather=any register=court notes="rice starch paste, faintly sour"
/add term="temple bell (bonshō) after storm" category=sound locale=Japan era=Heian weather=rain register=ritual notes="low bronze bloom through mist"
/add term="ichu grass after rain" category=smell locale=Andes era=Chimú weather=rain register=common notes="resinous, sun-baked hay released by the rain"
/add term="spondylus shell tang" category=smell locale=Andes era=Chimú weather=any register=ritual notes="briny, coppery edge from sacred shell"
```

> That `|` in notes tests Markdown escaping in your `/md` export.

---

## Duplicate checks (should return `status: exists`)

```
/add term="temple bell (bonshō) after storm" category=sound locale=Japan era=Heian weather=rain register=ritual notes="duplicate check"
```

---

## List filters, limits, sorting

```
/list locale=Japan era=Heian category=smell sort=term limit=10
/list term_contains=ink locale=Japan era=Heian
/list notes_contains=bronze locale=Japan era=Heian
/list register=court locale=Japan era=Heian sort=category,term limit=50
/list locale=Andes era=Chimú sort=category,term
```

---

## Edit (dry run, then commit), identity change, duplicate block

```
/edit match_term="rain on paulownia leaves" match_category=smell match_locale=Japan match_era=Heian notes="metallic-cool sweetness, bruised leaf green" dry_run=true
/edit match_term="rain on paulownia leaves" match_category=smell match_locale=Japan match_era=Heian notes="metallic-cool sweetness, bruised leaf green"
/edit match_term="silk sleeve whisper" match_category=sound match_locale=Japan match_era=Heian term="silk sleeve rustle"
/edit match_term="silk sleeve rustle" match_category=sound match_locale=Japan match_era=Heian term="temple bell (bonshō) after storm"
```

> The last line should return `status: duplicate`.

---

## Delete (preview then commit)

```
/delete match_term="scroll backing paste" match_category=smell match_locale=Japan match_era=Heian dry_run=true
/delete match_term="scroll backing paste" match_category=smell match_locale=Japan match_era=Heian
```

---

## Exports (CSV / Markdown / XLSX)

```
/csv locale=Japan era=Heian register=court out=heian_court.csv
/csv locale=Japan era=Heian register=court out=heian_court.csv append=true include_header=false
/md  category=sound locale=Japan era=Heian out=heian_sounds.md
/md  term_contains=plink locale=Japan era=Heian out=heian_misc.md append=true include_header=false
/xlsx category=smell locale=Japan era=Heian out=heian_smells.xlsx overwrite=true
```

---

## File routing explicit (force a target CSV), then list by file

```
/add term="cedar incense ash" category=smell locale=Japan era=Heian weather=any register=ritual notes="dry resin ash" file=jp_sensory.csv
/list file=jp_sensory.csv term_contains=incense sort=term
```

---

## Suggestions + rewrite sanity

```
/suggest
/rewrite After the storm the courtyard felt rinsed and quiet; she lingered beneath the eaves and tried to breathe it in.
```

---

## Batch run (optional)

You can pipe this whole block into the chat loop (with server running) to exercise the flow end-to-end:

```bash
python strands_chat.py <<'EOF'
/add term="rain on paulownia leaves" category=smell locale=Japan era=Heian weather=rain register=court notes="metallic-cool sweetness from bruised leaves"
/add term="silk sleeve whisper" category=sound locale=Japan era=Heian weather=any register=court notes="brushed along lacquer rail"
/add term="wet lacquer rail" category=touch locale=Japan era=Heian weather=rain register=court notes="cool slickness under sleeve"
/add term="salt pickled plum" category=taste locale=Japan era=Heian weather=any register=court notes="briny tartness lingering on tongue"
/add term="bamboo downspout plink" category=sound locale=Japan era=Heian weather=rain register=common notes="plink | plunk pattern under eaves"
/add term="temple bell (bonshō) after storm" category=sound locale=Japan era=Heian weather=rain register=ritual notes="low bronze bloom through mist"
/list locale=Japan era=Heian category=smell sort=term limit=10
/edit match_term="silk sleeve whisper" match_category=sound match_locale=Japan match_era=Heian term="silk sleeve rustle"
/md  category=sound locale=Japan era=Heian out=heian_sounds.md
/csv locale=Japan era=Heian register=court out=heian_court.csv
/xlsx category=smell locale=Japan era=Heian out=heian_smells.xlsx overwrite=true
/rewrite After the storm the courtyard felt rinsed and quiet; she lingered beneath the eaves and tried to breathe it in.
/quit
EOF
```

