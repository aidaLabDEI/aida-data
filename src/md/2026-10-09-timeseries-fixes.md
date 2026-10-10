# Plan: tolerate `:` in `.tsf` attribute values

Follow-up to `2026-10-08-timeseries.md`, from the code review of
`c8c3f90..6e3d787` (finding 7). Nothing here is implemented yet.

Goal: `_load_tsf` reads series lines whose `start_timestamp` is written as
`HH:MM:SS`, and gives a clear error for a line it really cannot split.

## 1. The problem

`_load_tsf` (`src/aida_data/timeseries.py:128-131`) splits each data line
like this:

```python
fields, _, raw = line.rpartition(":")
fields = fields.split(":")
if len(fields) != len(attributes):
    raise ValueError(f"{path}: expected {len(attributes)} attributes")
```

and parses the timestamp with `clock.replace('-', ':')`, which assumes the
Monash form `2010-01-01 00-00-00`.

How serious this is: the Monash format writes time with dashes precisely
because `:` is the field separator. As far as I remember, the reference
`convert_tsf_to_dataframe` also splits on every `:` and rejects a line with
extra fields, so files from the Monash archive never have this problem. This
still needs checking against the Monash repository before relying on it. The
risk is in `.tsf` files from other writers (MOMENTI and others) that use
`HH:MM:SS`. Today those fail with a misleading "expected N attributes"
error. **Low priority**: it doesn't affect any registered dataset
(`oikolab-weather` parses fine).

### Why not the review's suggested fix

The review proposes `line.split(":", len(attributes))`. That splits from the
left, so a colon in the *last* attribute (where `start_timestamp` goes in
the Monash files) still splits the timestamp, and its tail ends up glued to
the values (`"00:00,1.0,2.0..."`). The values float parse then fails, or
worse. It only helps when the colons are in the values, and the values never
contain `:`.

## 2. Design

- **Values never contain `:`** (comma-separated numbers or `?`), so
  `rpartition(":")` is the right way to separate them. Keep it.
- Split the attribute part with `fields.split(":", len(attributes) - 1)`.
  The first N − 1 attributes are taken from the left, and any extra colons
  stay in the last attribute. This handles a colon in the last attribute,
  which is the `start_timestamp` case.
- Colons in any other attribute make the line ambiguous. Keep raising, but
  with a message that says why: `"{path}: line {i} has more ':' than
  attributes; only the last attribute ({attributes[-1]}) may contain ':'"`.
  This needs the line number, so loop with `enumerate(f, 1)`.
- If the file has fewer fields than attributes, keep the current error and
  add the line number.
- Timestamp: accept both clock forms. After `partition(" ")`, normalise with
  `clock.replace("-", ":")`. This is already a no-op for `HH:MM:SS`, so the
  parse needs no change once the split leaves the timestamp in one piece.
  Also accept a date with no clock (`"2010-01-01"`): today `partition` gives
  `clock == ""` and the `datetime64` parse of `"2010-01-01T"` fails.
- Edge case: zero `@attribute` lines. `split(":", -1)` would split
  everything. Check this case up front: with no attributes the line must be
  just the values, and `fields` is empty.

## 3. Implementation steps

1. [ ] **Check the claim** in §1 against the Monash `data_loader.py`
   (`convert_tsf_to_dataframe`) and correct this note if it is wrong. It
   only changes the priority, not the design.
2. [ ] **Tests first** in `tests/test_timeseries.py`, next to the existing
   `test_load_tsf_*` tests:
   - `start_timestamp` as `2010-01-01 00:00:00` as the last attribute →
     parses, and `time[0]` equals `np.datetime64("2010-01-01T00:00:00")`;
   - the same file with the Monash form `00-00-00` gives identical output
     (this test already exists in spirit, keep it);
   - a date-only `start_timestamp` → the time stamps start at midnight;
   - a colon in a non-last attribute → `ValueError` whose message names the
     line number and the last attribute;
   - too few fields → `ValueError` with the line number.
3. [ ] **Fix `_load_tsf`** as in §2.
4. [ ] Run `tests/test_timeseries.py`, then reload `oikolab-weather` from a
   fresh download (`AIDA_DATA_KEEP_RAW=1`, delete its cache) and compare it
   with the old cache: values, `dim_names` and `time` must be identical.
