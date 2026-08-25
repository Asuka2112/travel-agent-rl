# TravelPlanner Validation Failure Decomposition

This is a read-only diagnosis of the frozen Base, SFT, and checkpoint-200 outputs on all
180 official TravelPlanner validation tasks. No model output, adapter output, prompt,
checkpoint, reward, data, seed, tool behavior, or evaluator result was changed or
regenerated. Per-task verdicts reproduce the official aggregate numerators exactly using
TravelPlanner evaluator commit `e52c87f4ac348a3410c46dc3553c519db5ec5e23`.

## Executive finding

The zero Final Pass result is dominated by a systematic output/adapter incompatibility,
not by 180 independently demonstrated planning failures. Every final answer that Base,
SFT, or RL produced had at least one adapter/schema finding. In particular, the adapter's
day-block fallback commonly put an entire prose day into `attraction` and filled
`current_city`, `transportation`, meals, and accommodation with `-`.

For RL, the 411 commonsense micro passes decompose exactly as 106 restaurant-diversity
passes + 106 attraction-diversity passes + 106 minimum-stay passes + 55 current-city
passes + 38 sandbox passes. The first three validators pass vacuously when their fields
are absent. At the same time, all 106 delivered RL plans failed `is_not_absent`,
`is_reasonable_visiting_city`, and `is_valid_transportation`. Consequently no RL task
reached the official hard-constraint gate and Commonsense Macro, Hard Micro/Macro, and
Final Pass were all zero.

This is evidence of a pure engineering problem in the frozen interface adapter. Per the
protocol, it is reported only: the adapter is not changed and evaluation is not rerun.
The data do not support labeling the observed 0/180 result as a clean generalization gap.
There are also genuine model/agent weaknesses—especially failure to return a final answer
and infeasible plans in the manual sample—but their official constraint impact is
confounded by the adapter failure.

## Aggregate metrics and delivery

| Metric | Base | SFT | RL |
|---|---:|---:|---:|
| Commonsense Micro | 40/1,440 (2.7778%) | 27/1,440 (1.8750%) | 411/1,440 (28.5417%) |
| Commonsense Macro | 0/180 | 0/180 | 0/180 |
| Hard Micro | 0/420 | 0/420 | 0/420 |
| Hard Macro | 0/180 | 0/180 | 0/180 |
| Final Pass | 0/180 | 0/180 | 0/180 |
| Official Delivery (`plan != null`) | 8/180 (4.4444%) | 6/180 (3.3333%) | 106/180 (58.8889%) |
| Project format-valid / Task Success | 6/180 (3.3333%) | 5/180 (2.7778%) | 78/180 (43.3333%) |
| Final answer present | 30/180 | 25/180 | 166/180 |

Project format-valid means only “adapter produced exactly the requested number of rows
with official schema keys.” It does not require non-missing field values and is not
official Final Pass.

## Official commonsense verdict counts

`NE` means the official evaluator did not evaluate the constraint because no plan was
delivered. Counts are pass / fail / NE.

| Official constraint | Base | SFT | RL |
|---|---:|---:|---:|
| Reasonable city route | 0 / 8 / 172 | 0 / 6 / 174 | 0 / 106 / 74 |
| Diverse restaurants | 8 / 0 / 172 | 6 / 0 / 174 | 106 / 0 / 74 |
| Diverse attractions | 8 / 0 / 172 | 6 / 0 / 174 | 106 / 0 / 74 |
| Minimum nights stay | 8 / 0 / 172 | 6 / 0 / 174 | 106 / 0 / 74 |
| Non-conflicting transportation | 0 / 8 / 172 | 0 / 6 / 174 | 0 / 106 / 74 |
| Within current city | 8 / 0 / 172 | 5 / 1 / 174 | 55 / 51 / 74 |
| Within official sandbox | 8 / 0 / 172 | 4 / 2 / 174 | 38 / 68 / 74 |
| Complete information (`is_not_absent`) | 0 / 8 / 172 | 0 / 6 / 174 | 0 / 106 / 74 |

The official hard evaluator was gated off for all 540 model-task pairs because no
delivered plan simultaneously passed completeness and sandbox validation. Thus every
official hard verdict—budget, room rule, cuisine, room type, and transportation—is
`not_evaluated` for Base 180/180, SFT 180/180, and RL 180/180. Diagnostic ungated calls
are retained in the JSON but are not substituted for official hard metrics; they are
also unreliable on missing-field plans. For example, ungated `valid_cost` passes all
delivered plans because absent expenses contribute zero.

## Distance to Final Pass

Distance is the number of failed/not-evaluated commonsense checks plus failed/not-evaluated
applicable hard checks from diagnostic ungated calls. It is a constraint-check distance,
not a claim that fixing that many text spans would make the plan valid.

| Distance | Base tasks | SFT tasks | RL tasks |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 5 | 3 | 25 |
| 4 | 3 | 2 | 27 |
| 5 | 0 | 0 | 41 |
| 6 | 0 | 1 | 13 |
| 9 | 59 | 58 | 27 |
| 10 | 55 | 57 | 22 |
| 12 | 58 | 59 | 25 |

There are no tasks only one or two official checks from Final Pass. The 25 closest RL
tasks are three checks away, but all 25 still have the same field-collapse adapter defect;
they are not near-valid official itineraries.

## Failure taxonomy

Counts within a column can overlap. “Confirmed database mismatch” requires a structurally
usable entity string that is absent/different in the official database. Sandbox failures
on already malformed adapter output are conservatively assigned to adapter confounding,
not to the database.

| Failure evidence | Base | SFT | RL |
|---|---:|---:|---:|
| A: no final answer / other agent completion failure | 150 | 155 | 14 |
| A: confirmed budget failure, unconfounded | 0 | 0 | 0 |
| A: confirmed transportation failure, unconfounded | 0 | 0 | 0 |
| A: confirmed accommodation failure, unconfounded | 0 | 0 | 0 |
| A: confirmed cuisine failure, unconfounded | 0 | 0 | 0 |
| A: confirmed attraction failure, unconfounded | 0 | 0 | 0 |
| A: confirmed temporal/city-coverage failure, unconfounded | 0 | 0 | 0 |
| B: parser failure (final answer present, `plan=null`) | 22 | 19 | 60 |
| B: missing required field | 8 | 6 | 106 |
| B: malformed itinerary structure | 11 findings | 8 findings | 157 findings |
| B: entity validation confounded by adapter | 0 | 1 | 52 |
| B: tasks with any parser/schema/adapter failure | 30 | 25 | 166 |
| C: confirmed official database/entity mismatch tasks | 0 | 0 | 0 |
| C: confirmed official tool-unavailable mismatch tasks | 0 | 0 | 0 |
| C: confirmed local-vs-official record mismatch tasks | 0 | 0 | 0 |

The zeros in the unconfounded A rows do not establish that planning was correct: there
were no schema-clean delivered plans on which those causes could be isolated. The manual
audit below contains qualitative planning failures, but they cannot be assigned an
official constraint verdict after `plan=null` or field collapse.

## RL 78/180 internal Task Success subset

All 78 project-format-valid RL tasks still failed official Final Pass. Their blocking
constraint counts were:

| Blocking check | Tasks |
|---|---:|
| Complete information | 78/78 |
| Reasonable city route | 78/78 |
| Non-conflicting transportation | 78/78 |
| Within official sandbox | 49/78 |
| Within current city | 36/78 |
| Diagnostic ungated cuisine | 24/78 |

Every one of the 78 has missing required fields and malformed itinerary structure. Their
distance histogram is 3: 19 tasks, 4: 20, 5: 28, and 6: 11. In 36 cases an official
sandbox rejection is directly confounded by adapter field assignment. Therefore the
78/180 project Task Success statistic measures row-count/schema-key delivery, not usable
official itinerary compliance.

## Random manual audit

Sampling used fixed seed 20260826. The closest pool contained 25 RL tasks at distance 3;
the worst pool contained 25 at distance 12.

### Ten closest RL tasks

| Official idx | Audit observation |
|---:|---|
| 19 | Prose contains flights/lodging/attractions; adapter stored each day only as `attraction`, leaving six required fields empty. |
| 32 | Five dated days collapsed into `attraction`; city route and flights disappeared from their official fields. |
| 36 | Flight-rich Texas itinerary collapsed into `attraction`; no current city or transportation survived. |
| 39 | California route has dated prose but the same six-field omission. |
| 55 | Seven-day answer became only four adapted rows, all attraction-only; row count and fields are malformed. |
| 99 | Flight, hotel, and city prose present, but only attraction is populated in the official plan. |
| 118 | Seven-day Illinois prose became seven attraction-only rows; accommodation constraint cannot be evaluated correctly. |
| 121 | Three-day no-driving answer collapsed into attraction-only rows. |
| 122 | Smoking/non-shared lodging prose present, but official lodging and transport fields are empty. |
| 125 | Flights and lodging are present in prose; adapter drops them into attraction-only rows. |

All ten share the same systematic adapter failure. None is a schema-clean near-pass.

### Ten worst RL tasks

| Official idx | Audit observation |
|---:|---|
| 123 | Final prose exists but has no parseable daily schema; `plan=null`. |
| 124 | Section-based flights/accommodation/dining answer; parser produces `plan=null`. |
| 126 | `plan=null`; prose also admits accommodation choices exceed budget, providing independent planning-failure evidence. |
| 131 | `plan=null`; response cannot supply required pet-friendly lodging. |
| 132 | `plan=null`; response states no smoking-allowed lodging was found. |
| 142 | `plan=null`; response cannot satisfy pet-friendly entire-home requirement. |
| 171 | No final answer after the frozen turn budget. |
| 172 | Grouped date ranges rather than daily rows; `plan=null`. |
| 174 | `plan=null`; response declares the requested non-flight/private-room trip infeasible. |
| 177 | No final answer after the frozen turn budget. |

Eight of the ten worst tasks are parser failures and two have no final answer. Several
also show real planning/coverage gaps, so repairing the adapter would not imply that
Final Pass becomes high; it would only make a clean generalization measurement possible.

## Required decision

There is a systematic pure engineering issue: the frozen models generate prose while the
official evaluator requires fully populated daily fields, and the adapter does not
semantically map that prose. This explains the combination of RL Commonsense Micro
28.5417% and Final Pass 0/180. No fix or rerun has been performed. Approval is required
before any adapter correction or new evaluation; training must remain permanently frozen.
