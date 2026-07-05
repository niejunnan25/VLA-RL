# OpenPI One-Shot Rot180 LIBERO Evaluation

- Generated: 2026-07-05 10:36:40 CST
- Output root: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515`
- Suites: `libero_spatial, libero_object, libero_goal, libero_10`
- Trials: `50` per task
- Action chunk execution: `replan_steps=5` (only the first 5 actions from each 10-action prediction are executed)

## Checkpoint Summary

| Policy | Step | Host | GPU | Status | Success | Episodes | Rate |
|---|---:|---|---:|---|---:|---:|---:|
| pi05_libero_one_shot_spatial_traj1_rot180 | 10000 | 20250106-instance | 0 | logs complete | 299 | 2000 | 14.9% |
| pi05_libero_one_shot_spatial_traj1_rot180 | 20000 | 20250106-instance | 1 | logs complete | 260 | 2000 | 13.0% |
| pi05_libero_one_shot_spatial_traj1_rot180 | 30000 | 20250106-instance | 4 | logs complete | 273 | 2000 | 13.7% |
| pi05_libero_one_shot_spatial_traj1_rot180 | 40000 | 20250106-instance | 6 | logs complete | 247 | 2000 | 12.3% |
| pi05_libero_one_shot_spatial_traj1_rot180 | 49999 | 20250106-instance | split-suite | logs complete | 261 | 2000 | 13.1% |
| pi05_libero_one_shot_all_traj1_rot180 | 10000 | 20250106-instance | 1 | logs complete | 1014 | 2000 | 50.7% |
| pi05_libero_one_shot_all_traj1_rot180 | 20000 | 20250106-instance | 2 | logs complete | 952 | 2000 | 47.6% |
| pi05_libero_one_shot_all_traj1_rot180 | 30000 | 20250106-instance | 3 | logs complete | 926 | 2000 | 46.3% |
| pi05_libero_one_shot_all_traj1_rot180 | 40000 | 20250106-instance | 7 | logs complete | 905 | 2000 | 45.2% |
| pi05_libero_one_shot_all_traj1_rot180 | 49999 | 20250106-instance | 5 | logs complete | 872 | 2000 | 43.6% |

## pi05_libero_one_shot_spatial_traj1_rot180 / 10000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `0`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/10000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_spatial_traj1_rot180_10000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 274 | 500 | 54.8% |
| libero_object | 0 | 500 | 0.0% |
| libero_goal | 25 | 500 | 5.0% |
| libero_10 | 0 | 500 | 0.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 40 | 50 | 80.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 27 | 50 | 54.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 44 | 50 | 88.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 26 | 50 | 52.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 38 | 50 | 76.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 24 | 50 | 48.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 5 | 50 | 10.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 29 | 50 | 58.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 32 | 50 | 64.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 9 | 50 | 18.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0 | 50 | 0.0% |
| 1 | pick up the cream cheese and place it in the basket | 0 | 50 | 0.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 0 | 50 | 0.0% |
| 6 | pick up the butter and place it in the basket | 0 | 50 | 0.0% |
| 7 | pick up the milk and place it in the basket | 0 | 50 | 0.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 0 | 50 | 0.0% |
| 9 | pick up the orange juice and place it in the basket | 0 | 50 | 0.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 0 | 50 | 0.0% |
| 1 | put the bowl on the stove | 0 | 50 | 0.0% |
| 2 | put the wine bottle on top of the cabinet | 0 | 50 | 0.0% |
| 3 | open the top drawer and put the bowl inside | 0 | 50 | 0.0% |
| 4 | put the bowl on top of the cabinet | 0 | 50 | 0.0% |
| 5 | push the plate to the front of the stove | 0 | 50 | 0.0% |
| 6 | put the cream cheese in the bowl | 0 | 50 | 0.0% |
| 7 | turn on the stove | 0 | 50 | 0.0% |
| 8 | put the bowl on the plate | 25 | 50 | 50.0% |
| 9 | put the wine bottle on the rack | 0 | 50 | 0.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 | 50 | 0.0% |
| 1 | put both the cream cheese box and the butter in the basket | 0 | 50 | 0.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 | 50 | 0.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 50 | 0.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 | 50 | 0.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0 | 50 | 0.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 0 | 50 | 0.0% |

## pi05_libero_one_shot_spatial_traj1_rot180 / 20000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `1`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/20000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_spatial_traj1_rot180_20000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 231 | 500 | 46.2% |
| libero_object | 0 | 500 | 0.0% |
| libero_goal | 29 | 500 | 5.8% |
| libero_10 | 0 | 500 | 0.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 32 | 50 | 64.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 19 | 50 | 38.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 43 | 50 | 86.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 19 | 50 | 38.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 29 | 50 | 58.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 18 | 50 | 36.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 6 | 50 | 12.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 30 | 50 | 60.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 30 | 50 | 60.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 5 | 50 | 10.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0 | 50 | 0.0% |
| 1 | pick up the cream cheese and place it in the basket | 0 | 50 | 0.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 0 | 50 | 0.0% |
| 6 | pick up the butter and place it in the basket | 0 | 50 | 0.0% |
| 7 | pick up the milk and place it in the basket | 0 | 50 | 0.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 0 | 50 | 0.0% |
| 9 | pick up the orange juice and place it in the basket | 0 | 50 | 0.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 0 | 50 | 0.0% |
| 1 | put the bowl on the stove | 0 | 50 | 0.0% |
| 2 | put the wine bottle on top of the cabinet | 0 | 50 | 0.0% |
| 3 | open the top drawer and put the bowl inside | 0 | 50 | 0.0% |
| 4 | put the bowl on top of the cabinet | 0 | 50 | 0.0% |
| 5 | push the plate to the front of the stove | 0 | 50 | 0.0% |
| 6 | put the cream cheese in the bowl | 0 | 50 | 0.0% |
| 7 | turn on the stove | 0 | 50 | 0.0% |
| 8 | put the bowl on the plate | 29 | 50 | 58.0% |
| 9 | put the wine bottle on the rack | 0 | 50 | 0.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 | 50 | 0.0% |
| 1 | put both the cream cheese box and the butter in the basket | 0 | 50 | 0.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 | 50 | 0.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 50 | 0.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 | 50 | 0.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0 | 50 | 0.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 0 | 50 | 0.0% |

## pi05_libero_one_shot_spatial_traj1_rot180 / 30000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `4`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/30000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_spatial_traj1_rot180_30000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 245 | 500 | 49.0% |
| libero_object | 0 | 500 | 0.0% |
| libero_goal | 28 | 500 | 5.6% |
| libero_10 | 0 | 500 | 0.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 35 | 50 | 70.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 22 | 50 | 44.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 46 | 50 | 92.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 28 | 50 | 56.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 28 | 50 | 56.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 17 | 50 | 34.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 8 | 50 | 16.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 25 | 50 | 50.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 31 | 50 | 62.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 5 | 50 | 10.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0 | 50 | 0.0% |
| 1 | pick up the cream cheese and place it in the basket | 0 | 50 | 0.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 0 | 50 | 0.0% |
| 6 | pick up the butter and place it in the basket | 0 | 50 | 0.0% |
| 7 | pick up the milk and place it in the basket | 0 | 50 | 0.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 0 | 50 | 0.0% |
| 9 | pick up the orange juice and place it in the basket | 0 | 50 | 0.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 0 | 50 | 0.0% |
| 1 | put the bowl on the stove | 0 | 50 | 0.0% |
| 2 | put the wine bottle on top of the cabinet | 0 | 50 | 0.0% |
| 3 | open the top drawer and put the bowl inside | 0 | 50 | 0.0% |
| 4 | put the bowl on top of the cabinet | 0 | 50 | 0.0% |
| 5 | push the plate to the front of the stove | 0 | 50 | 0.0% |
| 6 | put the cream cheese in the bowl | 0 | 50 | 0.0% |
| 7 | turn on the stove | 0 | 50 | 0.0% |
| 8 | put the bowl on the plate | 28 | 50 | 56.0% |
| 9 | put the wine bottle on the rack | 0 | 50 | 0.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 | 50 | 0.0% |
| 1 | put both the cream cheese box and the butter in the basket | 0 | 50 | 0.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 | 50 | 0.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 50 | 0.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 | 50 | 0.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0 | 50 | 0.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 0 | 50 | 0.0% |

## pi05_libero_one_shot_spatial_traj1_rot180 / 40000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `6`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/40000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_spatial_traj1_rot180_40000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 210 | 500 | 42.0% |
| libero_object | 0 | 500 | 0.0% |
| libero_goal | 37 | 500 | 7.4% |
| libero_10 | 0 | 500 | 0.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 26 | 50 | 52.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 14 | 50 | 28.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 42 | 50 | 84.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 13 | 50 | 26.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 30 | 50 | 60.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 25 | 50 | 50.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 4 | 50 | 8.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 27 | 50 | 54.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 25 | 50 | 50.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 4 | 50 | 8.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0 | 50 | 0.0% |
| 1 | pick up the cream cheese and place it in the basket | 0 | 50 | 0.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 0 | 50 | 0.0% |
| 6 | pick up the butter and place it in the basket | 0 | 50 | 0.0% |
| 7 | pick up the milk and place it in the basket | 0 | 50 | 0.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 0 | 50 | 0.0% |
| 9 | pick up the orange juice and place it in the basket | 0 | 50 | 0.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 0 | 50 | 0.0% |
| 1 | put the bowl on the stove | 0 | 50 | 0.0% |
| 2 | put the wine bottle on top of the cabinet | 0 | 50 | 0.0% |
| 3 | open the top drawer and put the bowl inside | 0 | 50 | 0.0% |
| 4 | put the bowl on top of the cabinet | 0 | 50 | 0.0% |
| 5 | push the plate to the front of the stove | 0 | 50 | 0.0% |
| 6 | put the cream cheese in the bowl | 0 | 50 | 0.0% |
| 7 | turn on the stove | 0 | 50 | 0.0% |
| 8 | put the bowl on the plate | 37 | 50 | 74.0% |
| 9 | put the wine bottle on the rack | 0 | 50 | 0.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 | 50 | 0.0% |
| 1 | put both the cream cheese box and the butter in the basket | 0 | 50 | 0.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 | 50 | 0.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 50 | 0.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 | 50 | 0.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0 | 50 | 0.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 0 | 50 | 0.0% |

## pi05_libero_one_shot_spatial_traj1_rot180 / 49999

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `split-suite`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_spatial_traj1_rot180/49999`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_spatial_traj1_rot180_49999`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 227 | 500 | 45.4% |
| libero_object | 0 | 500 | 0.0% |
| libero_goal | 34 | 500 | 6.8% |
| libero_10 | 0 | 500 | 0.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 30 | 50 | 60.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 17 | 50 | 34.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 45 | 50 | 90.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 19 | 50 | 38.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 33 | 50 | 66.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 19 | 50 | 38.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 5 | 50 | 10.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 27 | 50 | 54.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 26 | 50 | 52.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 6 | 50 | 12.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 0 | 50 | 0.0% |
| 1 | pick up the cream cheese and place it in the basket | 0 | 50 | 0.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 0 | 50 | 0.0% |
| 6 | pick up the butter and place it in the basket | 0 | 50 | 0.0% |
| 7 | pick up the milk and place it in the basket | 0 | 50 | 0.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 0 | 50 | 0.0% |
| 9 | pick up the orange juice and place it in the basket | 0 | 50 | 0.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 0 | 50 | 0.0% |
| 1 | put the bowl on the stove | 0 | 50 | 0.0% |
| 2 | put the wine bottle on top of the cabinet | 0 | 50 | 0.0% |
| 3 | open the top drawer and put the bowl inside | 0 | 50 | 0.0% |
| 4 | put the bowl on top of the cabinet | 0 | 50 | 0.0% |
| 5 | push the plate to the front of the stove | 0 | 50 | 0.0% |
| 6 | put the cream cheese in the bowl | 0 | 50 | 0.0% |
| 7 | turn on the stove | 0 | 50 | 0.0% |
| 8 | put the bowl on the plate | 34 | 50 | 68.0% |
| 9 | put the wine bottle on the rack | 0 | 50 | 0.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 | 50 | 0.0% |
| 1 | put both the cream cheese box and the butter in the basket | 0 | 50 | 0.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 0 | 50 | 0.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 50 | 0.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 | 50 | 0.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0 | 50 | 0.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 0 | 50 | 0.0% |

## pi05_libero_one_shot_all_traj1_rot180 / 10000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `1`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_all_traj1_rot180/10000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_all_traj1_rot180_10000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 339 | 500 | 67.8% |
| libero_object | 190 | 500 | 38.0% |
| libero_goal | 286 | 500 | 57.2% |
| libero_10 | 199 | 500 | 39.8% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 44 | 50 | 88.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 48 | 50 | 96.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 48 | 50 | 96.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 47 | 50 | 94.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 35 | 50 | 70.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 38 | 50 | 76.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 5 | 50 | 10.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 28 | 50 | 56.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 23 | 50 | 46.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 23 | 50 | 46.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 37 | 50 | 74.0% |
| 1 | pick up the cream cheese and place it in the basket | 4 | 50 | 8.0% |
| 2 | pick up the salad dressing and place it in the basket | 4 | 50 | 8.0% |
| 3 | pick up the bbq sauce and place it in the basket | 1 | 50 | 2.0% |
| 4 | pick up the ketchup and place it in the basket | 1 | 50 | 2.0% |
| 5 | pick up the tomato sauce and place it in the basket | 6 | 50 | 12.0% |
| 6 | pick up the butter and place it in the basket | 41 | 50 | 82.0% |
| 7 | pick up the milk and place it in the basket | 47 | 50 | 94.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 16 | 50 | 32.0% |
| 9 | pick up the orange juice and place it in the basket | 33 | 50 | 66.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 3 | 50 | 6.0% |
| 1 | put the bowl on the stove | 50 | 50 | 100.0% |
| 2 | put the wine bottle on top of the cabinet | 48 | 50 | 96.0% |
| 3 | open the top drawer and put the bowl inside | 7 | 50 | 14.0% |
| 4 | put the bowl on top of the cabinet | 49 | 50 | 98.0% |
| 5 | push the plate to the front of the stove | 14 | 50 | 28.0% |
| 6 | put the cream cheese in the bowl | 37 | 50 | 74.0% |
| 7 | turn on the stove | 29 | 50 | 58.0% |
| 8 | put the bowl on the plate | 44 | 50 | 88.0% |
| 9 | put the wine bottle on the rack | 5 | 50 | 10.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 5 | 50 | 10.0% |
| 1 | put both the cream cheese box and the butter in the basket | 31 | 50 | 62.0% |
| 2 | turn on the stove and put the moka pot on it | 36 | 50 | 72.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 9 | 50 | 18.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 17 | 50 | 34.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 49 | 50 | 98.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 1 | 50 | 2.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 40 | 50 | 80.0% |
| 8 | put both moka pots on the stove | 4 | 50 | 8.0% |
| 9 | put the yellow and white mug in the microwave and close it | 7 | 50 | 14.0% |

## pi05_libero_one_shot_all_traj1_rot180 / 20000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `2`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_all_traj1_rot180/20000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_all_traj1_rot180_20000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 353 | 500 | 70.6% |
| libero_object | 182 | 500 | 36.4% |
| libero_goal | 257 | 500 | 51.4% |
| libero_10 | 160 | 500 | 32.0% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 44 | 50 | 88.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 47 | 50 | 94.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 50 | 50 | 100.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 37 | 50 | 74.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 42 | 50 | 84.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 27 | 50 | 54.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 11 | 50 | 22.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 33 | 50 | 66.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 30 | 50 | 60.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 32 | 50 | 64.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 24 | 50 | 48.0% |
| 1 | pick up the cream cheese and place it in the basket | 8 | 50 | 16.0% |
| 2 | pick up the salad dressing and place it in the basket | 1 | 50 | 2.0% |
| 3 | pick up the bbq sauce and place it in the basket | 1 | 50 | 2.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 8 | 50 | 16.0% |
| 6 | pick up the butter and place it in the basket | 39 | 50 | 78.0% |
| 7 | pick up the milk and place it in the basket | 48 | 50 | 96.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 25 | 50 | 50.0% |
| 9 | pick up the orange juice and place it in the basket | 28 | 50 | 56.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 2 | 50 | 4.0% |
| 1 | put the bowl on the stove | 49 | 50 | 98.0% |
| 2 | put the wine bottle on top of the cabinet | 40 | 50 | 80.0% |
| 3 | open the top drawer and put the bowl inside | 8 | 50 | 16.0% |
| 4 | put the bowl on top of the cabinet | 47 | 50 | 94.0% |
| 5 | push the plate to the front of the stove | 9 | 50 | 18.0% |
| 6 | put the cream cheese in the bowl | 40 | 50 | 80.0% |
| 7 | turn on the stove | 13 | 50 | 26.0% |
| 8 | put the bowl on the plate | 44 | 50 | 88.0% |
| 9 | put the wine bottle on the rack | 5 | 50 | 10.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 1 | 50 | 2.0% |
| 1 | put both the cream cheese box and the butter in the basket | 27 | 50 | 54.0% |
| 2 | turn on the stove and put the moka pot on it | 14 | 50 | 28.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 8 | 50 | 16.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 17 | 50 | 34.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 49 | 50 | 98.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 3 | 50 | 6.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 32 | 50 | 64.0% |
| 8 | put both moka pots on the stove | 4 | 50 | 8.0% |
| 9 | put the yellow and white mug in the microwave and close it | 5 | 50 | 10.0% |

## pi05_libero_one_shot_all_traj1_rot180 / 30000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `3`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_all_traj1_rot180/30000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_all_traj1_rot180_30000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 356 | 500 | 71.2% |
| libero_object | 203 | 500 | 40.6% |
| libero_goal | 243 | 500 | 48.6% |
| libero_10 | 124 | 500 | 24.8% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 49 | 50 | 98.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 49 | 50 | 98.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 50 | 50 | 100.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 42 | 50 | 84.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 43 | 50 | 86.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 33 | 50 | 66.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 8 | 50 | 16.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 31 | 50 | 62.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 29 | 50 | 58.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 22 | 50 | 44.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 30 | 50 | 60.0% |
| 1 | pick up the cream cheese and place it in the basket | 10 | 50 | 20.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 0 | 50 | 0.0% |
| 4 | pick up the ketchup and place it in the basket | 1 | 50 | 2.0% |
| 5 | pick up the tomato sauce and place it in the basket | 18 | 50 | 36.0% |
| 6 | pick up the butter and place it in the basket | 43 | 50 | 86.0% |
| 7 | pick up the milk and place it in the basket | 45 | 50 | 90.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 27 | 50 | 54.0% |
| 9 | pick up the orange juice and place it in the basket | 29 | 50 | 58.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 7 | 50 | 14.0% |
| 1 | put the bowl on the stove | 49 | 50 | 98.0% |
| 2 | put the wine bottle on top of the cabinet | 41 | 50 | 82.0% |
| 3 | open the top drawer and put the bowl inside | 6 | 50 | 12.0% |
| 4 | put the bowl on top of the cabinet | 47 | 50 | 94.0% |
| 5 | push the plate to the front of the stove | 3 | 50 | 6.0% |
| 6 | put the cream cheese in the bowl | 34 | 50 | 68.0% |
| 7 | turn on the stove | 3 | 50 | 6.0% |
| 8 | put the bowl on the plate | 45 | 50 | 90.0% |
| 9 | put the wine bottle on the rack | 8 | 50 | 16.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 1 | 50 | 2.0% |
| 1 | put both the cream cheese box and the butter in the basket | 27 | 50 | 54.0% |
| 2 | turn on the stove and put the moka pot on it | 4 | 50 | 8.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 5 | 50 | 10.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 10 | 50 | 20.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 47 | 50 | 94.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 25 | 50 | 50.0% |
| 8 | put both moka pots on the stove | 1 | 50 | 2.0% |
| 9 | put the yellow and white mug in the microwave and close it | 4 | 50 | 8.0% |

## pi05_libero_one_shot_all_traj1_rot180 / 40000

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `7`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_all_traj1_rot180/40000`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_all_traj1_rot180_40000`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 335 | 500 | 67.0% |
| libero_object | 210 | 500 | 42.0% |
| libero_goal | 244 | 500 | 48.8% |
| libero_10 | 116 | 500 | 23.2% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 48 | 50 | 96.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 48 | 50 | 96.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 50 | 50 | 100.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 23 | 50 | 46.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 40 | 50 | 80.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 29 | 50 | 58.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 8 | 50 | 16.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 35 | 50 | 70.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 33 | 50 | 66.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 21 | 50 | 42.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 35 | 50 | 70.0% |
| 1 | pick up the cream cheese and place it in the basket | 16 | 50 | 32.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 1 | 50 | 2.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 24 | 50 | 48.0% |
| 6 | pick up the butter and place it in the basket | 39 | 50 | 78.0% |
| 7 | pick up the milk and place it in the basket | 49 | 50 | 98.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 27 | 50 | 54.0% |
| 9 | pick up the orange juice and place it in the basket | 19 | 50 | 38.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 5 | 50 | 10.0% |
| 1 | put the bowl on the stove | 50 | 50 | 100.0% |
| 2 | put the wine bottle on top of the cabinet | 41 | 50 | 82.0% |
| 3 | open the top drawer and put the bowl inside | 6 | 50 | 12.0% |
| 4 | put the bowl on top of the cabinet | 44 | 50 | 88.0% |
| 5 | push the plate to the front of the stove | 3 | 50 | 6.0% |
| 6 | put the cream cheese in the bowl | 33 | 50 | 66.0% |
| 7 | turn on the stove | 10 | 50 | 20.0% |
| 8 | put the bowl on the plate | 47 | 50 | 94.0% |
| 9 | put the wine bottle on the rack | 5 | 50 | 10.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 2 | 50 | 4.0% |
| 1 | put both the cream cheese box and the butter in the basket | 16 | 50 | 32.0% |
| 2 | turn on the stove and put the moka pot on it | 1 | 50 | 2.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 8 | 50 | 16.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 16 | 50 | 32.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 44 | 50 | 88.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 | 50 | 0.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 22 | 50 | 44.0% |
| 8 | put both moka pots on the stove | 1 | 50 | 2.0% |
| 9 | put the yellow and white mug in the microwave and close it | 6 | 50 | 12.0% |

## pi05_libero_one_shot_all_traj1_rot180 / 49999

- Status: `logs complete`
- Host/GPU: `20250106-instance` / `5`
- Checkpoint: `/vla/users/niejunnan/assets/openpi-assets/pi05_libero_one_shot_all_traj1_rot180/49999`
- Logs: `/vla/users/niejunnan/codebase/openpi/outputs/openpi_one_shot_rot180_eval_20260705_010515/logs/pi05_libero_one_shot_all_traj1_rot180_49999`

### Suite Summary

| Suite | Success | Episodes | Rate |
|---|---:|---:|---:|
| libero_spatial | 348 | 500 | 69.6% |
| libero_object | 193 | 500 | 38.6% |
| libero_goal | 234 | 500 | 46.8% |
| libero_10 | 97 | 500 | 19.4% |

### libero_spatial

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 47 | 50 | 94.0% |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 47 | 50 | 94.0% |
| 2 | pick up the black bowl from table center and place it on the plate | 50 | 50 | 100.0% |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 33 | 50 | 66.0% |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 40 | 50 | 80.0% |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 33 | 50 | 66.0% |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 9 | 50 | 18.0% |
| 7 | pick up the black bowl on the stove and place it on the plate | 37 | 50 | 74.0% |
| 8 | pick up the black bowl next to the plate and place it on the plate | 31 | 50 | 62.0% |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 21 | 50 | 42.0% |

### libero_object

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | pick up the alphabet soup and place it in the basket | 34 | 50 | 68.0% |
| 1 | pick up the cream cheese and place it in the basket | 8 | 50 | 16.0% |
| 2 | pick up the salad dressing and place it in the basket | 0 | 50 | 0.0% |
| 3 | pick up the bbq sauce and place it in the basket | 1 | 50 | 2.0% |
| 4 | pick up the ketchup and place it in the basket | 0 | 50 | 0.0% |
| 5 | pick up the tomato sauce and place it in the basket | 25 | 50 | 50.0% |
| 6 | pick up the butter and place it in the basket | 34 | 50 | 68.0% |
| 7 | pick up the milk and place it in the basket | 49 | 50 | 98.0% |
| 8 | pick up the chocolate pudding and place it in the basket | 26 | 50 | 52.0% |
| 9 | pick up the orange juice and place it in the basket | 16 | 50 | 32.0% |

### libero_goal

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 4 | 50 | 8.0% |
| 1 | put the bowl on the stove | 49 | 50 | 98.0% |
| 2 | put the wine bottle on top of the cabinet | 40 | 50 | 80.0% |
| 3 | open the top drawer and put the bowl inside | 6 | 50 | 12.0% |
| 4 | put the bowl on top of the cabinet | 43 | 50 | 86.0% |
| 5 | push the plate to the front of the stove | 1 | 50 | 2.0% |
| 6 | put the cream cheese in the bowl | 42 | 50 | 84.0% |
| 7 | turn on the stove | 4 | 50 | 8.0% |
| 8 | put the bowl on the plate | 44 | 50 | 88.0% |
| 9 | put the wine bottle on the rack | 1 | 50 | 2.0% |

### libero_10

| Task | Description | Success | Episodes | Rate |
|---:|---|---:|---:|---:|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 1 | 50 | 2.0% |
| 1 | put both the cream cheese box and the butter in the basket | 10 | 50 | 20.0% |
| 2 | turn on the stove and put the moka pot on it | 0 | 50 | 0.0% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 3 | 50 | 6.0% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 8 | 50 | 16.0% |
| 5 | pick up the book and place it in the back compartment of the caddy | 45 | 50 | 90.0% |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 1 | 50 | 2.0% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 22 | 50 | 44.0% |
| 8 | put both moka pots on the stove | 0 | 50 | 0.0% |
| 9 | put the yellow and white mug in the microwave and close it | 7 | 50 | 14.0% |
