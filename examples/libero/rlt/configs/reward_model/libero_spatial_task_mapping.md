# LIBERO Spatial Task Mapping

This file records the task mapping used by the reward-model RLT configs.
The local HDF5 files do not store a clean prompt attribute; prompt matching is checked through the LIBERO benchmark task metadata and the matching yixin LeRobot `tasks.jsonl` files.

- HDF5 root: `/vla/users/niejunnan/datasets/libero_spatial`
- Stage1 LeRobot home: `/vla/users/yixin/LIBERO/Libero_Lerobot`
- OpenPI checkpoint/assets: `/vla/users/niejunnan/assets/openpi-assets/serl_torch_ckpt/pi0_10000_pytorch`

| task_id | prompt | HDF5 file | Stage1 repo_id_override |
|---:|---|---|---|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` |
| 2 | pick up the black bowl from table center and place it on the plate | `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` |
| 3 | pick up the black bowl on the cookie box and place it on the plate | `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` |
| 5 | pick up the black bowl on the ramekin and place it on the plate | `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` |
| 7 | pick up the black bowl on the stove and place it on the plate | `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` |
| 8 | pick up the black bowl next to the plate and place it on the plate | `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5` | `libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` |
