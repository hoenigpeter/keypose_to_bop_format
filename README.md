# Keypose to BOP Dataset Format Converter
Keypose: https://sites.google.com/view/keypose/ <br>
BOP Dataset Format: https://bop.felk.cvut.cz/challenges/ <br>
 <br>

This dataset conversion currently processes the Keypose dataset into the BOP dataset format for 6D known object pose estimation. <br>
Therefore, the current conversion does not support category-level pose estimation, instance level only. <br>
Some code was taken from the original Keypose repository: https://github.com/google-research/google-research/tree/master/keypose

## Convert annotation to BOP format
`python convert_to_bop.py <input_folder> <mesh_folder> --depth_type [opaque / transparent] --copy_images`

## Convert obj Models
`python convert_meshes.py <directory-to-meshes>`

## Differences to the original Keypose dataset!
- The resizing operation was omitted