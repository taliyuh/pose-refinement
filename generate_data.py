import blenderproc as bproc
import numpy as np
import json
import os
import random
import math
import bpy

# initialise blenderproc + clear the default scene 
bproc.init()

# config variables
NUM_FRAMES = 5000  # how many images to generate
OUTPUT_DIR = "synthetic_dataset" # where to generate
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/annotations", exist_ok=True)

# camera res
bproc.camera.set_resolution(640, 480)

### board ###

# import the .stl object
obj = bproc.loader.load_obj("nist_atb_m1.stl")[0]
# scale board down to ~20 cm
SCALE_FACTOR = 0.006
obj.set_scale([SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR])
# board slightly up to not collide with tablecloth
obj.set_location([0, 0, 0.005])
obj.set_cp("category_id", 1)  # explicit id for segmentation masks

# create an opaque dark matte material for the board (like a real pcb/circuit board)
board_mat = bproc.material.create("board_material")
board_mat.set_principled_shader_value("Base Color", [0.08, 0.12, 0.18, 1.0])  # dark navy/slate
board_mat.set_principled_shader_value("Roughness", 0.7)  # semi-matte surface
board_mat.set_principled_shader_value("Specular IOR Level", 0.08)  # very low specular

obj.add_material(board_mat)

# specific landmarks on the board for nn
keypoints_3d = np.array([
    [-0.0059, 1.2542, 2.5],
    [-0.2742, 2.131, 2.5],
    [11.8038, 6.143, 2.5],
    [12.8518, 6.8691, 2.5],
    [-6.7563, -3.8596, 2.5],
    [-6.1847, -3.0241, 2.5],
    [5.9785, -9.4519, 2.5],
    [5.3693, -10.423, 2.5],
    [16.0, -12.5, 0.0],
    [-16.0, -12.5, 2.5],
    [16.0, -12.5, 2.5],
    [-16.0, 12.5, 0.0],
    [-16.0, 12.5, 2.5],
    [16.0, 12.5, 2.5]
]) * SCALE_FACTOR

### table ###

# table
desk = bproc.object.create_primitive('PLANE')
desk.set_scale([0.8, 0.8, 1])   # ~1.6m x 1.6m desk area
desk.set_location([0, 0, -0.002])  # bottom layer

desk_mat = bproc.material.create("desk_surface")
desk_bsdf = desk_mat.get_the_one_node_with_type("ShaderNodeBsdfPrincipled")
desk_bsdf.inputs["Base Color"].default_value = [0.55, 0.45, 0.35, 1.0]  # warm wood tone
desk_bsdf.inputs["Roughness"].default_value = 0.6
desk.add_material(desk_mat)

# checkerboard cloth
cloth = bproc.object.create_primitive('PLANE')
cloth.set_scale([0.35, 0.35, 1])   # ~0.7m checkerboard, sitting on desk
cloth.set_location([0, 0, -0.001])  # middle layer between table and the board

cloth_mat = bproc.material.create("checker_cloth")
mapping_node = cloth_mat.new_node("ShaderNodeMapping")
tex_coord_node = cloth_mat.new_node("ShaderNodeTexCoord")  
checker_node = cloth_mat.new_node("ShaderNodeTexChecker")
diffuse_node = cloth_mat.get_the_one_node_with_type("ShaderNodeBsdfPrincipled")

cloth_mat.link(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
cloth_mat.link(mapping_node.outputs["Vector"], checker_node.inputs["Vector"])
cloth_mat.link(checker_node.outputs["Color"], diffuse_node.inputs["Base Color"])
cloth.add_material(cloth_mat)

# store per-frame random seeds for lighting/pattern variation
random_seeds = [random.randint(0, 2**31) for _ in range(NUM_FRAMES)]

# i did not wanna replicate 1:1 cloth so i did some variations in generated images
def randomize_cloth(frame_idx):
    rng = random.Random(random_seeds[frame_idx])
    
    # square size variation
    scale_factor = rng.uniform(2.5, 6.0)
    mapping_node.inputs["Scale"].default_value = [scale_factor, scale_factor, 1.0]
    mapping_node.inputs["Scale"].keyframe_insert("default_value", frame=frame_idx)
    
    # slight rotation
    z_rotation = rng.uniform(-0.2, 0.2)
    mapping_node.inputs["Rotation"].default_value = [0.0, 0.0, z_rotation]
    mapping_node.inputs["Rotation"].keyframe_insert("default_value", frame=frame_idx)
    
    # classic black & white calibration checkerboard
    color1 = [0.9, 0.9, 0.9, 1.0]   # bright white
    color2 = [0.05, 0.05, 0.05, 1.0]  # near black
    
    checker_node.inputs["Color1"].default_value = color1
    checker_node.inputs["Color1"].keyframe_insert("default_value", frame=frame_idx)
    checker_node.inputs["Color2"].default_value = color2
    checker_node.inputs["Color2"].keyframe_insert("default_value", frame=frame_idx)

### lighting ###

# one large light above
ceiling_light = bproc.types.Light("AREA", name="ceiling_light")
ceiling_light.set_location([0, 0, 2.0])
ceiling_light.set_radius(1.5)
ceiling_light.set_energy(60)

# small fill light to reduce harsh shadows
fill_light = bproc.types.Light("POINT", name="fill_light")
fill_light.set_location([1.5, 1.5, 1.0])
fill_light.set_energy(10)

def setup_lighting(frame_idx):
    """Subtle lighting variation per frame — realistic indoor range."""
    rng = random.Random(random_seeds[frame_idx] + 9999)
    ceiling_light.set_energy(rng.uniform(40, 80), frame=frame_idx)
    fill_light.set_energy(rng.uniform(5, 20), frame=frame_idx)

### generation loop ####

bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"],
                                          default_values={"category_id": 0})

# list to track positions
camera_poses = []

# background + light randomisation
for frame_idx in range(NUM_FRAMES):
    randomize_cloth(frame_idx)
    setup_lighting(frame_idx)
    
    # camera View Positioning
    rng = random.Random(random_seeds[frame_idx] + 5555)
    r = rng.uniform(0.25, 0.64) # distance 
    azimuth = rng.uniform(0, 2 * math.pi) # horizontal circle
    elevation = rng.uniform(0, np.pi/4) # elevation angle
    
    # convert into spherical coordinates
    x = r * math.cos(azimuth) * math.sin(elevation)
    y = r * math.sin(azimuth) * math.sin(elevation)
    z = r * math.cos(elevation)
    
    # calculate focal target point +  offset to avoid dead-centering
    look_at = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), 0.0])
    
    # compute camera orientation vector
    cam_pose = bproc.camera.rotation_from_forward_vec(look_at - np.array([x, y, z]), up_axis='Y')
    hom_matrix = bproc.math.build_transformation_mat([x, y, z], cam_pose)
    bproc.camera.add_camera_pose(hom_matrix)
    camera_poses.append(hom_matrix)

# render all frames at once
data = bproc.renderer.render()

### labelling ###

# fetch camera matrix
cam_K = bproc.camera.get_intrinsics_as_K_matrix()

for frame_idx in range(NUM_FRAMES):
    # get the camera-to-world matrix for this frame, invert to get world-to-camera
    cam2world = bproc.camera.get_camera_pose(frame=frame_idx)
    world2cam = np.linalg.inv(cam2world)
    
    # convert points into homogenous vector and into camera frame
    keypoints_2d = []
    for kp in keypoints_3d:
        kp_hom = np.append(kp, 1.0)
        kp_cam = (world2cam @ kp_hom)[:3]
        kp_pixel_hom = cam_K @ kp_cam
        
        # transform into pixel coordinates on the image
        u = float(kp_pixel_hom[0] / kp_pixel_hom[2])
        v = float(kp_pixel_hom[1] / kp_pixel_hom[2])
        keypoints_2d.append([u, v])
        
    # package data into json
    annotation = {
        "frame_id": frame_idx,
        "keypoints_2d": keypoints_2d,
        "camera_matrix_K": cam_K.tolist(),
        "image_resolution": [640, 480]
    }
    
    with open(f"{OUTPUT_DIR}/annotations/frame_{frame_idx}.json", "w") as f:
        json.dump(annotation, f, indent=4)

# export rendered data to .h5
bproc.writer.write_hdf5(OUTPUT_DIR, data)