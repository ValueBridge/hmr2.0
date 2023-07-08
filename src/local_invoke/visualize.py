"""
Modul with visualization commands
"""

import invoke


@invoke.task
def predictions_on_photobridge_data(_context, config_path):
    """
    Visualize hmr2 prediction on photobridge data

    Args:
        _context (invoke.Conext): context instance
        config_path (str): path to configuration file
    """

    import pyglet
    pyglet.options["headless"] = True

    import glob
    import os

    import numpy as np
    import tqdm

    import main.config
    import main.model
    import visualise.trimesh_renderer
    import visualise.vis_util

    import photobridge.utilities

    config = photobridge.utilities.read_yaml(config_path)

    class DemoConfig(main.config.Config):

        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../logs/{}/{}'.format("paired(joints)", "base_model"))
        INITIALIZE_CUSTOM_REGRESSOR = False
        JOINT_TYPE = "cocoplus"

    demo_config = DemoConfig()

    # initialize model
    model = main.model.Model()

    logger = photobridge.utilities.get_logger(config.logging_path)

    for image_path in tqdm.tqdm(sorted(glob.glob(os.path.join(config.test_data_dir, "*.jpg")))):

        original_img, input_img, params = visualise.vis_util.preprocess_image(
            image_path, demo_config.ENCODER_INPUT_SHAPE[0])

        result = model.detect(input_img)

        cam = np.squeeze(result['cam'].numpy())[:3]
        vertices = np.squeeze(result['vertices'].numpy())
        joints = np.squeeze(result['kp2d'].numpy())
        joints = ((joints + 1) * 0.5) * params['img_size']

        renderer = visualise.trimesh_renderer.TrimeshRenderer()

        visualise.vis_util.visualize_v2(
            renderer=renderer,
            img=original_img,
            image_path=image_path,
            logger=logger,
            params=params,
            verts=vertices,
            cam=cam,
            joints=joints)


@invoke.task
def render_simple_mesh(_context):
    """
    Use trimesh to create and render a mesh of a simple cube
    """

    import pyglet
    pyglet.options["headless"] = True

    import cv2
    import numpy as np
    import trimesh
    import trimesh.viewer

    mesh = trimesh.creation.box()

    # Rotate mesh by 45 degrees
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(45), [0, 0, 1]))

    # Rotate mesh by 30 degrees towards the camera
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [1, 0, 0]))

    # Make every face of mesh different color
    mesh.visual.face_colors = trimesh.visual.random_color()

    # Draw mesh on a numpy array
    scene = mesh.scene()

    image_bytes = scene.save_image(resolution=[1920, 1080], background=(20, 200, 127, 1), visible=True)

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    cv2.imwrite("/logs/render.jpg", image)
