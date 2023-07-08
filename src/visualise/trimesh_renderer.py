import cv2
import icecream
import numpy as np
import pickle
import trimesh
import trimesh.transformations as trans

import main.config


def load_faces():
    c = main.config.Config()
    with open(c.SMPL_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model["f"].astype(np.int32)

class TrimeshRenderer:

    def __init__(self, img_size=(224, 224), focal_length=5.):
        self.h, self.w = img_size[0], img_size[1]
        self.focal_length = focal_length
        self.faces = load_faces()

    def __call__(self, verts, img=None, img_size=None, bg_color=None):
        """Render smpl mesh
        Args:
            verts: [6890 x 3], smpl vertices
            img: [h, w, channel] (optional)
            img_size: [h, w] specify frame size of rendered mesh (optional)
        """

        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h, w = img_size[0], img_size[1]
        else:
            h, w = self.h, self.w

        mesh = self.mesh(verts)
        scene = mesh.scene()

        mesh_image_bytes = scene.save_image(resolution=(w, h), background=bg_color, visible=True)

        mesh_image = cv2.imdecode(np.frombuffer(mesh_image_bytes, np.uint8), -1)

        if img is not None:

            overlay_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

            mesh_mask = mesh_image[:, :, 0] > 0
            overlay_image[mesh_mask, :3] = mesh_image[mesh_mask, :3]

            return cv2.cvtColor(overlay_image, cv2.COLOR_RGBA2RGB)

        return mesh_image

    def mesh(self, verts) -> trimesh.Trimesh:

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=self.faces,
            vertex_colors=[200, 255, 255, 255],
            face_colors=[0, 0, 0, 0],
            use_embree=False,
            process=False)

        # this transform is necessary to get correct image
        # because z axis is other way around in trimesh
        transform = trans.rotation_matrix(np.deg2rad(-180), [1, 0, 0], mesh.centroid)
        mesh.apply_transform(transform)

        return mesh

    def rotated(self, verts, deg, axis='y', img=None, img_size=None):
        rad = np.deg2rad(deg)

        if axis == 'x':
            mat = [rad, 0, 0]
        elif axis == 'y':
            mat = [0, rad, 0]
        else:
            mat = [0, 0, rad]

        around = cv2.Rodrigues(np.array(mat))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(new_v, img=img, img_size=img_size)
