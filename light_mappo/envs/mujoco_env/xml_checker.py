import mujoco
import mujoco.viewer
import time
import os

xml_path = "./xml/collab_mirobot_for_paper.xml"

if not os.path.exists(xml_path):
    print(f"file not exist: {xml_path}")
    exit()

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

except Exception as e:
    print(f"Error loading model: {e}")