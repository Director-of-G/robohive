"""
    This env is modified from relocate_v0.py, with support for Mujoco3.0
"""

import numpy as np
from gym import utils
from mjrl.envs import mujoco3_env
import mujoco.viewer
import os
import re

ADD_BONUS_REWARDS = True
TAXEL_NAME_PATTERN = r"_T_r\d+c\d+$"
HAND_JOINT_NAME_PATTERN = r"^(WR|FF|MF|RF|LF|TH)J[0-9]$"

def get_actuator_id(sim, name):
    return mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def get_site_id(sim, name):
    return mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_SITE, name)

def get_body_id(sim, name):
    return mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_BODY, name)

def get_mocap_id(sim, name):
    return sim["model"].body_mocapid[mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_BODY, name)]

def get_all_taxel_joint_ids(sim):
    """
        The joints connecting taxels and knuckles contribute to nq.
        Thus their ids are needed to specifically reset the hand DoFs.
    """
    taxel_jpos_ids, taxel_jvel_ids = [], []
    taxel_sensor_names = []
    for i in range(sim["model"].nbody):
        body_name = mujoco.mj_id2name(sim["model"], mujoco.mjtObj.mjOBJ_BODY, i)
        if re.search(TAXEL_NAME_PATTERN, body_name):
            jid = sim["model"].body_jntadr[i]
            taxel_jpos_ids.append(sim["model"].jnt_qposadr[jid])
            taxel_jvel_ids.append(sim["model"].jnt_dofadr[jid])
            taxel_sensor_names.append(body_name.replace("T", "S"))
    taxel_jpos_ids.sort()
    taxel_jvel_ids.sort()
    print(f"Found {len(taxel_sensor_names)} taxel sensors in the hand model!")

    # parse taxel names
    knuckles = set()
    for name in taxel_sensor_names:
        knuckles.add(name.split("_")[0])
    knuckles = list(knuckles)
    print(f"Found {len(knuckles)} knuckles in the hand model!")

    taxel_meta = {}
    for name in knuckles:
        all_taxels = [n for n in taxel_sensor_names if n.startswith(name)]
        # _T_rxcx
        nrow = int(max([n.split("_", -1)[-1][1] for n in all_taxels]))+1
        ncol = int(max([n.split("_", -1)[-1][3] for n in all_taxels]))+1
        taxel_meta[name] = (nrow, ncol)

    return taxel_jpos_ids, taxel_jvel_ids, taxel_meta

def get_Adroit_joint_ids(sim):
    wrist_jpos_ids, hand_jpos_ids = [], []
    wrist_jnames = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
    for jname in wrist_jnames:
        jid = mujoco.mj_name2id(sim["model"], mujoco.mjtObj.mjOBJ_JOINT, jname)
        wrist_jpos_ids.append(sim["model"].jnt_qposadr[jid])
    wrist_jpos_ids.sort()

    for i in range(sim["model"].nq):
        jname = mujoco.mj_id2name(sim["model"], mujoco.mjtObj.mjOBJ_JOINT, i)
        if re.search(HAND_JOINT_NAME_PATTERN, jname):
            hand_jpos_ids.append(i)
    hand_jpos_ids.sort()

    return wrist_jpos_ids, hand_jpos_ids    

def read_taxel_data(sim, taxel_meta):
    """
        taxel_meta contains knuckle_name:(nrow, ncol) as key:value
    """
    taxel_data = {}
    for knuckle, size in taxel_meta.items():
        nrow, ncol = size
        panel_readings = np.zeros((nrow, ncol))
        for i in range(nrow):
            for j in range(ncol):
                taxel_sensor = f"{knuckle}_S_r{i}c{j}"
                panel_readings[i, j] = sim["data"].sensor(taxel_sensor).data
        taxel_data[knuckle] = panel_readings
    return taxel_data

class RelocateEnvV1(mujoco3_env.Mujoco3Env, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.taxel_meta = {}
        self.wrist_jpos_ids, self.hand_jpos_ids = [-1]*6, [-1]*24
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco3_env.Mujoco3Env.__init__(self, curr_dir+'/assets/DAPG_relocate_v1.xml', 5)
        
        # change actuator sensitivity
        self.sim["model"].actuator_gainprm[get_actuator_id(self.sim, "A_WRJ1"):get_actuator_id(self.sim, "A_WRJ0")+1,:3] = np.array([10, 0, 0])
        self.sim["model"].actuator_gainprm[get_actuator_id(self.sim, "A_FFJ3"):get_actuator_id(self.sim, 'A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim["model"].actuator_biasprm[get_actuator_id(self.sim, 'A_WRJ1'):get_actuator_id(self.sim, 'A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim["model"].actuator_biasprm[get_actuator_id(self.sim, 'A_FFJ3'):get_actuator_id(self.sim, 'A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = get_site_id(self.sim, "target")
        self.S_grasp_sid = get_site_id(self.sim, 'S_grasp')
        self.obj_bid = get_body_id(self.sim, 'Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

        self.taxel_jpos_ids, self.taxel_jvel_ids, self.taxel_meta = get_all_taxel_joint_ids(self.sim)
        self.non_taxel_jpos_ids = [i for i in list(range(self.sim["model"].nq)) if i not in self.taxel_jpos_ids]
        self.non_taxel_jvel_ids = [i for i in list(range(self.sim["model"].nv)) if i not in self.taxel_jvel_ids]

        self.wrist_jpos_ids, self.hand_jpos_ids = get_Adroit_joint_ids(self.sim)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos  = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        # qpos for hand (wrist dof 6 + hand dof 24)
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()[self.wrist_jpos_ids+self.hand_jpos_ids]
        obj_pos  = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.xpos[self.target_obj_sid].ravel()
        # taxel readings
        self.taxel_data = read_taxel_data(self.sim, self.taxel_meta)
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])
       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.data.xpos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.data.xpos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.data.xpos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.xpos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.data.xpos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        mujoco.mj_forward(self.sim["model"], self.sim["data"])
        return self.get_obs()
    
    def reset_object_properties(self):
        """
            Reset object mass, scale, ...
        """
        new_obj_mass = np.random.uniform(0.179594-0.08, 0.179594+0.08)
        new_obj_scale = 0.035
        self.model.body_mass[self.obj_bid] = new_obj_mass
        geom_id = self.model.body_geomadr[self.obj_bid]
        self.model.geom_size[geom_id, 0] = new_obj_scale
        print(f"reset obj mass={new_obj_mass}, scale={new_obj_scale}")

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[self.wrist_jpos_ids+self.hand_jpos_ids]
        obj_pos  = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']

        qp_all = np.zeros(self.sim["model"].nq,)
        qp_all[self.non_taxel_jpos_ids] = qp
        qv_all = np.zeros(self.sim["model"].nv,)
        qv_all[self.non_taxel_jvel_ids] = qv
        
        self.set_state(qp_all, qv_all)
        self.sim["model"].body_pos[self.obj_bid] = obj_pos
        self.sim["model"].site_pos[self.target_obj_sid] = target_pos
        mujoco.mj_forward(self.sim["model"], self.sim["data"])

    def mj_viewer_setup(self):
        self.viewer = mujoco.viewer.launch_passive(self.sim["model"], self.sim["data"])
        self.viewer.cam.azimuth = 90
        # self.sim.forward()
        self.viewer.cam.distance = 1.5

        # render hand collision mesh
        # self.viewer.opt.geomgroup[4] = 1

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
    
    # ====== for debug purposes ======
    def render_panel_for_debug(self, infos):
        from scipy.spatial.transform import Rotation as SciR
        for key, rtrans in infos.items():
            mocap_id = get_mocap_id(self.sim, key+"_mocap")
            self.sim["data"].mocap_pos[mocap_id] = rtrans[:3, 3]
            self.sim["data"].mocap_quat[mocap_id] = SciR.from_matrix(rtrans[:3, :3]).as_quat()[[3, 0, 1, 2]]
