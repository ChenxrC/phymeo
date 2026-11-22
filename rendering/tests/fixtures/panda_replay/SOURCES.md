# Classic Panda model inputs

The bundled `panda.urdf` is an offline, modified expansion of the **Franka Emika Panda** with its standard two-finger hand. Source release `franka_ros 0.10.1`, commit `30e598aa6fb703cc80a203481e6427f397337b4c`. The manufacturer's `franka_description/package.xml` explicitly licenses this description package, including its URDF, under Apache-2.0. The accompanying `LICENSE` and `NOTICE` are verbatim from that same commit, including the Franka Emika GmbH copyright. [Package license declaration](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/package.xml) · [Upstream notice](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/NOTICE)

| Input within that licensed package | Data included |
| --- | --- |
| [panda/panda.urdf.xacro](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/panda/panda.urdf.xacro) and [common/franka_robot.xacro](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/common/franka_robot.xacro) | Panda selection, hand mount and TCP configuration |
| [common/franka_arm.xacro](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/common/franka_arm.xacro) | Seven arm joint origins, axes, flange and structural collision proxies |
| [panda/joint_limits.yaml](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/panda/joint_limits.yaml) | Position limits (rad), velocity limits (rad/s) and effort limits (N·m) |
| [common/franka_hand.xacro](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/common/franka_hand.xacro) | Palm, TCP, two physical slides, equal-displacement mimic and all eight authored finger collision boxes |
| [common/inertial.yaml](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/common/inertial.yaml) | Masses, COM offsets and six symmetric tensor coefficients. The upstream file credits Gaz et al.'s identified parameters and distinguishes them from factory inertial specifications. |
| [common/utils.xacro](https://github.com/frankarobotics/franka_ros/blob/30e598aa6fb703cc80a203481e6427f397337b4c/franka_description/robots/common/utils.xacro) | Standard URDF cylinder plus two sphere representation of each coarse capsule |

## Changes for the portable model

The regeneration tool expands the pinned original Xacros twice using the real xacro processor. It keeps the inertial links from the Gazebo expansion and the nominal collision envelopes from the other expansion. It removes rendering meshes, environment-collision mesh references, Gazebo world/actuator plugins and transmissions, and names the finger boxes. The self-collision anticipation margin is set to zero. The distal hand safety envelope crosses the grasp aperture and is omitted. The proximal palm envelope and all eight original finger boxes remain. The URDF contains no third-party meshes, textures or package URIs. Phymeo verifies each original cylinder/sphere triplet before registering its equivalent single finite capsule.

The importer keeps the welded flange, hand and TCP frames accessible as named sites, accumulates the hand mass/inertia about the combined COM on joint 7, and rigidly aligns each dynamic link frame with its aggregate principal axes. Both finger translations remain independent dynamic coordinates, with the specified mimic mapping when constructing symmetric opening commands. `asset.configuration(arm_q, opening_m)` accepts the seven arm coordinates in radians and the total nominal opening in metres. Existing solvers receive nine individual scalar coordinates. Joint limit and passive dynamics tags in the URDF describe the asset. An actuator or stop program supplies physical loads.

`asset.sites["panda_leftpad"]` and `asset.sites["panda_rightpad"]` locate the inward face centers of the source rubber-tip boxes. Each box measures 0.0175 × 0.0152 × 0.0185 m. Its center is 0.00758 m from its finger's Y origin. The two opposing faces therefore have a signed normal spacing of commanded opening minus 0.00004 m. The source describes rigid box geometry. Material friction and compliance belong to the chosen experiment.

## Regeneration

Check out the pinned upstream commit, then run the following with `xacro==2.1.1` and `PyYAML==6.0.3` available in an isolated preparation environment. The tool verifies the bytes of all eight inputs before expansion. The installed Phymeo library reads the packaged XML directly and only needs its usual NumPy dependency.

```text
python src/phymeo/builders/panda_assets/regenerate.py --source /path/to/franka_ros/franka_description --output src/phymeo/builders/panda_assets/panda.urdf
```
