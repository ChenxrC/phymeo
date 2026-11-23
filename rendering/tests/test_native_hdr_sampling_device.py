"""Opt-in HDR sampling regression on actual public GPU radiance.

PHYMEO_NATIVE_HDR_SAMPLING_DLL enables this test; companion options are
PHYMEO_NATIVE_HDR_SAMPLING_SHADERS, _DEVICE_INDEX (default 1), _VALIDATION
(default 1), and _OUTPUT (optional NEW evidence directory).

Frozen before device results: 64x64 original quad; central 24x24 ROI; four
seeds [0,1,7,19], ONE 16-sample batch each. A 512x256 actual RGBE panorama has
an 16x12-texel bright rectangle (RGB 64,16,4) against 1/256 grey. Its cosine
hit probability is only a few per thousand. For the old 4x4 cosine estimator,
independent angular integration computes the stratified variance directly:
Var(mean16) = sum(E[L^2 * 1_stratum])/16 - sum(E[L * 1_stratum]^2).
The fixture must predict relative noise >1.2; the registered device gate is
<=0.60, not a threshold fitted to the old or new seed outputs. Relative pooled
mean error must also be <=12%. Four seeds x576 pixels supply 2304 observations;
pixels are not asserted IID, and no confidence interval is claimed.

The CPU oracle integrates the ACTUAL declared L0 cube payload (public input
preparation, hash-bound to the upload), using independently implemented
bilinear face lookup and exact cell solid angles at 128/256 subdivisions.
It uses neither SH9 nor a device sample sequence. A black dielectric control
has the same F0=.04, roughness and view as white, eliminating the actual
specular term without approximating it. Their raw-radiance difference must
be .7*.96 times independently integrated incoming diffuse radiance.

Budget: 11 public 64x64 captures across these three methods; one raw-mean read
per capture; no hidden convergence loop, physical stepping or adaptive seed.
"""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.backends.native import NativeBackend
from phymeo_rendering.cameras import Camera
from phymeo_rendering.environment import RadianceEnvironment, prepare_native_environment
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene
from phymeo_rendering.studio import StudioLighting
from test_native_texture_device import quad

SIZE = 64
ROI = (slice(20, 44), slice(20, 44))
SEEDS = (0, 1, 7, 19)
BACKGROUND = 1.0 / 256.0
RHO = .7 * .96
SENSORS = (Channel.FORWARD_DEPTH, Channel.INSTANCE_ID, Channel.VALID)


def authored_hdr(path):
    """Exact linear RGBE values; no image-generation or external asset input."""
    low, bright = bytes((128, 128, 128, 121)), bytes((128, 32, 8, 135))
    body = bytearray()
    for y in range(256):
        body.extend(low*248 + bright*16 + low*248 if 80 <= y < 92 else low*512)
    data = b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 256 +X 512\n' + body
    path.write_bytes(data)
    return RadianceEnvironment(str(path), hashlib.sha256(data).hexdigest(), face_size=64)


def cube_integral(payload, normal, subdivisions):
    """Independent world-Z-up angular quadrature of native sampler XZY L0.

    The bright support lies strictly inside the +X cube face. Other face edges
    are uniform grey, so clamped intra-face bilinear lookup is also the exact
    seamless lookup there. No product direction/sampler/integrator is called.
    """
    import numpy as np
    side = payload.side
    cube = np.frombuffer(payload.rgba_f32le, '<f4', count=6*side*side*4).reshape(6, side, side, 4)[..., :3]
    edges = np.linspace(-1., 1., subdivisions+1)
    centers = (edges[:-1]+edges[1:])*.5
    s, t = np.meshgrid(centers, centers)
    one = np.ones_like(s)
    # Vulkan face order +X,-X,+Y,-Y,+Z,-Z; sampler coordinates = world.xzy.
    faces = ((one, -s, -t), (-one, s, -t), (s, t, one),
             (s, -t, -one), (s, one, -t), (-s, -one, -t))
    corner_x, corner_y = np.meshgrid(edges, edges)
    primitive = np.arctan2(corner_x*corner_y, np.sqrt(1+corner_x**2+corner_y**2))
    omega = primitive[1:, 1:] - primitive[:-1, 1:] - primitive[1:, :-1] + primitive[:-1, :-1]
    if abs(float(omega.sum())*6-4*math.pi) > 1e-10:
        raise ValueError('independent cube solid-angle partition failed')
    u = (s+1)*side*.5-.5
    v = (t+1)*side*.5-.5
    ix, iy = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u-ix)[..., None], (v-iy)[..., None]
    ax, bx = np.clip(ix, 0, side-1), np.clip(ix+1, 0, side-1)
    ay, by = np.clip(iy, 0, side-1), np.clip(iy+1, 0, side-1)
    integral = np.zeros(3)
    second = np.zeros(3)
    stratum_means = np.zeros((16, 3))
    support = []
    n = np.asarray(normal, dtype=float)
    for face, coordinates in enumerate(faces):
        direction = np.stack(coordinates, axis=-1)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        rgb = ((1-fy)*((1-fx)*cube[face, ay, ax]+fx*cube[face, ay, bx]) +
               fy*((1-fx)*cube[face, by, ax]+fx*cube[face, by, bx]))
        weight = np.maximum(direction@n, 0.)*omega/math.pi
        integral += np.sum(rgb*weight[..., None], axis=(0, 1))
        second += np.sum(rgb*rgb*weight[..., None], axis=(0, 1))
        if tuple(normal) == (0., 0., 1.):
            # Mathematical cosine disk strata, independent of shader hash/RNG.
            radial = np.clip(1-direction[..., 2]**2, 0., np.nextafter(1., 0.))
            azimuth = np.mod(np.arctan2(direction[..., 1], direction[..., 0])/(2*math.pi), 1.)
            strata = np.floor(4*radial).astype(int) + 4*np.floor(4*azimuth).astype(int)
            for index in range(16):
                mask = strata == index
                stratum_means[index] += np.sum(rgb[mask]*weight[mask, None], axis=0)
        selected = (rgb[..., 0] > BACKGROUND+1e-7) & (weight > 0)
        if np.any(selected):
            support.append(direction[selected])
    variance = second/16 - np.sum(stratum_means**2, axis=0)
    return {'incoming': integral, 'cosine_stratified_variance': np.maximum(variance, 0.),
            'bright_support': np.concatenate(support) if support else np.empty((0, 3))}


@unittest.skipUnless(os.environ.get('PHYMEO_NATIVE_HDR_SAMPLING_DLL'), 'set dedicated PHYMEO_NATIVE_HDR_SAMPLING_DLL for actual GPU HDR sampling')
class NativeHdrSamplingDevice(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = TemporaryDirectory(prefix='phymeo-hdr-sampling-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        target = os.environ.get('PHYMEO_NATIVE_HDR_SAMPLING_OUTPUT')
        self.evidence = Path(target)/self._testMethodName if target else None
        if self.evidence:
            self.evidence.mkdir(parents=True, exist_ok=False)
        quad(self.root/'receiver.gltf')
        self.environment = authored_hdr(self.root/'bright.hdr')
        self.payloads = {}
        self.records = []
        self.addCleanup(self.write_evidence)

    def write_evidence(self):
        if self.evidence:
            (self.evidence/'receipt.json').write_text(json.dumps({
                'test': self.id(), 'records': self.records,
                'raw_space': 'scene-linear-before-exposure',
                'limits': 'one-bounce environment regression; narrow authored HDR, four fixed seeds; no universal variance guarantee'
            }, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            (self.evidence/'bright.hdr').write_bytes((self.root/'bright.hdr').read_bytes())

    def payload(self, yaw):
        if yaw not in self.payloads:
            environment = replace(self.environment, yaw_degrees=yaw)
            data = prepare_native_environment(environment)
            self.payloads[yaw] = (environment, data)
        return self.payloads[yaw]

    def oracle(self, normal, yaw=0.):
        np = self.np
        _, payload = self.payload(yaw)
        coarse = cube_integral(payload, normal, 128)
        fine = cube_integral(payload, normal, 256)
        self.assertLess(float(np.linalg.norm(coarse['incoming']-fine['incoming'])/np.linalg.norm(fine['incoming'])), .003,
                        'independent angular quadrature must converge before device comparison')
        return fine

    def capture(self, *, albedo=.7, seed=0, normal_x=False, yaw=0., blocked=False):
        np = self.np
        q = (0., math.sqrt(.5), 0., math.sqrt(.5)) if normal_x else (0., 0., 0., 1.)
        scene = VisualScene('hdr-sampling-regression', (Mesh('quad',str(self.root/'receiver.gltf')),),
                            (Material('receiver',(albedo,)*3,0.,1.), Material('absorber',(0.,)*3,1.,.04)),
                            (Instance(1,'quad','receiver','receiver'),) +
                            ((Instance(2,'quad','absorber','block'),) if blocked else ()), ())
        nodes = (NodePose('receiver',Pose(orientation_xyzw=q)),)
        if blocked:
            self.assertFalse(normal_x)
            nodes += (NodePose('block',Pose((2.7,0.,-1.5),(0.,math.sqrt(.5),0.,math.sqrt(.5)))),)
        key = FrameKey(scene.key,'world',0,0,0,'authored')
        frame = SceneFrame(key,0.,nodes)
        camera = Camera('front',key,Pose(orientation_xyzw=q),SIZE,SIZE,60.,60.,32.,32.,.1,10.)
        request = CaptureRequest(camera,RenderPath.RASTER,tuple(Channel))
        environment, payload = self.payload(yaw)
        library = Path(os.environ['PHYMEO_NATIVE_HDR_SAMPLING_DLL']).resolve()
        shaders = Path(os.environ.get('PHYMEO_NATIVE_HDR_SAMPLING_SHADERS',str(library.parent))).resolve()
        backend = NativeBackend(library_path=library,shader_directory=shaders,
                                device_index=int(os.environ.get('PHYMEO_NATIVE_HDR_SAMPLING_DEVICE_INDEX','1')),
                                validation=os.environ.get('PHYMEO_NATIVE_HDR_SAMPLING_VALIDATION','1')=='1',
                                resident=True,shadow_mode='ray_query',indirect_mode='diffuse_bounce',
                                indirect_seed=seed,progressive=True,indirect_bounces=1,spatial_filter='none',
                                studio=StudioLighting(panel_strength=0.,environment_radiance=0.,environment_map=environment))
        record = {'albedo':albedo,'seed':seed,'normal':'+X' if normal_x else '+Z','yaw':yaw,'blocked':blocked,
                  'library':str(library),'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'shaders':str(shaders)}
        self.records.append(record)
        owner = backend.open_scene(scene)
        try:
            shot = owner.capture(frame,request)
            planes = {plane.channel:plane.data for plane in shot.planes}
            packet = owner.read_linear_mean()
            self.assertEqual((packet['width'],packet['height'],packet['format'],packet['space'],packet['batches'],packet['primary_samples_per_pixel']),
                             (64,64,'RGBA32F_LE','scene-linear-before-exposure',1,16))
            self.assertIsInstance(packet['data'],bytes)
            self.assertEqual(len(packet['data']),64*64*16)
            raw = np.frombuffer(packet['data'],'<f4').reshape(64,64,4).copy()
            self.assertTrue(np.isfinite(raw).all())
            ids = np.frombuffer(planes[Channel.INSTANCE_ID],'<u4').reshape(64,64)
            depth = np.frombuffer(planes[Channel.FORWARD_DEPTH],'<f4').reshape(64,64)
            valid = np.frombuffer(planes[Channel.VALID],'u1').reshape(64,64)
            self.assertTrue((ids[ROI]==1).all())
            self.assertTrue((valid[ROI]==1).all())
            self.assertLess(float(np.max(np.abs(depth[ROI]-2.))),2e-6)
            diag = owner.diagnostics()
            self.assertEqual(diag['progressive']['batches'],1)
            self.assertEqual(diag['environment']['rgba_sha256'],hashlib.sha256(payload.rgba_f32le).hexdigest())
            self.assertEqual(diag['indirect']['samples_per_pixel'],16)
            record.update(diagnostics=diag,raw_sha256=hashlib.sha256(packet['data']).hexdigest(),
                          plane_sha256={channel.value:hashlib.sha256(data).hexdigest() for channel,data in planes.items()})
            if self.evidence:
                np.savez_compressed(self.evidence/f'capture-{len(self.records):02d}.npz',rgba32f=raw,instance_id=ids,depth=depth,valid=valid,
                                    rgb8=np.frombuffer(planes[Channel.RGB],'u1').reshape(64,64,3))
            return raw[ROI][..., :3].astype(np.float64), {channel:planes[channel] for channel in SENSORS}
        finally:
            owner.close()
            info = owner.diagnostics()['device']
            record['closed'] = info
            for name in ('validation_errors','validation_warnings','live_contexts','live_frames','live_resources'):
                self.assertEqual(info[name],0,name)
            memory = info['resident_telemetry']
            self.assertTrue(memory['final_after_destroy'])
            self.assertEqual(memory['device_local_bytes_current'],0)
            self.assertEqual(memory['host_visible_bytes_current'],0)
            self.assertEqual(memory['allocation_count'],memory['free_count'])

    def record_metrics(self, value):
        self.records.append({'metrics':value})
        print(json.dumps({'test':self.id(),**value},allow_nan=False),flush=True)

    def test_narrow_bright_hdr_absolute_diffuse_and_low_sample_seed_variance(self):
        np = self.np
        reference = self.oracle((0.,0.,1.))
        expected = reference['incoming']*RHO
        theoretical = math.sqrt(float(reference['cosine_stratified_variance'].mean()))/math.sqrt(float(np.mean(reference['incoming']**2)))
        self.assertGreater(theoretical,1.2,'fixture no longer separates the old sparse cosine estimator')
        black,sensors = self.capture(albedo=0.)
        samples = []
        for seed in SEEDS:
            white,actual_sensors = self.capture(seed=seed)
            self.assertEqual(actual_sensors,sensors)
            samples.append(white-black)
        samples = np.stack(samples)
        mean = samples.mean(axis=(0,1,2))
        relative_bias = float(np.linalg.norm(mean-expected)/np.linalg.norm(expected))
        seed_noise = float(np.sqrt(np.mean(np.var(samples,axis=0,ddof=1)))/np.sqrt(np.mean(expected**2)))
        relative_rmse = float(np.sqrt(np.mean((samples-expected)**2))/np.sqrt(np.mean(expected**2)))
        self.record_metrics({'oracle_diffuse_rgb':expected.tolist(),'actual_diffuse_rgb':mean.tolist(),
                             'relative_bias':relative_bias,'relative_cross_seed_noise':seed_noise,'relative_rmse':relative_rmse,
                             'old_stratified_cosine_predicted_relative_noise':theoretical,'seeds':list(SEEDS),
                             'samples_per_pixel_per_seed':16,'roi_pixels_per_seed':576,
                             'gates':{'seed_noise_max':.60,'relative_mean_error_max':.12,'relative_rmse_max':.65}})
        self.assertLessEqual(seed_noise,.60,'narrow HDR cross-seed variance exceeds the preregistered 16-spp budget')
        self.assertLessEqual(relative_bias,.12,'HDR diffuse mean differs from independent cube cosine integration')
        self.assertLessEqual(relative_rmse,.65,'low noise alone must not hide a biased HDR estimator')

    def test_world_axis_and_yaw_match_independent_hemisphere_integral(self):
        np = self.np
        outcomes = []
        sensors = None
        for yaw in (0.,180.):
            expected = self.oracle((1.,0.,0.),yaw)['incoming']*RHO
            black,black_sensors = self.capture(albedo=0.,normal_x=True,yaw=yaw)
            white,white_sensors = self.capture(normal_x=True,yaw=yaw)
            self.assertEqual(black_sensors,white_sensors)
            if sensors is not None:self.assertEqual(sensors,white_sensors)
            sensors = white_sensors
            actual = (white-black).mean(axis=(0,1))
            relative = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            outcomes.append({'yaw':yaw,'expected_rgb':expected.tolist(),'actual_rgb':actual.tolist(),'relative_error':relative})
            self.assertLessEqual(relative,.15,'world +X / positive-yaw HDR hemisphere differs')
        self.assertGreater(outcomes[0]['expected_rgb'][0],30*outcomes[1]['expected_rgb'][0])
        self.assertGreater(outcomes[0]['actual_rgb'][0],20*outcomes[1]['actual_rgb'][0])
        self.record_metrics({'axis_cases':outcomes,'relative_error_limit':.15})

    def test_actual_black_blocker_removes_the_bright_patch_without_hiding_receiver(self):
        np = self.np
        reference = self.oracle((0.,0.,1.))
        rays = reference['bright_support']
        self.assertGreater(len(rays),0)
        self.assertTrue((rays[:,0]>0).all())
        # Independent ray/finite-plane proof at all receiver ROI corner points.
        # The blocker occupies x=.7, y in [-1,1], z in [-2.5,-.5]. The full
        # rectangle's projected extrema occur at these ROI corners for each ray.
        for x in (20,43):
            for y in (20,43):
                point = np.array(((x+.5-32)/60*2,-(y+.5-32)/60*2,-2.))
                distance = (.7-point[0])/rays[:,0]
                hit = point + distance[:,None]*rays
                self.assertTrue((distance>0).all())
                self.assertLess(float(np.max(np.abs(hit[:,1]))),.99)
                self.assertGreater(float(hit[:,2].min()),-2.49)
                self.assertLess(float(hit[:,2].max()),-.51)
        black,black_sensors = self.capture(albedo=0.,blocked=True)
        white,white_sensors = self.capture(blocked=True)
        self.assertEqual(black_sensors,white_sensors)
        difference = white-black
        mean = difference.mean(axis=(0,1))
        # The low-roughness black metal has zero diffuse/F0; its reflected
        # direction points toward -X, away from the +X bright patch. Schlick
        # grazing terms are not claimed identically zero. This generous bound
        # allows the low background and a residual terminal specular response;
        # it does not claim that the blocker removes every sky direction.
        limit = 3*RHO*BACKGROUND + 2e-5
        self.record_metrics({'blocked_diffuse_rgb':mean.tolist(),'unblocked_oracle_diffuse_rgb':(reference['incoming']*RHO).tolist(),
                             'blocked_mean_upper_bound':limit,'blocker_primary_roi_unchanged':True})
        self.assertGreaterEqual(float(difference.min()),-2e-5)
        self.assertLessEqual(float(mean.max()),limit,'visible HDR energy leaked through the actual black blocker')
        self.assertLess(float(mean[0]),.10*float(reference['incoming'][0]*RHO))


if __name__ == '__main__':
    unittest.main()
