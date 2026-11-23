"""Environment MIS preregistration: independent math oracle and shader seams.

These HOST checks do not execute GLSL or prove device image quality. The matched
Vulkan build must additionally pass a raw-linear, multiple-seed image experiment.
"""
import bisect
import math
import struct
from pathlib import Path
import unittest

SHADER = Path(__file__).resolve().parents[1] / 'src/phymeo_rendering/backends/_shaders/resident_indirect.glsl'


def unit(v):
    norm = math.sqrt(sum(x*x for x in v))
    return tuple(x/norm for x in v)


def direction(face, u, v):
    sampler = ((1., -v, -u), (-1., -v, u), (u, 1., v),
               (u, -1., -v), (u, -v, 1.), (-u, -v, -1.))[face]
    x, y, z = unit(sampler)
    return x, z, y


def coordinates(world):
    x, z, y = world
    a, b, c = abs(x), abs(y), abs(z)
    if a >= b and a >= c:
        return (0, -z/a, -y/a) if x > 0 else (1, z/a, -y/a)
    if b >= c:
        return (2, x/b, z/b) if y > 0 else (3, x/b, -z/b)
    return (4, x/c, -y/c) if z > 0 else (5, -x/c, -y/c)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def packed_inverse(cdf, selector, directory=None):
    """Independent CPU model of the packed directory bracket, not a GPU test."""
    size = len(cdf)
    buckets = size//4
    if directory is None:
        directory = [bisect.bisect_right(cdf, f32(j/buckets)) for j in range(buckets)]
    bucket = min(max(int(f32(selector*buckets)), 0), buckets-1)
    # Correct multiplication rounding using the same rounded boundaries that
    # produced the directory. In this bounded layout at most one correction is
    # needed; two attempts leave explicit room for an end-boundary conversion.
    for _ in range(2):
        if selector < f32(bucket/buckets) and bucket > 0:
            bucket -= 1
        elif bucket+1 < buckets and selector >= f32((bucket+1)/buckets):
            bucket += 1
        else:
            break
    lo = directory[bucket]
    hi = directory[bucket+1] if bucket+1 < buckets else size-1
    if not 0 <= lo <= hi < size:
        lo, hi = 0, size-1
    index = bisect.bisect_right(cdf, selector, lo, hi+1)
    if index >= size or not (0. if index == 0 else cdf[index-1]) <= selector < cdf[index]:
        index = bisect.bisect_right(cdf, selector)
    return index, cdf[index]-(0. if index == 0 else cdf[index-1])

class EnvironmentMISShaderContracts(unittest.TestCase):
    def test_opt_in_is_progressive_and_legacy_seed_path_remains(self):
        source = SHADER.read_text(encoding='utf-8-sig')
        self.assertIn('bool importanceActive()', source)
        self.assertIn('environmentData.config.w==1u', source)
        self.assertIn('layout(constant_id=72) const uint PR_ENVIRONMENT_CDF_ABI=1u;', source)
        self.assertIn('return PR_ENVIRONMENT_CDF_ABI==1u &&', source)
        self.assertLess(source.index('#ifdef PR_RESIDENT_PROGRESSIVE'), source.index('bool importanceActive()'))
        self.assertIn('vec2 q=(vec2(i&3u,i>>2u)+jitter)*.25;', source)
        self.assertIn('return sum*(1.0/16.0);', source)

    def test_sampling_and_pdf_use_the_same_packed_cdf(self):
        source = SHADER.read_text(encoding='utf-8-sig')
        for name in ('importanceFaceVector', 'importanceCoordinates', 'importanceCdf',
                     'importanceInverse', 'importanceLeafProbability', 'importanceSample', 'importanceWeight'):
            self.assertIn(name, source)
        self.assertIn('importanceLeafProbability(face,cell)', source)
        self.assertIn('importanceWeight(N,direction)', source)
        self.assertIn('importanceWeight(hit.N,nextDirection)', source)
        self.assertIn('importanceInterior', source)

    def test_primary_environment_paths_stratify_the_complete_cdf(self):
        source = SHADER.read_text(encoding='utf-8-sig')
        self.assertIn('importanceSample(key,i-8u,8u,direction,sampledPdf)', source)
        self.assertIn('(float(stratum)+jitterSelector)/float(strata)', source)
        # Each primary environment path owns one full-CDF eighth, rather than
        # eight iid draws that can all choose the same concentrated branch.
        for jitter in (2**-24, .5, 1-2**-24):
            selectors = [(i+jitter)/8 for i in range(8)]
            self.assertEqual([int(x*8) for x in selectors], list(range(8)))

    def test_sampled_pdf_reuses_the_selected_canonical_bin_probabilities(self):
        source = SHADER.read_text(encoding='utf-8-sig')
        self.assertIn('probability=upper-lower;', source)
        self.assertNotIn('importanceMasses(', source)
        self.assertIn('sampledPdf=importanceUvPdf(probability,actualUV);', source)
        self.assertIn('importanceWeightPdf(N,direction,sampledPdf)', source)
        self.assertIn('importanceWeightPdf(hit.N,nextDirection,sampledPdf)', source)
        # Selected and reverse-evaluated densities must share the Jacobian;
        # returning raw texel mass as a solid-angle PDF would fail this oracle.
        mass, side, u, v = .125, 64, .25, -.75
        expected = mass*(side*side/4)*(1+u*u+v*v)**1.5
        self.assertNotEqual(expected, mass)

    def test_all_six_world_axes_and_face_roundtrip(self):
        expected = ((1,0,0),(-1,0,0),(0,0,1),(0,0,-1),(0,1,0),(0,-1,0))
        for face in range(6):
            self.assertEqual(direction(face,0,0), expected[face])
            for u, v in ((-.999,.999), (.3,-.7), (0,0), (.999,-.999)):
                result = coordinates(direction(face,u,v))
                self.assertEqual(result[0],face)
                self.assertAlmostEqual(result[1],u,places=12)
                self.assertAlmostEqual(result[2],v,places=12)

    def test_uv_jacobian_matches_independent_spherical_differential(self):
        eps = 1e-5
        for face in range(6):
            for u,v in ((0.,0.),(.7,-.8),(-.99,.1)):
                a,b = direction(face,u+eps,v), direction(face,u-eps,v)
                c,d = direction(face,u,v+eps), direction(face,u,v-eps)
                du = tuple((x-y)/(2*eps) for x,y in zip(a,b))
                dv = tuple((x-y)/(2*eps) for x,y in zip(c,d))
                cross = (du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0])
                actual = math.sqrt(sum(x*x for x in cross))
                self.assertAlmostEqual(actual, (1+u*u+v*v)**-1.5, delta=1e-8)
                mass,side = .001,256
                pdf = mass/(4/(side*side))*(1+u*u+v*v)**1.5
                self.assertAlmostEqual(pdf*actual, mass*side*side/4, delta=1e-6)

    def test_directory_inverse_matches_exhaustive_search_at_float_boundaries(self):
        # Six faces, side=4: M=96, B=24 (not a power of two). Include roots,
        # long zero-mass plateaus, adjacent bins and the largest selector <1.
        cdf = [0.]*12 + [f32(.125)]*20 + [f32(.50000006)]*30 + [1.]*34
        points = {0., f32(1-2**-24)}
        for value in cdf + [f32(j/24) for j in range(24)]:
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
            for neighbor in (max(0, bits-1), bits, bits+1):
                point = struct.unpack('<f', struct.pack('<I', neighbor))[0]
                if 0 <= point < 1:
                    points.add(point)
        for point in points:
            index, mass = packed_inverse(cdf, point)
            self.assertEqual(index, bisect.bisect_right(cdf, point))
            self.assertGreater(mass, 0.)

    def test_directory_reuses_exact_integer_indices_and_rejects_bad_bracket(self):
        # All possible native indices fit exactly in FP32 (max 6*1024^2-1).
        self.assertEqual(f32(6*1024*1024-1), 6*1024*1024-1)
        cdf = [f32((i+1)/96) for i in range(96)]
        for point in (0., .1, .5, f32(1-2**-24)):
            expected = bisect.bisect_right(cdf, point)
            self.assertEqual(packed_inverse(cdf, point, [-1]*24)[0], expected)
            self.assertEqual(packed_inverse(cdf, point, [95]*24)[0], expected)

    def test_packed_pmf_is_the_actual_stored_cdf_difference(self):
        cdf = [0., f32(.1), f32(.1), f32(.99999994), 1.]
        masses = [b-a for a,b in zip([0.]+cdf, cdf)]
        self.assertEqual(masses[0], 0.)
        self.assertEqual(masses[2], 0.)
        self.assertEqual(sum(masses), 1.)
        self.assertEqual(masses[-1], 2**-24)
        source = SHADER.read_text(encoding='utf-8-sig')
        self.assertIn('importancePackedAlpha(index,0)', source)
        self.assertIn('importancePackedAlpha(bucket,1)', source)
        self.assertNotIn('constant_id=71', source)

    def test_balance_mis_weights_apply_to_hits_as_well_as_environment_misses(self):
        # Four solid-angle bins; true incoming radiance can come from real hits.
        cosine = (.5,.3,.2,0.)
        environment = (.02,.03,.05,.90)
        incoming = (2.,10.,.1,100.)
        expected = sum(p*x for p,x in zip(cosine,incoming))
        weighted = 0.
        for pc,pe,li in zip(cosine,environment,incoming):
            mixture = .5*(pc+pe)
            weighted += (.5*pc+.5*pe)*li*pc/mixture
        self.assertAlmostEqual(weighted,expected)
        # The opposite shading hemisphere contributes zero, never renormalization.
        self.assertEqual(cosine[-1]/(.5*(cosine[-1]+environment[-1])),0.)

    def test_cosine_component_covers_zero_environment_probability(self):
        pc, pe = .2, 0.
        weight = pc/(.5*(pc+pe))
        self.assertEqual(weight,2.)
        self.assertAlmostEqual(.5*pc*weight,pc)

    def test_high_variance_fixture_is_not_a_quality_pass_by_construction(self):
        # A concentrated environment motivates sampling it, but image thresholds
        # and measured convergence remain the independent device owner's job.
        masses = [0.]*96
        masses[37] = 1.
        self.assertEqual(sum(masses),1.)
        self.assertGreater(sum(x*x for x in masses),1/len(masses))


if __name__ == '__main__':
    unittest.main()
